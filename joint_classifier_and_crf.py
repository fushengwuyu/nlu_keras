# @Author:sunshine
# @Time  : 2020/5/8 上午9:25

"""
联合模型: 意图识别, 领域识别, 槽填充
bert/electra + softmax|softmax|crf
"""

from bert4keras.tokenizers import load_vocab, Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.layers import ConditionalRandomField
from bert4keras.optimizers import AdaFactor
from keras.layers import Lambda, Dense, Input
from keras.models import Model
import keras.backend as K
import keras
import json
import random
from tqdm import tqdm
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

config_path = '/home/chenbing/pretrain_models/electra/chinese_electra_base_L-12_H-768_A-12/electra_config.json'
checkpoint_path = '/home/chenbing/pretrain_models/electra/chinese_electra_base_L-12_H-768_A-12/electra_base'
vocab_path = '/home/chenbing/pretrain_models/electra/chinese_electra_base_L-12_H-768_A-12/vocab.txt'

batch_size = 32
intent_num = 23
domain_num = 29
slot_num = 119
lr_multiplier = 100
domain_label2id, domain_id2label, intent_label2id, intent_id2label, slot_label2id, slot_id2label = json.load(
    open('data/labels.json', 'r', encoding='utf-8')
)

# 加载数据
data = json.load(open('data/train.json', 'r', encoding='utf-8'))
random.shuffle(data)
valid_data = data[:len(data) // 8]
train_data = data[len(data) // 8:]

# 加载并精简词表,建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=vocab_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
)

tokenizer = Tokenizer(token_dict)


# 数据迭代器
class MyDataGenerator(DataGenerator):
    def __init__(self, data, batch_size=32, buffer_size=None):
        super(MyDataGenerator, self).__init__(data, batch_size, buffer_size)

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, Y1, Y2, Y3 = [], [], [], [], []

        for is_end, item in self.sample(random):

            text = item['text']
            token_ids, segment_ids = tokenizer.encode(first_text=text.lower())

            y1 = domain_label2id.get(item['domain'])
            y2 = intent_label2id.get(item['intent'])

            tokens = tokenizer.tokenize(text.lower())[1:-1]
            tag_labels = [0] * len(tokens)  # 先排除[cls], [sep]的影响
            for k, v in item['slots'].items():
                v_tokens = tokenizer.tokenize(v)[1: -1]
                len_v = len(v_tokens)
                for i in range(len(tokens) - len_v + 1):
                    if tokens[i: i + len_v] == v_tokens:
                        tag_labels[i] = slot_label2id.get('B-' + k)
                        tag_labels[i + 1: i + len_v] = [slot_label2id.get('I-' + k)] * (len_v - 1)
                        break

            y3 = [0] + tag_labels + [0]

            if len(token_ids) != len(y3):
                print(text)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            Y1.append([y1])
            Y2.append([y2])
            Y3.append(y3)

            if len(batch_token_ids) == self.batch_size or is_end:
                # padding
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)

                Y1 = sequence_padding(Y1)
                Y2 = sequence_padding(Y2)
                Y3 = sequence_padding(Y3)

                yield [batch_token_ids, batch_segment_ids], [Y1, Y2, Y3]
                batch_token_ids, batch_segment_ids, Y1, Y2, Y3 = [], [], [], [], []


# 补充输入
# intent_labels = Input(shape=(intent_num,), name='intent_labels')
# domain_labels = Input(shape=(domain_num,), name='domain_labels')
# slot_labels = Input(shape=(None, slot_num), name='slot_labels')

# 搭建网络
electra_model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='electra',
    return_keras_model=False
)

classify_output = Lambda(lambda x: x[:, 0], name='CLS-token')(electra_model.model.output)

# 领域识别模型
domain_output = Dense(domain_num, activation='softmax', kernel_initializer=electra_model.initializer,
                      name='domain_classifier')(classify_output)
domain_model = Model(electra_model.input, domain_output)

# 意图识别模型
intent_output = Dense(intent_num, activation='softmax', kernel_initializer=electra_model.initializer,
                      name='intent_classifier')(classify_output)
intent_model = Model(electra_model.model.input, intent_output)

# 槽填充
x = Dense(slot_num)(electra_model.model.output)
CRF = ConditionalRandomField(lr_multiplier=lr_multiplier, name='slots_tagger')
slot_output = CRF(x)
slot_model = Model(electra_model.input, slot_output)

# 训练模型
train_model = Model(
    # electra_model.input + [intent_labels, domain_labels, slot_labels],
    electra_model.model.input,
    [domain_output, intent_output, slot_output]
)

# # mask
# mask = electra_model.model.get_layer('Embedding-Token').output_mask
# mask = K.cast(mask, K.floatx())
#
# # intent_loss
# intent_loss = K.sparse_categorical_crossentropy(intent_labels, intent_output)
# intent_loss = K.mean(intent_loss, 2)
# intent_loss = K.sum(intent_loss * mask) / K.sum(mask)
#
# # domain_loss
# domain_loss = K.sparse_categorical_crossentropy(domain_labels, domain_output)
# domain_loss = K.mean(domain_loss, 2)
# domain_loss = K.sum(domain_loss * mask) / K.sum(mask)
#
# # slot_loss
# slot_loss = CRF.sparse_loss(slot_labels, slot_output)
# slot_loss = K.sum(K.mean(slot_loss, 3), 2)
# slot_loss = K.sum(slot_loss * mask) / K.sum(mask)
#
# # 组合loss
# loss = 0.2 * intent_loss + 0.2 * domain_loss + 0.6 * slot_loss

# train_model.add_loss(loss)

loss = {
    "domain_classifier": 'sparse_categorical_crossentropy',
    "intent_classifier": 'sparse_categorical_crossentropy',
    'slots_tagger': CRF.sparse_loss
}

weight = {
    "domain_classifier": 1.0,
    "intent_classifier": 1.0,
    'slots_tagger': 3.0
}

optimizer = AdaFactor(learning_rate=1e-3)
train_model.compile(optimizer=optimizer, loss=loss, loss_weights=weight)
train_model.summary()


def viterbi_decode(nodes, trans):
    """Viterbi算法求最优路径
    其中nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
    labels = np.arange(slot_num).reshape((1, -1))
    scores = nodes[0].reshape((-1, 1))
    scores[1:] -= np.inf  # 第一个标签必然是0
    paths = labels
    for l in range(1, len(nodes)):
        M = scores + trans + nodes[l].reshape((1, -1))
        idxs = M.argmax(0)
        scores = M.max(0).reshape((-1, 1))
        paths = np.concatenate([paths[:, idxs], labels], 0)
    return paths[:, scores[:, 0].argmax()]


def decode(text, tags):
    """解码"""
    result = []
    tokens = tokenizer.tokenize(text)[1:-1]

    text_tags_lsit = list(zip(tokens, tags))
    for index, (s, t) in enumerate(text_tags_lsit):
        if len(t) > 1:  # 不为O
            head, tag = t.split('-')
            if head == 'B':
                tmp_str = s
                for index1, (s1, t1) in enumerate(text_tags_lsit[index + 1:]):
                    if t1 == 'I-' + tag:
                        tmp_str += s1
                    else:
                        break
                result.append((tag, tmp_str))
    return result


def extract_item(text):
    """
    提取模型识别结果
    :param text:
    :return: [intent_output, domain_output, slot_output] , slot_output: key1:value1, ...
    """
    token_ids, segment_ids = tokenizer.encode(first_text=text.lower())

    # intent
    intent_preds_id = intent_model.predict([[token_ids], [segment_ids]])[0].argmax()
    intent_pred = intent_id2label.get(str(intent_preds_id))

    # domain
    domain_pred_id = domain_model.predict([[token_ids], [segment_ids]])[0].argmax()
    domain_pred = domain_id2label.get(str(domain_pred_id))

    # slot
    slot_pred_tags = slot_model.predict([[token_ids], [segment_ids]])[0]
    raw = viterbi_decode(slot_pred_tags, K.eval(CRF.trans))[1: -1]  # 减去收尾: cls, sep
    raw = [slot_id2label.get(str(i)) for i in raw]
    slot_pred = decode(text, raw)
    result = [intent_pred, domain_pred]
    if slot_pred:
        result = result + slot_pred
    return result


def evalute(valid_data):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for item in tqdm(valid_data):
        text = item['text']
        R = set(extract_item(text))
        T_slot = [(k, v) for k, v in item['slots'].items()]
        T = set([item['intent'], item['domain']] + T_slot)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Z, Y / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型"""

    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evalute(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            train_model.save('best_model.h5')

        print(
            'f1: %0.5f, precision: %0.5f, recall: %.5f, best f1: %.5f  \n' % (f1, precision, recall, self.best_val_f1))


if __name__ == '__main__':

    train_model.load_weights('best_model.h5')

    # train_D = MyDataGenerator(train_data, batch_size)
    # evalutor = Evaluator()
    #
    # train_model.fit_generator(train_D.forfit(), epochs=500, steps_per_epoch=len(train_D), callbacks=[evalutor])
    # train_model.save('best_model.h5')

    # train_model.load_weights('best_model.h5')
    # text = '打开江苏卫视'
    # result = extract_item(text)
    # i = set(result)
    # print(result)

    f1, precision, recall = evalute(valid_data)
    print(
        'f1: %0.5f, precision: %0.5f, recall: %.5f \n' % (f1, precision, recall))