# @Author:sunshine
# @Time  : 2020/5/8 上午9:25

"""
工具包
"""

import json


def generate_labels():
    """生成三个label文件"""
    domains, intents, slots = set(), set(), set()
    data = json.load(open('data/train.json', 'r', encoding='utf-8'))
    for item in data:
        domain = item['domain']
        intent = item['intent']
        slot = list(item['slots'].keys())

        domains.add(domain)
        intents.add(intent)
        slots.update(slot)

    # 生成label
    domain_label2id = dict(zip(domains, range(len(domains))))
    domain_id2label = {v: k for k, v in domain_label2id.items()}

    intent_label2id = dict(zip(intents, range(len(intents))))
    intent_id2label = {v: k for k, v in intent_label2id.items()}

    slot_tmp = []
    for i in slots:
        slot_tmp.append('B-' + i)
        slot_tmp.append('I-' + i)

    slot_label2id = {"O": 0}
    slot_label2id.update(dict(zip(slot_tmp, range(1, len(slot_tmp) + 1))))
    slot_id2label = {v: k for k, v in slot_label2id.items()}

    dumps = [domain_label2id, domain_id2label, intent_label2id, intent_id2label, slot_label2id, slot_id2label]
    json.dump(dumps, open('data/labels.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    generate_labels()
