### NLU的keras实现
意图识别+领域识别+槽填充的联合模型. 使用electra来抽取特征.  


##### 1. 执行方法
* precess_data.py: 生成labels
* joint_classifier_and_crf.py: 模型训练/测试/预测

#### 2. 数据
数据来源于SMP2019 ECDT
* train.json: 训练数据(验证数据从里面拆分)
* previous: SMP前两届提供的数据
* rule.xlsx: 数据分析生成的excel表

#### 3. electra模型
哈工大版本的electra: https://github.com/ymcui/Chinese-ELECTRA  
默认下载的中文electra模型缺少配置文件.  
这里提供一份配置好的模型文件: 
链接: https://pan.baidu.com/s/1svWQHfsA6GtLVZUa2b0uwQ 提取码: jq7x


参考:
1. https://github.com/OnionWang/SMP2019-ECDT-NLU