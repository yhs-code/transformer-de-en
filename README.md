# Transformer-de-en

#### 介绍
学习并实现Transformer从德语到英语的翻译任务，采用wmt16语料库

#### 软件架构
训练代码：transformer_de_en_autoDL.py
推理代码：推理.py

#### 操作流程
1、在transformer-de-en目录下先执行以下命令对数据集进行BPE分词：sh data_multi30k.sh wmt16 wmt16_cut de en
2、分词结束后执行以下指令开启训练：python3 transformer_de_en_autoDL.py
3、训练结束后执行：python3 推理.py即可执行使用翻译任务

#### 使用说明

1. 翻译设置的最大长度是128
2. 如果要修改相关配置重新进行训练，请打开transformer_de_en_autoDL.py文件后修改config字典（通过关键字搜索：config = ）

