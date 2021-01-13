#!/usr/bin/env python
# coding: utf-8

##################################################

#    @Date: 2021-01-13 17:43:15
#    @FileDescription: 模型文件
#    @FirstEditors: 大卫
#    @LastEditors: 大卫
#    @Department: None
#    @LastEditTime: 2021-01-13 17:43:16

##################################################

import os

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BertModel


class Config(object):
    """配置参数"""
    def __init__(self):
        current_path = os.path.dirname(__file__)
        self.model_name = 'pytorch_model.bin'
        self.bert_path = os.path.join(current_path, "models/bert-chinese")
        self.train_file = os.path.join(current_path, "datasets/THUCNews/train.txt")  # 训练数据路径
        self.test_file = os.path.join(current_path, "datasets/THUCNews/test.txt")    # 测试数据路径
        self.save_path = os.path.join(current_path, "models/finetuned_model")  # 模型训练结果保存路径

        self.num_classes = 10                    # 类别数(按需修改)
        self.hidden_size = 768                   # 隐藏层输出维度
        self.hidden_dropout_prob = 0.1           # dropout比例
        self.batch_size = 128                    # mini-batch大小
        self.max_len = 32                        # 句子的最长padding长度

        self.epochs = 3                          # epoch数
        self.bert_learning_rate = 2e-5                  # BERT学习率
        self.fc_learning_rate = 2e-5                    # FC学习率
        self.use_cuda = True                     # 是否使用 GPU 训练
        self.device_id = 0                       # 如果使用 GPU，选择 GPU 编号，默认0

        # self.fp16 = False
        # self.fp16_opt_level = 'O1'
        # self.gradient_accumulation_steps = 1
        # self.warmup_ratio = 0.06
        # self.warmup_steps = 0
        # self.max_grad_norm = 1.0
        # self.adam_epsilon = 1e-8
        # self.class_list = class_list                              # 类别名单
        # self.require_improvement = 1000                           # 若超过1000batch效果还没提升，则提前结束训练


# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
config = Config()


class BertForSentClfWithFeatures(nn.Module):
    """
    BERT句子分类模型。本模型增加了额外特征输入 add_feature，可以依葫芦画瓢添加更多手动特征。

    想要对短文本进行分类，短文本会进入BERT模型进行 Embdedding 操作。
    add_feature_0: 属于额外的特征，一般在 BERT Embedding 后的 768 维向量拼接。
    tensor(add_feature_0)，也就是变成了 768 + 1 = 769 维，再过一个 线形层 + softmax 输出分类结果。
    """
    def __init__(self):
        super(BertForSentClfWithFeatures, self).__init__()
        self.num_labels = config.num_classes
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)  # 原生模型，768 维度 -> label数
        self.classifier_add = nn.Linear(config.hidden_size + 1, config.num_classes)  # 这里维度 +1 放入add_features

    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor,
                token_type_ids: Tensor = None,
                position_ids: Tensor = None,
                head_mask: Tensor = None,
                inputs_embeds: Tensor = None,
                labels: Tensor = None,
                add_feature_0: Tensor = None) -> set:
        """
        前向传播过程

        Args:
            input_ids (Tensor): 这里是 BertTokenizer 对句子处理的结果.
            attention_mask (Tensor): 处理padding. 1 是句子本身，0 是padding部分.
            token_type_ids (Tensor, optional): 上下句标记. Defaults to None.
            position_ids (Tensor, optional): 字位置标记. Defaults to None.
            head_mask (Tensor, optional): Mask heads if we want to. Defaults to None.
            inputs_embeds (Tensor, optional): 可以选择不传入input_ids而是直接传入embedding结果. Defaults to None.
            labels (Tensor, optional): 真实标签. Defaults to None.
            add_feature_0 (Tensor, optional): 额外特征，手工传入. Defaults to None.

        Returns:
            set: (loss), logits, (hidden_states), (attentions)
        """
        outputs = self.bert(
                            input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,)
        pooled_output = outputs[1]  # [CLS]
        pooled_output = self.dropout(pooled_output)

        # add_features 拼接在这里，按自己需要增减，这里举例子就给了一个
        if add_feature_0 is not None:
            add_f_0 = add_feature_0.float()
            pooled_output = torch.cat((pooled_output, add_f_0), dim=1)
            # pooled_output = torch.cat((pooled_output, add_f_0, add_f_1, ...), dim=1)
            logits = self.classifier_add(pooled_output)
        else:
            logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            if self.num_labels == 1:
                #  If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss)
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)
