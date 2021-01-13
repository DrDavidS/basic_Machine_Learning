#!/usr/bin/env python
# coding: utf-8

##################################################

#    @Date: 2021-01-13 10:14:02
#    @FileDescription: 基于Pytorch BERT-Chinse模型，用于文本分类
#    @FirstEditors: 大卫
#    @LastEditors: 大卫
#    @Department: None
#    @LastEditTime: 2021-01-13 10:29:26
#
#    @软件版本备注：
#    PyTorch 版本： 1.7.1
#    Transformers 版本： 3.1.0

##################################################
import os
from typing import List, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer
from transformers import AdamW

from .model import Config, BertForSentClfWithFeatures


config = Config()
model = BertForSentClfWithFeatures()


class BertTrainer(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        if torch.cuda.is_available() and config.use_cuda:
            device_id = config.device_id
            self.device = torch.device("cuda:" + str(device_id))  # 注意选择
        else:
            self.device = torch.device("cpu")
        print(f"当前设备： {self.device}")

    def _train_file_reader(self) -> List[str]:
        """
        读取文件

        Returns:
            List[str]: ['中华女子学院：本科层次仅1专业招男生\t3\n', '两天价网站背后重重迷雾：做个网站究竟要多少钱\t4\n', ...]
        """
        file = config.train_file
        with open(file, encoding="utf-8") as f:
            sentences_and_labels = [line for line in f.readlines()]
        f.close()
        return sentences_and_labels

    def _sent_tokenizer(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        print("Tokenize 中...")
        sentences = []
        train_labels = []
        clean_labels = []  # 清洗掉\n的int类型的标签
        sentences_and_labels = self._train_file_reader()
        for sentence_with_label in sentences_and_labels:
            sentence, label = sentence_with_label.split('\t')
            assert isinstance(sentence, str)
            assert isinstance(label, str)
            sentences.append(sentence)
            train_labels.append(label)
        for label in train_labels:
            clean_labels.append(int(label.strip('\n')))

        encoding = self.tokenizer(sentences,
                                  return_tensors='pt',           # pt 指 pytorch，tf 就是 tensorflow
                                  padding=True,                  # padding到最长的那句话
                                  truncation=True,               # 激活并控制截断
                                  max_length=config.max_len)     # 最长padding
        input_ids = encoding['input_ids']
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding['token_type_ids']
        train_labels = torch.tensor(clean_labels).clone().detach()
        print("Tokenize 完毕...")
        return input_ids, attention_mask, token_type_ids, train_labels

    def _to_dataloader(self) -> DataLoader:
        """
        包装在 Dataloader 中，供模型读取数据特征。当前放入其他 add_feature，如果有需要可以在这里加入。

        Returns:
            DataLoader: 模型数据 Dataloader
        """
        print("Dataloader 准备中...")
        input_ids, attention_mask, token_type_ids, train_labels = self._sent_tokenizer()
        # put into Dataloader
        train_dataset = TensorDataset(
            input_ids,
            attention_mask,
            token_type_ids,
            train_labels
        )
        train_sampler = SequentialSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=config.batch_size
        )
        print("Dataloader 准备完毕")
        return train_dataloader

    def _optimizer_config(self) -> Optimizer:
        """
        优化器配置。这里选用的是 AdamW 优化器。
        优化器相关配置，如学习率等是分层的。一般来说，BERT学习率很小，而FC、CRF等层可以大一些。

        Returns:
            Optimizer: 返回优化器对象
        """
        print("优化器参数加载...")
        bert_param_optimizer = list(model.bert.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']

        # 权重衰减
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01,
             'lr': config.bert_learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             'lr': config.bert_learning_rate},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01,
             'lr': config.fc_learning_rate},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             'lr': config.fc_learning_rate}
        ]
        # 优化器
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.bert_learning_rate)
        print("优化器参数加载完毕")
        return optimizer

    def train(self, model: nn.Module) -> nn.Module:
        optimizer = self._optimizer_config()
        train_dataloader = self._to_dataloader()
        if torch.cuda.is_available() and config.use_cuda:
            model.cuda()
        # BERT training loop
        for _ in range(config.epochs):
            print(f"当前epoch： {_}")
            # 开启训练模式
            model.train()
            tr_loss = 0  # train loss
            nb_tr_examples, nb_tr_steps = 0, 0
            # Train the data for one epoch
            for batch in train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_input_token_type, b_train_labels = batch
                # 梯度归零
                optimizer.zero_grad()
                # 前向传播loss计算
                output = model(input_ids=b_input_ids,
                               attention_mask=b_input_mask,
                               token_type_ids=b_input_token_type,
                               labels=b_train_labels)
                loss = output[0]
                # 反向传播
                loss.backward()
                # 更新模型参数
                optimizer.step()
                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print(f"当前 epoch 的 Train loss: {tr_loss/nb_tr_steps}")
        return model

    def save_model(self) -> str:
        """
        保存模型

        Args:
            model (nn.Module): 训练完成的模型文件

        Returns:
            str: 模型保存路径
        """
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
            print("文件夹不存在，创建文件夹!")
        model_dict = model.state_dict()
        torch.save(model_dict, os.path.join(config.save_path, config.model_name))    # 模型保存
        return os.path.join(config.save_path, config.model_name)


if __name__ == "__main__":
    news_trainer = BertTrainer()
    trained_model = news_trainer.train(model=model)
    news_trainer.save_model()  # 保存模型
