#!/usr/bin/env python
# coding: utf-8

##################################################

#    @Date: 2021-01-13 17:47:12
#    @FileDescription: 模型预测用【注意本文件暂未经过测试】
#    @FirstEditors: 大卫
#    @LastEditors: 大卫
#    @Department: None
#    @LastEditTime: 2021-01-13 17:47:14

##################################################

import os
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer

from .model import Config, BertForSentClfWithFeatures


config = Config()
model = BertForSentClfWithFeatures()


class BertPredictor(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        if torch.cuda.is_available() and config.use_cuda:
            self.device_id = config.device_id  # GPU id
            self.device = torch.device("cuda:" + str(config.device_id))  # 注意选择
        else:
            self.device = torch.device("cpu")

    def load_model(self, model_file: str = None, use_cuda: bool = True):
        """
        读取模型到 显存 或者 内存

        Args:
            model_file (str, optional): 模型文件的路径，如果不写，则默认是 config.save_path, config.model_name. Defaults to None.
            use_cuda (bool, optional): 是否使用 CUDA 加速预测。如果不写默认使用，但是也会判断CUDA是否可用. Defaults to True.
        """
        if not model_file:
            model_file = os.path.join(config.save_path, config.model_name)
        print(f"Loading model at: {model_file}")
        if torch.cuda.is_available() and use_cuda:
            device_id = self.device_id
            torch.cuda.set_device(device_id)
            model.load_state_dict(
                torch.load(model_file)
            )
            model.cuda()
            print("Model already in CUDA.")
        else:
            model.load_state_dict(
                torch.load(
                    model_file,
                    map_location=torch.device("cpu"),
                )
            )
            print("Model already in CPU.")

    def _test_file_reader(self) -> List[str]:
        """
        读取文件。等待预测的

        Returns:
            List[str]: ['中华女子学院：本科层次仅1专业招男生\t3\n', '两天价网站背后重重迷雾：做个网站究竟要多少钱\t4\n', ...]
        """
        file = config.test_file
        with open(file, encoding="utf-8") as f:
            sentences_and_labels = [line for line in f.readlines()]
        f.close()
        return sentences_and_labels

    def _sent_tokenizer(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        sentences = []
        test_labels = []
        clean_labels = []  # 清洗掉\n的int类型的标签
        sentences_and_labels = self._test_file_reader()
        for sentence_with_label in sentences_and_labels:
            sentence, label = sentence_with_label.split('\t')
            assert isinstance(sentence, str)
            assert isinstance(label, str)
            sentences.append(sentence)
            test_labels.append(label)
        for label in test_labels:
            clean_labels.append(int(label.strip('\n')))

        encoding = self.tokenizer(sentences,
                                  return_tensors='pt',           # pt 指 pytorch，tf 就是 tensorflow
                                  padding=True,                  # padding到最长的那句话
                                  truncation=True,               # 激活并控制截断
                                  max_length=config.max_len)     # 最长padding
        input_ids = encoding['input_ids']
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding['token_type_ids']
        test_labels = torch.tensor(clean_labels).clone().detach()
        return input_ids, attention_mask, token_type_ids, test_labels

    def _to_dataloader(self) -> DataLoader:
        """
        包装在 Dataloader 中，供模型读取数据特征。当前放入其他 add_feature，如果有需要可以在这里加入。

        Returns:
            DataLoader: 模型数据 Dataloader
        """
        input_ids, attention_mask, token_type_ids, test_labels = self._sent_tokenizer()
        # put into Dataloader
        train_dataset = TensorDataset(
            input_ids,
            attention_mask,
            token_type_ids,
            test_labels
        )
        train_sampler = SequentialSampler(train_dataset)
        test_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=config.batch_size
        )
        return test_dataloader

    def predict(self) -> List[np.array]:
        """
        预测

        Returns:
            List[np.array]: 预测结果，需要自行从数字对应到汉字类别
        """
        model.eval()
        test_dataloader = self._to_dataloader()
        pred = []
        # real_label = []  # 这里可以增加 real_label 用于对比

        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            (
                b_input_ids,
                b_input_mask,
                b_input_token_type,
                b_test_labels
            ) = batch
            with torch.no_grad():  # 不保存梯度，不然显存爆炸
                outputs = model(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                    token_type_ids=b_input_token_type
                )
            # Move logits and labels to CPU
            scores = outputs[0].detach().cpu().numpy()
            pred_flat = np.argmax(scores, axis=1).flatten()
            pred.append(pred_flat)
        return pred


if __name__ == "__main__":
    news_predictor = BertPredictor()
    news_predictor.load_model()
    list_p = news_predictor.predict()  # 预测结果
    print(list_p[0])  # np.array
