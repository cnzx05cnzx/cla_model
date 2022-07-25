# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    """配置参数"""

    def __init__(self, embedding):
        self.model_name = 'TextCNN'
        self.file_path = './data/news.csv'  # 总集
        self.train_path = './data/train.txt'  # 训练集
        self.dev_path = './data/dev.txt'  # 验证集
        self.test_path = './data/test.txt'  # 测试集

        self.vocab_path = './data/vocab.txt'  # 词表
        self.save_path = './saved_dict/' + self.model_name + '.pkl'  # 模型训练结果
        self.log_path = './log/' + self.model_name
        self.embedding_pretrained = torch.tensor(np.load('./utils/' + embedding)["embeddings"].astype(
            'float32')) if embedding != 'random' else None  # 预训练词向量
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.device = torch.device('cpu')  # 设备

        self.seed = 721
        self.dropout = 0.5  # 随机失活
        self.early_stop = 10  # 早停机制
        self.num_classes = 5  # 类别数
        self.vocab_size = 0  # 词表大小，在运行时赋值
        self.num_epochs = 100  # epoch数
        self.batch_size = 32  # batch大小
        self.max_len = 40  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed_size = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 64  # 词向量嵌入维度
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 64  # 卷积核数量


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Sequential(
            nn.Linear(config.num_filters * len(config.filter_sizes), 128),
            nn.ReLU(),
            # nn.Dropout(config.dropout),
            nn.Linear(128, config.num_classes)
        )

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
