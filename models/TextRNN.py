# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np


class Config(object):
    """配置参数"""

    def __init__(self, embedding):
        self.model_name = 'TextRNN'
        self.file_path = './data/news.csv'  # 总集
        self.train_path = './data/train.txt'  # 训练集
        self.dev_path = './data/dev.txt'  # 验证集
        self.test_path = './data/test.txt'  # 测试集

        self.vocab_path = './data/vocab.txt'  # 词表
        self.save_path = './saved_dict/' + self.model_name + '.pkl'  # 模型训练结果

        self.embedding_pretrained = torch.tensor(np.load('./utils/' + embedding)["embeddings"].astype(
            'float32')) if embedding != 'random' else None  # 预训练词向量
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.device = torch.device('cpu')  # 设备

        self.seed = 721
        self.dropout = 0.5  # 随机失活
        self.early_stop = 10  # 早停机制
        self.num_classes = 5  # 类别数
        self.vocab_size = 652  # 词表大小，在运行时赋值
        self.num_epochs = 100  # epoch数
        self.batch_size = 32  # batch大小
        self.max_len = 40  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.out_dim = 64  # 输出维度
        self.embed_size = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 64  # 词向量嵌入维度


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=1)

        self.lstm = nn.LSTM(config.embed_size, config.out_dim, batch_first=True, dropout=config.dropout,
                            bidirectional=True, num_layers=2)
        self.linear = nn.Sequential(
            nn.Linear(2 * config.out_dim, config.num_classes),
            # nn.Dropout(config.dropout),
            # nn.ReLU(),
            # nn.Linear(32, config.num_classes)
        )

        self.dropouts = nn.ModuleList([nn.Dropout(config.dropout) for _ in range(3)])

    def forward(self, x):
        # 初始输入格式为(length, batch_size)
        out = self.embedding(x)
        # (length, batch_size, emb) -> (batch_size, length , emb )

        # out = torch.transpose(out, 0, 1)

        out, (h, c) = self.lstm(out)
        out = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)

        # multi-sample dropout 不一定有效
        # for i, dropout in enumerate(self.dropouts):
        #     if i == 0:
        #         res = dropout(out)
        #         res = self.linear(res)
        #     else:
        #         temp_out = dropout(out)
        #         res = res + self.linear(temp_out)
        # return res

        out = self.linear(out)
        return out
