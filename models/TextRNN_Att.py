import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    """配置参数"""

    def __init__(self, embedding):
        self.model_name = 'TextRNN_Att'
        self.file_path = './data/news.csv'  # 总集
        self.train_path = './data/train.txt'  # 训练集
        self.dev_path = './data/dev.txt'  # 验证集
        self.test_path = './data/test.txt'  # 测试集

        self.vocab_path = './data/vocab.txt'  # 词表
        self.save_path = './saved_dict/' + self.model_name + '.pkl'  # 模型训练结果

        self.embedding_pretrained = torch.tensor(np.load('./utils/' + embedding)["embeddings"].astype(
            'float32')) if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        # self.device = torch.device('cpu')  # 设备

        self.seed = 721
        self.dropout = 0.5  # 随机失活
        self.early_stop = 10  # 早停机制
        self.num_classes = 5  # 类别数
        self.vocab_size = 0  # 词表大小，在运行时赋值
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
        self.lstm = nn.LSTM(config.embed_size, config.out_dim, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.out_dim * 2))
        self.tanh2 = nn.Tanh()
        self.linear = nn.Sequential(
            nn.Linear(2 * config.out_dim, config.num_classes),
            # nn.Dropout(config.dropout),
            # nn.ReLU(),
            # nn.Linear(32, config.num_classes)
        )

    def forward(self, x):

        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, (h, c) = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(out)  # [128, 32, 256]

        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = out * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = self.linear(out)
        return out
