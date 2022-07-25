from collections import Counter
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import numpy as np
import pandas as pd

import pkuseg


class TextCNNDataSet(Dataset):
    def __init__(self, data, data_targets):
        self.content = torch.LongTensor(data)
        self.pos = torch.LongTensor(data_targets)

    def __getitem__(self, index):
        return self.content[index], self.pos[index]

    def __len__(self):
        return len(self.pos)


def get_dataloader(config):
    data = pd.read_csv(config.file_path)

    # data = data[:20000]
    # data = data[:2000]
    seg = pkuseg.pkuseg()

    data['cut'] = data["comment"].apply(lambda x: list(seg.cut(x)))

    # 请使用测试集来构建词表
    with open(config.vocab_path, 'w', encoding='utf-8') as fou:
        fou.write("<unk>\n")
        fou.write("<pad>\n")
        # 使用 < unk > 代表未知字符且将出现次数为1的作为未知字符
        # 实用 < pad > 代表需要padding的字符(句子长度进行统一)

        vocab = [word for word, freq in Counter(j for i in data['cut'] for j in i).most_common() if freq > 5]

        for v in vocab:
            fou.write(v + "\n")

    # 初始化生成 词对序 与 序对词 表
    with open(config.vocab_path, encoding='utf-8') as fin:
        vocab = [i.strip() for i in fin]
    char2idx = {i: index for index, i in enumerate(vocab)}
    idx2char = {index: i for index, i in enumerate(vocab)}

    pad_id = char2idx["<pad>"]
    unk_id = char2idx["<unk>"]

    max_len = config.max_len

    def tokenizer(name):
        inputs = []
        sentence_char = [[j for j in i] for i in data[name]]
        # 将输入文本进行padding
        for index, i in enumerate(sentence_char):
            temp = [char2idx.get(j, unk_id) for j in i]
            if len(temp) < max_len:
                # temp.extend(str(pad_id) * (max_len - len(temp)))
                temp.extend([pad_id for i in range(max_len - len(temp))])
            else:
                temp = temp[:max_len]
            inputs.append(temp)
        return inputs

    # 针对只有整个数据集的处理
    DataSet = TextCNNDataSet(tokenizer('cut'), list(data["pos"]))
    train_size = int(len(list(data["pos"])) * 0.6)
    test_size = len(list(data["pos"])) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(DataSet, [train_size, test_size])

    test_size = int(len(list(test_dataset)) * 0.5)
    valid_size = int(len(list(test_dataset))) - test_size
    valid_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [valid_size, test_size])

    batch_size = config.batch_size
    TrainLoader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    ValidLoader = DataLoader(valid_dataset, batch_size, shuffle=False, drop_last=True)
    TestLoader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True)

    print('Training {} samples...'.format(len(TrainLoader.dataset)))
    print('Validing {} samples...'.format(len(ValidLoader.dataset)))
    print('Testing {} samples...'.format(len(TestLoader.dataset)))
    return TrainLoader, ValidLoader, TestLoader, len(vocab)


if __name__ == "__main__":
    pass
