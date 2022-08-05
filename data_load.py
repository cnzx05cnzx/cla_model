from collections import Counter

from sklearn.model_selection import train_test_split
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

    def tokenizer(df):
        inputs = []
        sentence_char = [[j for j in i[0]] for step, i in df.iterrows()]
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
    x_train, x_test, y_train, y_test = train_test_split(data['cut'], data['pos'], test_size=0.2, random_state=721)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=721)

    train_dataset = TextCNNDataSet(tokenizer(x_train), list(y_train))
    valid_dataset = TextCNNDataSet(tokenizer(x_valid), list(y_valid))
    test_dataset = TextCNNDataSet(tokenizer(x_test), list(y_test))

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
