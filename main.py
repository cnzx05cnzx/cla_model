import random
import time
import torch
import numpy as np
from train_eval import train, test, mul_test
from importlib import import_module
import argparse
from data_load import get_dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Transformer', required=False, help='choose a model')
parser.add_argument('--embedding', default='SougouNews.npz', type=str, help='random or other')
# parser.add_argument('--embedding', default='random', type=str, help='random or other')
args = parser.parse_args()


def seed_init(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def one_model():
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer

    choose = import_module('models.' + model_name)
    config = choose.Config(args.embedding)
    seed_init(config.seed)

    print("Loading data...")

    trainl, devl, testl, config.vocab_size = get_dataloader(config)

    model = choose.Model(config).to(config.device)
    train(config, model, trainl, devl)
    test(config, model, testl)


def many_model():
    name_lists = ['TextCNN', 'TextRNN', 'Transformer']

    choose = list(import_module('models.' + model_name) for model_name in name_lists)
    configs = list(choo.Config(args.embedding) for choo in choose)

    seed_init(configs[0].seed)

    print("Loading data...")

    trainl, devl, testl, vocab_size = get_dataloader(configs[0])

    model = list(choo.Model(config).to(config.device) for choo, config in zip(choose, configs))
    mul_test(configs[0], model, name_lists, testl, 'mean')


if __name__ == '__main__':
    one_model()
    # many_model()
