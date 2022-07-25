import random
import time
import torch
import numpy as np
from train_eval import train, test
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


if __name__ == '__main__':
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer

    choose = import_module('models.' + model_name)
    config = choose.Config(args.embedding)
    seed_init(config.seed)

    start_time = time.time()
    print("Loading data...")

    trainl, devl, testl, config.vocab_size = get_dataloader(config)

    model = choose.Model(config).to(config.device)
    train(config, model, trainl, devl)
    test(config, model, testl)

