# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time


def train(config, model, train_iter, dev_iter):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率

    best_acc = float(0)
    cnt = 0  # 记录多久没有模型效果提升
    stop_flag = False  # 早停标签

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))

        # 训练-------------------------------
        model.train()
        t_a = time.time()
        total_loss, total_acc = 0, 0

        for i, (data, pos) in enumerate(train_iter):
            outputs = model(data.to(config.device))
            pos = pos.to(config.device)

            model.zero_grad()
            loss = F.cross_entropy(outputs, pos)
            loss.backward()
            optimizer.step()

            true = pos.data.cpu()
            predict = torch.max(outputs.data, 1)[1].cpu()
            total_loss += float(loss.item())
            total_acc += torch.eq(predict, true).sum().float().item()

        total_acc = total_acc / len(train_iter.dataset)
        total_loss = total_loss / len(train_iter)

        t_b = time.time()
        msg = 'Train Loss: {0:>5.2},  Train Acc: {1:>6.2%},  Time: {2:>7.2}'
        print(msg.format(total_loss, total_acc, t_b - t_a))

        # 验证-------------------------------
        model.eval()
        t_a = time.time()
        total_loss, total_acc = 0, 0

        for i, (data, pos) in enumerate(dev_iter):
            with torch.no_grad():
                outputs = model(data.to(config.device))
                pos = pos.to(config.device)
                loss = F.cross_entropy(outputs, pos)

            true = pos.data.cpu()
            predict = torch.max(outputs.data, 1)[1].cpu()
            total_loss += float(loss.item())
            total_acc += torch.eq(predict, true).sum().float().item()

        total_acc = total_acc / len(dev_iter.dataset)
        total_loss = total_loss / len(dev_iter)

        t_b = time.time()
        msg = 'Eval Loss: {0:>5.2},  Eval Acc: {1:>6.2%},  Time: {2:>7.2}'
        print(msg.format(total_loss, total_acc, t_b - t_a))

        if total_acc > best_acc:
            best_acc = total_acc
            torch.save(model.state_dict(), config.save_path)
            print('效果提升,保存最优参数')
            cnt = 0
        else:
            cnt += 1
            if cnt > config.early_stop:
                print('模型已无提升停止训练,验证集最高精度:%.2f' % (100 * best_acc))
                stop_flag = True
                break
    if stop_flag:
        pass
    else:
        print('训练结束,验证集最高精度:%.2f' % (100 * best_acc))


def test(config, model, test_iter):
    # 测试-------------------------------
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    t_a = time.time()
    total_acc, total_loss = 0, 0
    for i, (data, pos) in enumerate(test_iter):
        with torch.no_grad():
            outputs = model(data.to(config.device))
            pos = pos.to(config.device)
            loss = F.cross_entropy(outputs, pos)

        true = pos.data.cpu()
        predict = torch.max(outputs.data, 1)[1].cpu()
        total_loss += float(loss.item())
        total_acc += torch.eq(predict, true).sum().float().item()

    total_acc = total_acc / len(test_iter.dataset)
    total_loss = total_loss / len(test_iter)

    t_b = time.time()
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%},  Time: {2:>7.2}'
    print(msg.format(total_loss, total_acc, t_b - t_a))
