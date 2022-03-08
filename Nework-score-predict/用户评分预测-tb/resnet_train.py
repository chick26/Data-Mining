import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class TrainTool:
    net = None
    path = ""

    def __init__(self, model, path, load=False, parallel=True):
        _net = model
        _net.training = True
        _net.cuda()
        if not isinstance(_net, nn.DataParallel) and parallel:
            _net = nn.DataParallel(_net)
        if load:
            _net.load_state_dict(torch.load(path))
            _net = _net.eval()
            _net.training = False
        self.net = _net
        self.path = path

    def train(self, criterion, optimizer, train_d, valid_d, jian_du=True, dataset_keys=['data', 'label'],
              batch_size=20, epochs=10,
              learning_rate=0.0001):
        train_loss, val_loss, train_acc, val_acc = [], [], [], []
        net = self.net
        optimizer = optimizer(net.parameters(), lr=learning_rate)
        train_loader = DataLoader(train_d, batch_size, shuffle=True)
        valid_loader = DataLoader(valid_d, batch_size, shuffle=True)
        val_loss_min = np.Inf
        val_acc_max = 0
        for epoch in range(epochs):
            net.train()
            correct = 0
            for i, data in enumerate(train_loader):
                inputs = data[dataset_keys[0]]
                if jian_du:
                    labels = data[dataset_keys[1]]
                else:
                    labels = inputs
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                labels = labels.squeeze()  # must be Tensor of dim BATCH_SIZE, MUST BE 1Dimensional!
                optimizer.zero_grad()
                outputs = net(inputs)  # calls "forward" method
                if isinstance(outputs, tuple):
                    loss = criterion(*outputs, labels)
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # running stats
                if (i + 1) % (train_d.size // batch_size // 10) == 0:
                    print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, epochs,
                                                                        i + 1, train_d.size // batch_size,
                                                                        loss.item()))
                if jian_du:
                    y_hat = net(inputs)
                    pred = y_hat.max(1, keepdim=True)[1]
                    # pred = value_index(y_hat)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
                    # correct += torch.sum(pred.eq(labels.view_as(pred)).sum(1) == 2)
                train_loss.append(loss.item())
                train_acc.append(100. * correct / train_d.size)
            if jian_du:
                print('Epoch: {}, Training Loss: {:.5f}, Training set accuracy: {}/{} ({:.3f}%)'.format(
                    epoch + 1, loss.item(), correct, train_d.size, train_acc[-1]))
            else:
                print('Epoch: {}, Training Loss: {:.5f}'.format(epoch + 1, loss.item()))
            net.eval()
            sum_loss = 0
            correct = 0
            with torch.no_grad():
                for i, data in enumerate(valid_loader):
                    inputs = data[dataset_keys[0]]
                    if jian_du:
                        labels = data[dataset_keys[1]]
                    else:
                        labels = inputs
                    labels = labels.squeeze()  # must be Tensor of dim BATCH_SIZE, MUST BE 1Dimensional!
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    optimizer.zero_grad()
                    y_hat = net(inputs)
                    if isinstance(y_hat, tuple):
                        loss = criterion(*y_hat, labels).item()
                    else:
                        loss = criterion(y_hat, labels).item()
                    sum_loss += loss * len(labels)  # sum up batch loss
                    if jian_du:
                        pred = y_hat.max(1, keepdim=True)[1]
                        correct += pred.eq(labels.view_as(pred)).sum().item()

            val_acc.append(100. * correct / valid_d.size)
            val_loss.append(sum_loss / valid_d.size)
            if jian_du:
                print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
                    val_loss[-1], correct, valid_d.size, val_acc[-1]))
                if val_acc[-1] > val_acc_max:
                    print('Validation accuracy has increased ({:.4f}%%-->{:.4f}%%). Model saved...'.format(val_acc_max,
                                                                                                           val_acc[-1]))
                    self.save_net()
                    val_acc_max = val_acc[-1]
            else:
                print('Test set: Average loss: {:.4f}'.format(val_loss[-1]))
                if val_loss[-1] < val_loss_min:
                    print('Validation loss has decreased ({:.4f}-->{:.4f}). Model saved...'.format(val_loss_min,
                                                                                                   val_loss[-1]))
                    self.save_net()
                    val_loss_min = val_loss[-1]
        return train_loss, val_loss, train_acc, val_acc

    def test(self, dataset, classes: list, batch_size=16):
        self.net.training = False
        train_loader = DataLoader(dataset, batch_size, shuffle=False)
        classes_len = len(classes)
        class_correct = list(0. for _ in range(classes_len))
        class_total = list(0. for _ in range(classes_len))
        class_incorrect = list(list(0. for _ in range(classes_len)) for _ in range(classes_len))
        for _, data in enumerate(train_loader):
            inputs, labels = data['data'], data['label']
            inputs = Variable(inputs.cuda())
            outputs = self.net(Variable(inputs).cuda())
            _, predicted = torch.max(outputs.data, 1)
            pd_c = predicted.cpu()
            c = (pd_c == labels.squeeze()).numpy()
            for i in range(len(labels)):
                label = int(labels[i][0])
                if c[i] != 1:
                    class_incorrect[label][pd_c[i]] += 1
                else:
                    class_incorrect[label][label] += 1
                class_correct[label] += c[i]
                class_total[label] += 1
        for i in range(len(classes)):
            print("Accuracy of %5s: %2d%%" % (classes[i], 100 * class_correct[i] / class_total[i]))
        for i in range(len(classes)):
            print("Wrong of %5s: " % (classes[i]), class_incorrect[i])
        return class_incorrect

    def save_net(self):
        self.net = self.net.eval()
        torch.save(self.net.state_dict(), self.path)

    def batch_predict(self, dataset, batch_size=1000):
        data_loader = DataLoader(dataset, batch_size, shuffle=False)
        r = []
        for _, data in enumerate(data_loader):
            data = Variable(data['data'].cuda())
            outputs = self.net(Variable(data).cuda())
            _, predicted = torch.max(outputs.data, 1)
            r.append(predicted.cpu().tolist())
        return sum(r, [])

    def batch_decode(self, dataset, batch_size=1000):
        data_loader = DataLoader(dataset, batch_size, shuffle=False)
        r = []
        for _, data in enumerate(data_loader):
            data = Variable(data['data'].cuda())
            outputs = self.net(Variable(data).cuda())
            predicted = outputs.data
            r.append(predicted.cpu().tolist())
        return sum(r, [])


def get_line_plot(path, train_loss, val_loss, train_acc, val_acc, jian_du=True):
    epochs_n = len(val_loss)
    iter_n = len(train_loss) / epochs_n
    fig, ax1 = plt.subplots(figsize=(9, 5))
    if jian_du:
        ax2 = ax1.twinx()  # 共享x轴
    x = [i / iter_n for i in range(len(train_acc))]
    ax1.plot(x, train_loss, 'r', label=u'Training loss')
    ax1.plot(range(len(val_loss) + 1), [val_loss[0] + 1] + val_loss, 'b', label=u'Validation loss')
    ax1.legend()
    if jian_du:
        ax2.plot(x, train_acc, 'r', label=u'Training accuracy')
        ax2.plot(range(len(val_acc) + 1), [0] + val_acc, 'b', label=u'Validation accuracy')
        ax2.tick_params(axis='y')
        ax2.legend()
        ax2.set_ylabel(u'accuracy')
    plt.xlabel(u'epoch')
    ax1.set_ylabel(u'loss')
    fig.savefig(path, bbox_inches='tight')
    plt.show()
