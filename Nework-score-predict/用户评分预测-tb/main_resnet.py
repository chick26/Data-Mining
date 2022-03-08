import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from dataFill import rf_fill
from dataProcess import data_process, imblearn
from pytorch_datasets import BaseDataset
from resnet import ResNet
from resnet_train import TrainTool

from sklearn.preprocessing import Normalizer

model_path = 'resnet.pth'
train_path = 'train.csv'
test_path = 'test.csv'


# K折交叉验证标签的可靠程度
def Fold(data, k=5):
    x = data.iloc[:, 0:-1].astype(float)
    y = data.iloc[:, -1].astype(int)
    # normalised = pd.DataFrame(Normalizer(norm='l1').fit_transform(x), columns=x.columns, index=x.index)
    # data = pd.concat([normalised, y], axis=1)
    # split data into k folds
    ss = KFold(n_splits=k, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam
    # perform evaluation on classification task
    i = 0
    for train, test in ss.split(x, y, y):
        print('*' * 25, '第', i + 1, '折', '*' * 25)

        # train a classification model with the selected features on the training dataset
        # tt = TrainTool(ResNet(), model_path, load=True)# 载入Resnet模型
        tt = TrainTool(ResNet(num_classes=3), model_path, load=True)  # 不载入Resnet模型
        tx, ty = imblearn(data.iloc[train].iloc[:, 0:-1], data.iloc[train].iloc[:, -1])
        # traind = pd.concat([tx, ty], axis=1)
        # traind = data.iloc[train]
        # tt.train(criterion, optimizer, BaseDataset(traind), BaseDataset(data.iloc[test]),
        #          epochs=50)
        # predict the class labels of test data
        xtest = x.iloc[test]
        y_predict = tt.batch_predict(BaseDataset(xtest))
        # obtain the classification accuracy on the test data
        print(classification_report(y.iloc[test].tolist(), y_predict, zero_division=0))
        i = i + 1
        return


if __name__ == '__main__':
    df = data_process(train_path)
    # df = rf_fill(df, df.columns[df.isnull().any()].to_list())
    df.fillna(0, inplace=True)
    Fold(df)
