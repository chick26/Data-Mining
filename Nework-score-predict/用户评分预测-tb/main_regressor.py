import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from dataProcess import data_process

model_path = 'rgr_model.pkl'
train_path = 'train.csv'
test_path = 'test.csv'


def fold(rgsor, csv_data, k=5):
    data = csv_data
    x = data.iloc[:, 0:-1].astype(float).to_numpy()
    # from sklearn.preprocessing import Normalizer
    # normalizer = Normalizer(norm='l1')
    # x = normalizer.fit_transform(x)
    y = data.iloc[:, -1].astype(int).to_numpy()
    # split data into k folds
    ss = KFold(n_splits=k, shuffle=True)

    # perform evaluation on classification task
    i = 0
    for train, test in ss.split(x, y, y):
        print('*' * 25, '第', i + 1, '折', '*' * 25)
        regr = rgsor()
        xt, yt = x[train], y[train]
        regr.fit(xt, yt)
        print(regr.score(xt, yt))
        print(regr.score(x[test], y[test]))
        i = i + 1
    csv_data['regsor'] = regr.predict(x)
    2  # 为了打断点


if __name__ == '__main__':
    csv_data = data_process(train_path)
    csv_data.fillna(0, inplace=True)
    # 1 GDBT 回归器
    from sklearn.ensemble import GradientBoostingRegressor

    rgr = lambda: GradientBoostingRegressor()
    # 2 RF 回归器
    # from sklearn.ensemble import RandomForestRegressor
    #
    # rgr = lambda: RandomForestRegressor()
    fold(rgr, csv_data)
