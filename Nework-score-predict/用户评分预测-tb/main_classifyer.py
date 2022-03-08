import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from dataProcess import data_process, imblearn
from dataFill import rf_fill

model_path = 'clf_model.pkl'
train_path = 'train.csv'
test_path = 'test.csv'


# 分类器

# K折交叉验证标签的可靠程度
def Fold(clf, csv_data, k=5):
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
        # train a classification model with the selected features on the training dataset
        xt, yt = x[train], y[train]
        # imblance 解决数据不均衡分布
        xt, yt = imblearn(xt, yt)
        model = clf()
        model.fit(xt, yt)
        # predict the class labels of test data
        y_predict = model.predict(x[test])
        # obtain the classification accuracy on the test data
        print(classification_report(y[test], y_predict, zero_division=0))
        i = i + 1
        return


def getmodel(clf, data):
    import os
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        x = data.iloc[:, 0:-1].astype(float).to_numpy()
        y = data.iloc[:, -1].astype(int).to_numpy()
        # imblance 解决数据不均衡分布
        x, y = imblearn(x, y)
        model = clf()
        model.fit(x, y)
        joblib.dump(model, model_path)
        return model


# le = preprocessing.LabelEncoder()


if __name__ == '__main__':
    # step1 获取处理过的数据
    data = data_process(train_path)

    # 这几种补全方法结果看并没有特别大的提升
    # data.fillna(0, inplace=True)
    # data=data.interpolate()
    data = rf_fill(data, data.columns[data.isnull().any()].to_list())

    # # 1 GDBT 分类器
    # from sklearn.ensemble import GradientBoostingClassifier
    #
    # clf = lambda: GradientBoostingClassifier()
    # 2 RF 随机森林分类器
    from sklearn.ensemble import GradientBoostingClassifier

    clf = lambda: GradientBoostingClassifier()

    # 3 xgboost分类器
    # from xgboost import XGBRFClassifier
    #
    # clf = lambda: XGBRFClassifier()
    # step2 k折交叉验证调试模型 调试数据的特征工程
    Fold(clf, data)

    # step3 根据训练好的模型来数据进行分类预测

    # _model = getmodel(clf, data)
    # test = pd.read_csv(test_path)
    # from dataProcess import selected_features
    #
    # test = test[selected_features]
    #
    # # 去掉和用户评级明显有关的特征（例如in_net_group）为空的记录
    # test = test[~test['in_net_group'].isna()]
    # test = test[~test['user_age'].isna()]
    # test.fillna(0, inplace=True)
    # trans = {
    #     1: '1-2',
    #     2: '3-4',
    #     3: '5-6',
    #     4: '7-8',
    #     5: '9-10',
    # }
    # test['value'] = [trans[i] for i in _model.predict(test)]
    # test.to_csv('predicted.csv')
