from pandas_profiling import ProfileReport
from sklearn.ensemble import RandomForestRegressor

from dataProcess import data_process


def rf_fill(data, missinglist):
    for i in missinglist:
        # 依次将待填充特征称为新的标签，缺失越大的放在越后面
        new_label = data.loc[:, i]
        # 根据缺失情况划分训练集和测试集
        y_train = new_label[new_label.notnull()]
        y_test = new_label[new_label.isnull()]
        x_train = data.iloc[y_train.index]
        x_test = data.iloc[y_test.index]
        # 其他特征的缺失部分仍要填充一个值
        x_train = x_train.fillna(0)
        x_test = x_test.fillna(0)
        # 训练RF回归模型，预测缺失特征的取值
        rfr = RandomForestRegressor(random_state=0, n_estimators=1000, n_jobs=-1)
        rfr = rfr.fit(x_train, y_train)
        Ypredict = rfr.predict(x_test)
        # 预测值回填到原始数据中
        data.loc[y_test.index, i] = Ypredict
    return data


if __name__ == '__main__':
    # 字段相关性，可视化 pandas_profiling工具
    df = data_process('train.csv')
    # profile = ProfileReport(df, title="train.cvs 报告")
    # profile.to_file("your_report.html")
    # 2分类任务 与其他字段均高相关度
    # 3分类任务 与gprs_bytes_type、flow_4g字段高相关度
    # 5分类任务 无相关度特别高的字段
    # 一般如果某特征的缺失量过大，我们会直接将该特征舍弃掉，否则可能反倒会带入较大的noise，对结果造成不良影响。
    # 如果特征缺失值都在10 % 以内，我们可以考虑用下面的方式来处理
    # 1、传统补全
    # df.fillna(0, inplace=True)
    # df=df.fillna(df.mean())

    # 用前一个数据代替NaN：method='pad'
    # df=df.fillna(method='pad')

    # 与pad相反，bfill表示用后一个数据代替NaN
    # df=df.fillna(method='bfill')
    # 2、机器学习补全

    # df = rf_fill(df, df.columns[df.isnull().any()].to_list())
    # 3、深度学习补全（只适用二维以上的数据）
