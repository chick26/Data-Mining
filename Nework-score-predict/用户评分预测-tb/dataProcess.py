import pandas as pd

selected_features: list


# 数据处理的逻辑代码
def data_process(path):
    csv_data = pd.read_csv(path)
    # night_cell_id 去语义 （弃用字段）
    # global le
    # le = le.fit(csv_data['night_cell_id'].tolist())
    # csv_data['night_cell_id'] = le.transform(csv_data['night_cell_id'].tolist())

    # 通过 .describe() 可以查看数值特征列的一些统计信息
    # describe种有每列的统计量，个数count、平均值mean、方差std、最小值min、中位数25% 50% 75% 、以及最大值
    # 看这个信息主要是瞬间掌握数据的大概的范围以及每个值的异常值的判断，
    # describe = csv_data.describe()
    # print(describe)
    # _info = csv_data.info()

    # 去掉和用户评级明显有关的特征（例如in_net_group）为空的记录
    csv_data = csv_data[~csv_data['in_net_group'].isna()]
    csv_data = csv_data[~csv_data['user_age'].isna()]
    # 很多用户直接给出了10分，不是特别具有参考意义
    csv_data = csv_data[csv_data['value'] != 10]
    # csv_data = csv_data[csv_data['value'] != 1]
    # 缺失值超过1/3
    csv_data.drop(
        ['if_sq', 'if_cx', 'kpi_vot_xsrvcc_swrate', 'kpi_vot_xsrvcc_swtime', 'kpi_vot_v2v_time', 'phone_num', 'dh_num',
         'price_group', 'kpi_vot_v2all_time', 'kpi_vot_netconnectionrate',
         'kpi_vot_mos_over3rate', 'kpi_avgvideobufferduration_4g'], axis=1, inplace=True)

    # 明显和预测任务无关的特征
    csv_data.drop(['phone'], axis=1, inplace=True)
    # 'night_cell_id' 这个字段是ID 而且测试集和数据集 元素集合不一致 弃用
    csv_data.drop(['night_cell_id'], axis=1, inplace=True)

    # 特征数值偏向一种结果的去掉
    csv_data.drop(['used_2g', 'used_3g', 'used_4g', 'kpi_web_e2e_excellentrate_3g', 'kpi_web_e2e_goodrate_3g',
                   'kpi_web_e2e_poorrate_3g', 'kpi_web_e2e_hysteresisrate', 'kpi_vot_dropcallrate',
                   'kpi_avgtcpdlrttdelay_2g', 'kpi_avgtcpdlrttdelay_5g',
                   'kpi_avgwebresdelay_2g', 'kpi_avgwebresdelay_3g', 'kpi_avgwebresdelay_5g',
                   'kpi_avgvideobufferduration_2g', 'kpi_avgvideobufferduration_3g', 'kpi_avgvideobufferduration_5g',
                   'kpi_initbuffersuccratio_2g', 'kpi_initbuffersuccratio_3g', 'kpi_initbuffersuccratio_4g',
                   'kpi_initbuffersuccratio_5g'
                   ], axis=1, inplace=True)

    # 缺失值填充
    # csv_data.fillna(0, inplace=True)

    csv_data = csv_data.round(1)
    # cns = csv_data.columns.tolist()  # 50
    # _value_counts = {}
    # for c in cns:
    #     _value_counts[c] = []
    # for c in cns:  # 50
    #     for i, g in csv_data.groupby('value'):  # 10
    #         count = g[c].value_counts()
    #         _value_counts[c].append(count)
    # for k, v in _value_counts.items():
    #     _value_counts[k] = pd.concat(v, axis=1)
    #     _value_counts[k].columns = map(lambda a: str(a + 1), range(10))
    # 数据分箱
    # TODO 选择合适的分类数
    # 二分类
    csv_data['value'] = pd.cut(csv_data['value'], [0, 5, 10], labels=False) + 1
    # 三分类
    # csv_data['value'] = pd.cut(csv_data['value'], [0, 4, 7, 10], labels=False) + 1
    # 五分类
    # csv_data['value'] = pd.cut(csv_data['value'], [0, 2, 4, 6, 8, 10], labels=False) + 1
    # csv_data = csv_data.drop(
    #     labels=['user_sex', 'online_2g', 'online_3g', 'flow_2g', 'flow_3g', 'bill', 'if_lucknum', 'is_volte'], axis=1)
    # csv_data = csv_data[['gprs_bytes_type',
    #                      'flow_4g', 'value'
    #                      ]]
    # TODO 使用Resnet 特征数不能小于7
    global selected_features
    selected_features = csv_data.columns.to_list()[0:-1]
    csv_data = csv_data.reset_index(drop=True)
    return csv_data


def imblearn(xt, yt):
    # 欠采样
    # from imblearn.under_sampling import RandomUnderSampler
    # ros = RandomUnderSampler()
    # return ros.fit_resample(xt, yt)
    # 欠采样
    # from imblearn.under_sampling import RepeatedEditedNearestNeighbours
    # ros = RepeatedEditedNearestNeighbours()
    # return ros.fit_resample(xt, yt)

    # 过采样
    # from imblearn.over_sampling import RandomOverSampler
    # ros = RandomOverSampler()
    # return ros.fit_resample(xt, yt)

    # from imblearn.over_sampling import SMOTE
    # smo = SMOTE()
    # return smo.fit_resample(xt, yt)

    # from imblearn.over_sampling import BorderlineSMOTE
    # smo = BorderlineSMOTE()
    # return smo.fit_resample(xt, yt)
    #
    # from imblearn.over_sampling import ADASYN
    # smo = ADASYN()
    # return smo.fit_resample(xt, yt)

    # 过采样与欠采样结合
    from imblearn.combine import SMOTEENN
    smote_enn = SMOTEENN(random_state=0)
    return smote_enn.fit_resample(xt, yt)
