import numpy as np
import pandas as pd
import os
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, HuberRegressor
from sklearn.preprocessing import StandardScaler
import pickle

TRAIN_PATH = "tabular-playground-series-aug-2022"

train_df = pd.read_csv(os.path.join(TRAIN_PATH, 'train.csv'))

train_x = train_df.drop('id', axis=1) #useless
train_x = train_x.drop('failure', axis=1) #for y
train_y = train_df['failure']
'''
def preprocess(data):

    feature = [ 'loading', 'measurement_0', 'measurement_1', 'measurement_2',
                'measurement_3', 'measurement_4', 'measurement_5', 'measurement_6',
                'measurement_7', 'measurement_8', 'measurement_9', 'measurement_10',
                'measurement_11', 'measurement_12', 'measurement_13', 'measurement_14',
                'measurement_15', 'measurement_16']
    m17_corre = {
        'A': ['measurement_5', 'measurement_6', 'measurement_8'],
        'B': ['measurement_4', 'measurement_5', 'measurement_7'],
        'C': ['measurement_5', 'measurement_7', 'measurement_8', 'measurement_9'],
        'D': ['measurement_5', 'measurement_6', 'measurement_7', 'measurement_8'],
        'E': ['measurement_4', 'measurement_5', 'measurement_6', 'measurement_8']
    }
    data['m3_missing'] = data.measurement_3.isna().astype('int64')
    data['m5_missing'] = data.measurement_5.isna().astype('int64')
    data['area'] = data['attribute_2'] * data['attribute_3']
    data['loading'] = np.log1p(data['loading'])
    for p_code in data.product_code.unique():
        for_other = KNNImputer(n_neighbors=3)
        data.loc[data.product_code==p_code, feature] = for_other.fit_transform(data.loc[data.product_code==p_code, feature])
        correlated_measurement = m17_corre[p_code]
        c_data = data[data.product_code==p_code]
        c_data_nona = c_data[correlated_measurement+['measurement_17']].dropna()
        print(c_data_nona.shape)
        c_data_miss_m17 = c_data[c_data['measurement_17'].isnull()]
        model = HuberRegressor()
        model.fit(c_data_nona[correlated_measurement], c_data_nona['measurement_17'])
        data.loc[(data.product_code==p_code)&(data['measurement_17'].isnull()), 'measurement_17'] = model.predict(c_data_miss_m17[correlated_measurement])
    
    return data
'''
def preprocess(data):
    feature =  ['loading', 'measurement_0', 'measurement_1', 'measurement_2',
                'measurement_3', 'measurement_4', 'measurement_5', 'measurement_6',
                'measurement_7', 'measurement_8', 'measurement_9', 'measurement_10',
                'measurement_11', 'measurement_12', 'measurement_13', 'measurement_14',
                'measurement_15', 'measurement_16', 'measurement_17']
    m17_corre = {
        'A': ['measurement_5', 'measurement_6', 'measurement_8'],
        'B': ['measurement_4', 'measurement_5', 'measurement_7'],
        'C': ['measurement_5', 'measurement_7', 'measurement_8', 'measurement_9'],
        'D': ['measurement_5', 'measurement_6', 'measurement_7', 'measurement_8'],
        'E': ['measurement_4', 'measurement_5', 'measurement_6', 'measurement_8']
    }
    data['loading'] = np.log1p(data['loading'])
    data['m3_missing'] = data.measurement_3.isna().astype('int64')
    data['m5_missing'] = data.measurement_5.isna().astype('int64')
    data['area'] = data['attribute_2'] * data['attribute_3']
    for p_code in data.product_code.unique():
        print("start processing product_code:",p_code)
        c_data = data[data.product_code==p_code]
        correlated_measurement = m17_corre[p_code]
        c_data_nona = c_data[correlated_measurement+['measurement_17']].dropna()
        c_data_miss_only_m17 = c_data[(~c_data[correlated_measurement].isnull().any(axis=1)) & (c_data['measurement_17'].isnull())]
        model = HuberRegressor()
        model.fit(c_data_nona[correlated_measurement], c_data_nona['measurement_17'])
        data.loc[(data.product_code==p_code)&(~c_data[correlated_measurement].isnull().any(axis=1))&(data['measurement_17'].isnull()), 'measurement_17'] = model.predict(c_data_miss_only_m17[correlated_measurement])
        for_rest = KNNImputer(n_neighbors=3)
        data.loc[data.product_code==p_code, feature] = for_rest.fit_transform(data.loc[data.product_code==p_code, feature])
        #for i in range(3,17):
            #data[data.product_code==code][f'measurement_{i}'].fillna(value=data[data.product_code==code][f'measurement_{i}'].mean(), inplace=True)
    #data['measurement_avg'] = data[[f'measurement_{i}' for i in range(3, 17)]].mean(axis=1)
    return data

train=preprocess(train_x)

select_feature = ['measurement_1','measurement_10','measurement_17', 'm3_missing', 'm5_missing', 'loading', 'area']
sc = StandardScaler()
train_x = sc.fit_transform(train[select_feature]) 
model = LogisticRegression(max_iter=500, C=0.0001, penalty='l2', solver='newton-cg') 
model.fit(train_x, train_y)
model_file = open('logistic_regression.sav', 'wb')
pickle.dump(model, model_file)
model_file.close()
print("well done!")