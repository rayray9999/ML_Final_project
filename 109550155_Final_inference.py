import numpy as np
import pandas as pd
import os
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, HuberRegressor
from sklearn.preprocessing import StandardScaler
import pickle
#print(pickle.format_version)
WEIGHT_PATH="" #for weight path
TEST_PATH = "tabular-playground-series-aug-2022" #for test.csv path
SUBMISSION_PATH="tabular-playground-series-aug-2022" #for sample_submission.csv path

submission = pd.read_csv(os.path.join(SUBMISSION_PATH, 'sample_submission.csv'))
test_df = pd.read_csv(os.path.join(TEST_PATH, 'test.csv'))
test_x = test_df.drop('id', axis=1)#useless
'''
def preprocess(data):

    feature = [ 'loading', 'measurement_0', 'measurement_1', 'measurement_2',
                'measurement_3', 'measurement_4', 'measurement_5', 'measurement_6',
                'measurement_7', 'measurement_8', 'measurement_9', 'measurement_10',
                'measurement_11', 'measurement_12', 'measurement_13', 'measurement_14',
                'measurement_15', 'measurement_16']
    m17_corre = {
        'F': ['measurement_4', 'measurement_5', 'measurement_6', 'measurement_7'],
        'G': ['measurement_4', 'measurement_6', 'measurement_8', 'measurement_9'],
        'H': ['measurement_4', 'measurement_5', 'measurement_7', 'measurement_8', 'measurement_9'],
        'I': ['measurement_3', 'measurement_7', 'measurement_8']
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
        'F': ['measurement_4', 'measurement_5', 'measurement_6', 'measurement_7'],
        'G': ['measurement_4', 'measurement_6', 'measurement_8', 'measurement_9'],
        'H': ['measurement_4', 'measurement_5', 'measurement_7', 'measurement_8', 'measurement_9'],
        'I': ['measurement_3', 'measurement_7', 'measurement_8']
    }
    data['loading'] = np.log1p(data['loading'])
    data['area'] = data['attribute_2'] * data['attribute_3']
    data['m3_missing'] = data.measurement_3.isna().astype('int64')
    data['m5_missing'] = data.measurement_5.isna().astype('int64')
    for p_code in data.product_code.unique():
        print("start processing product_code:",p_code)
        c_data = data[data.product_code==p_code]
        correlated_measurement = m17_corre[p_code]
        c_data_nona = c_data[correlated_measurement+['measurement_17']].dropna()
        c_data_miss_only_m17 = c_data[(~c_data[correlated_measurement].isnull().any(axis=1)) & (c_data['measurement_17'].isnull())]
        model = HuberRegressor(epsilon=2)
        model.fit(c_data_nona[correlated_measurement], c_data_nona['measurement_17'])
        data.loc[(data.product_code==p_code) & (~c_data[correlated_measurement].isnull().any(axis=1)) & (data['measurement_17'].isnull()), 'measurement_17'] = model.predict(c_data_miss_only_m17[correlated_measurement])
        #for_rest = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        for_rest = KNNImputer(n_neighbors=3)
        data.loc[data.product_code==p_code, correlated_measurement+['measurement_17']] = for_rest.fit_transform(data.loc[data.product_code==p_code, correlated_measurement+['measurement_17']])
        data.loc[data.product_code==p_code, feature] = for_rest.fit_transform(data.loc[data.product_code==p_code, feature])
        #for i in range(3,17):
            #data[data.product_code==code][f'measurement_{i}'].fillna(value=data[data.product_code==code][f'measurement_{i}'].mean(), inplace=True)
    #data['measurement_avg'] = data[[f'measurement_{i}' for i in range(3, 17)]].mean(axis=1)
    return data

test=preprocess(test_x)
select_feature = ['measurement_1','measurement_10','measurement_17', 'm3_missing', 'm5_missing', 'loading', 'area']

print("--start Standardize--")
sc = StandardScaler()
test=sc.fit_transform(test[select_feature]) 
print("--load model--")
model = LogisticRegression(max_iter=500, C=0.0001, penalty='l2', solver='newton-cg') 
model_file = open(os.path.join(WEIGHT_PATH, 'logistic_regression.sav'), 'rb')
model = pickle.load(model_file)
model_file.close()
print("--start predict---")
submission['failure']=model.predict_proba(test)[:, 1]
submission.to_csv('109550155.csv', index=False)
print("well done!")
