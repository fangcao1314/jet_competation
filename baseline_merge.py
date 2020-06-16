# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 19:59:45 2019

@author: jack
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import gc
import os
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from catboost import CatBoostRegressor, CatBoostClassifier

from xgboost import XGBClassifier
import pandas  as pd
from imblearn.over_sampling import SMOTE
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV 
from sklearn.externals import joblib
from sklearn import svm
from multiprocessing import Pool
import math

from xgboost import XGBClassifier
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
def count_column(df,column):
    tp = df.groupby(column).count().reset_index()
    tp = tp[list(tp.columns)[0:2]]
    tp.columns = [column, column+'_count']
    df=df.merge(tp,on=column,how='left')
    return df
def count_mean(df,base_column,count_column):
    tp = df.groupby(base_column).agg({count_column: ['mean']}).reset_index()
    tp.columns = [base_column, base_column+'_'+count_column+'_mean']
    df = df.merge(tp, on=base_column, how='left')
    return df
def count_count(df,base_column,count_column):
    tp = df.groupby(base_column).agg({count_column: ['count']}).reset_index()
    tp.columns = [base_column, base_column+'_'+count_column+'_count']
    df = df.merge(tp, on=base_column, how='left')
    return df
def count_sum(df,base_column,count_column):
    tp = df.groupby(base_column).agg({count_column: ['sum']}).reset_index()
    tp.columns = [base_column, base_column+'_'+count_column+'_sum']
    df = df.merge(tp, on=base_column, how='left')
    return df
def count_std(df,base_column,count_column):
    tp = df.groupby(base_column).agg({count_column: ['std']}).reset_index()
    tp.columns = [base_column, base_column+'_'+count_column+'_std']
    df = df.merge(tp, on=base_column, how='left')
    return df


train=pd.read_csv('./data/simple_train_R04_jet.csv')#,encoding = 'UTF-8')
test=pd.read_csv('./data/simple_test_R04_jet.csv')#,encoding = 'UTF-8')

def energy(df):
    x=df['jet_px']
    y=df['jet_py']
    z= df['jet_pz']
    return (x**2+y**2+z**2)**0.5
train['energy']=train.apply(energy,axis=1)
test['energy']=test.apply(energy,axis=1)


train['x_n']=train['jet_px']/train['energy']
train['y_n']=train['jet_py']/train['energy']
train['z_n']=train['jet_pz']/train['energy']

test['x_n']=test['jet_px']/test['energy']
test['y_n']=test['jet_py']/test['energy']
test['z_n']=test['jet_pz']/test['energy']



def x_sub_mean_del_std(df):
    df_mean = df.mean()
    df_std = df.std()
    return df.apply(lambda x:(x-df_mean) / df_std)


train['energy_sub_mean_del_std'] =  x_sub_mean_del_std(train['jet_energy'])
test['energy_sub_mean_del_std'] = x_sub_mean_del_std(test['jet_energy'])

train=count_mean(train,'event_id','energy_sub_mean_del_std')
train=count_sum(train,'event_id','energy_sub_mean_del_std')
train=count_std(train,'event_id','energy_sub_mean_del_std')
train=count_count(train,'event_id','energy_sub_mean_del_std')

test=count_mean(test,'event_id','energy_sub_mean_del_std')
test=count_sum(test,'event_id','energy_sub_mean_del_std')
test=count_std(test,'event_id','energy_sub_mean_del_std')
test=count_count(test,'event_id','energy_sub_mean_del_std')

train['mass_sub_mean_del_std'] =  x_sub_mean_del_std(train['jet_mass'])
test['mass_sub_mean_del_std'] = x_sub_mean_del_std(test['jet_mass'])


train=count_mean(train,'event_id','mass_sub_mean_del_std')
train=count_sum(train,'event_id','mass_sub_mean_del_std')
train=count_std(train,'event_id','mass_sub_mean_del_std')
train=count_count(train,'event_id','mass_sub_mean_del_std')

test=count_mean(test,'event_id','mass_sub_mean_del_std')
test=count_sum(test,'event_id','mass_sub_mean_del_std')
test=count_std(test,'event_id','mass_sub_mean_del_std')
test=count_count(test,'event_id','mass_sub_mean_del_std')


# def danwei(df):
# #    df_max = df.max()
# #    df_min = df.min()
#     # x = df['jet_px']
#     # y = df['jet_py']
#     # z = df['jet_pz']
#     x_norm = df.apply(lambda x: x['jet_px']/(x['jet_px']**2+x['jet_py']**2+x['jet_pz']**2 )**0.5)
#     y_norm = df.apply(lambda x: x['jet_py']/(x['jet_px']**2+x['jet_py']**2+x['jet_pz']**2 )**0.5)
#     z_norm = df.apply(lambda x: x['jet_pz']/(x['jet_px']**2+x['jet_py']**2+x['jet_pz']**2 )**0.5)
#     return x_norm,y_norm,z_norm
train['distence'] = (train['jet_px']**2 + train['jet_py']**2 + train['jet_pz']**2)**0.5
train['x_d'] = train['jet_px'] / train['distence']
train['y_d'] = train['jet_py'] / train['distence']
train['z_d'] = train['jet_pz'] / train['distence']
# train['y_d'] = danwei(train['jet_py'])
# train['z_d'] = danwei(train['jet_pz'])


#train['x_d'],train['y_d'],train['z_d'] = danwei(train)

#test['x_d'] = danwei(test['jet_px'])
#test['y_d'] = danwei(test['jet_py'])
#test['z_d'] = danwei(test['jet_pz'])
#test['x_d'],test['y_d'],test['z_d'] = danwei(test)

test['distence'] = (test['jet_px']**2 + test['jet_py']**2 + test['jet_pz']**2)**0.5
test['x_d'] = test['jet_px'] / test['distence']
test['y_d'] = test['jet_py'] / test['distence']
test['z_d'] = test['jet_pz'] / test['distence']


train['x_energy'] = train['x_d'] * train['jet_energy']
train['y_energy'] = train['y_d'] * train['jet_energy']
train['z_energy'] = train['z_d'] * train['jet_energy']

test['x_energy'] = test['x_d'] * test['jet_energy']
test['y_energy'] = test['y_d'] * test['jet_energy']
test['z_energy'] = test['z_d'] * test['jet_energy']


train=count_mean(train,'event_id','distence')
train=count_sum(train,'event_id','distence')
train=count_std(train,'event_id','distence')
train=count_count(train,'event_id','distence')


test=count_mean(test,'event_id','distence')
test=count_sum(test,'event_id','distence')
test=count_std(test,'event_id','distence')
test=count_count(test,'event_id','distence')

train=count_mean(train,'event_id','x_energy')
train=count_sum(train,'event_id','x_energy')
train=count_std(train,'event_id','x_energy')
train=count_count(train,'event_id','x_energy')

train=count_mean(train,'event_id','y_energy')
train=count_sum(train,'event_id','y_energy')
train=count_std(train,'event_id','y_energy')
train=count_count(train,'event_id','y_energy')

train=count_mean(train,'event_id','z_energy')
train=count_sum(train,'event_id','z_energy')
train=count_std(train,'event_id','z_energy')
train=count_count(train,'event_id','z_energy')




test=count_mean(test,'event_id','x_energy')
test=count_sum(test,'event_id','x_energy')
test=count_std(test,'event_id','x_energy')
test=count_count(test,'event_id','x_energy')

test=count_mean(test,'event_id','y_energy')
test=count_sum(test,'event_id','y_energy')
test=count_std(test,'event_id','y_energy')
test=count_count(test,'event_id','y_energy')

test=count_mean(test,'event_id','z_energy')
test=count_sum(test,'event_id','z_energy')
test=count_std(test,'event_id','z_energy')
test=count_count(test,'event_id','z_energy')


train=count_mean(train,'event_id','x_d')
train=count_sum(train,'event_id','x_d')
train=count_std(train,'event_id','x_d')
train=count_count(train,'event_id','x_d')



train=count_mean(train,'event_id','y_d')
train=count_sum(train,'event_id','y_d')
train=count_std(train,'event_id','y_d')
train=count_count(train,'event_id','y_d')



train=count_mean(train,'event_id','z_d')
train=count_sum(train,'event_id','z_d')
train=count_std(train,'event_id','z_d')
train=count_count(train,'event_id','z_d')


test=count_mean(test,'event_id','x_d')
test=count_sum(test,'event_id','x_d')
test=count_std(test,'event_id','x_d')
test=count_count(test,'event_id','x_d')



test=count_mean(test,'event_id','y_d')
test=count_sum(test,'event_id','y_d')
test=count_std(test,'event_id','y_d')
test=count_count(test,'event_id','y_d')



test=count_mean(test,'event_id','z_d')
test=count_sum(test,'event_id','z_d')
test=count_std(test,'event_id','z_d')
test=count_count(test,'event_id','z_d')




train=count_mean(train,'event_id','x_n')
train=count_sum(train,'event_id','x_n')
train=count_std(train,'event_id','x_n')
train=count_count(train,'event_id','x_n')

train=count_mean(train,'event_id','y_n')
train=count_sum(train,'event_id','y_n')
train=count_std(train,'event_id','y_n')
train=count_count(train,'event_id','y_n')

train=count_mean(train,'event_id','z_n')
train=count_sum(train,'event_id','z_n')
train=count_std(train,'event_id','z_n')
train=count_count(train,'event_id','z_n')

test=count_mean(test,'event_id','x_n')
test=count_sum(test,'event_id','x_n')
test=count_std(test,'event_id','x_n')
test=count_count(test,'event_id','x_n')



test=count_mean(test,'event_id','y_n')
test=count_sum(test,'event_id','y_n')
test=count_std(test,'event_id','y_n')
test=count_count(test,'event_id','y_n')


test=count_mean(test,'event_id','z_n')
test=count_sum(test,'event_id','z_n')
test=count_std(test,'event_id','z_n')
test=count_count(test,'event_id','z_n')


train['abs']=train['jet_energy']-train['energy']
test['abs']=test['jet_energy']-test['energy']

train['energy_sum'] = train['jet_energy']+train['energy']
test['energy_sum'] = test['jet_energy']+test['energy']



train['energy_every'] = train['energy_sum'] / train['number_of_particles_in_this_jet']
test['energy_every'] = test['energy_sum'] / test['number_of_particles_in_this_jet']

train['mul_energy_mass'] = train['jet_energy'] * train['jet_mass']
test['mul_energy_mass'] = test['jet_energy'] * test['jet_mass']

train['V'] = train['jet_energy'] / train['jet_mass']
test['V'] = test['jet_energy'] / test['jet_mass']

train['mvv'] = train['V']**2 * train['jet_mass']
test['mvv'] = test['V']**2 * test['jet_mass']






train = count_mean(train,'event_id','mvv')
train = count_sum(train,'event_id','mvv')
train = count_std(train,'event_id','mvv')
train = count_count(train,'event_id','mvv')

test = count_mean(test,'event_id','mvv')
test = count_sum(test,'event_id','mvv')
test = count_std(test,'event_id','mvv')
test = count_count(test,'event_id','mvv')



train = count_mean(train,'event_id','V')
train = count_sum(train,'event_id','V')
train = count_std(train,'event_id','V')
train = count_count(train,'event_id','V')

test = count_mean(test,'event_id','V')
test = count_sum(test,'event_id','V')
test = count_std(test,'event_id','V')
test = count_count(test,'event_id','V')




train['x_v'] = train['V'] / train['jet_px']
train['y_v'] = train['V'] / train['jet_py']
train['z_v'] = train['V'] / train['jet_pz']


test['x_v'] = test['V'] / test['jet_px']
test['y_v'] = test['V'] / test['jet_py']
test['z_v'] = test['V'] / test['jet_pz']



train = count_mean(train,'event_id','x_v')
train = count_sum(train,'event_id','x_v')
train = count_std(train,'event_id','x_v')
train = count_count(train,'event_id','x_v')


train = count_mean(train,'event_id','y_v')
train = count_sum(train,'event_id','y_v')
train = count_std(train,'event_id','y_v')
train = count_count(train,'event_id','y_v')


train = count_mean(train,'event_id','z_v')
train = count_sum(train,'event_id','z_v')
train = count_std(train,'event_id','z_v')
train = count_count(train,'event_id','z_v')



test = count_mean(test,'event_id','x_v')
test = count_sum(test,'event_id','x_v')
test = count_std(test,'event_id','x_v')
test = count_count(test,'event_id','x_v')


test = count_mean(test,'event_id','y_v')
test = count_sum(test,'event_id','y_v')
test = count_std(test,'event_id','y_v')
test = count_count(test,'event_id','y_v')


test = count_mean(test,'event_id','z_v')
test = count_sum(test,'event_id','z_v')
test = count_std(test,'event_id','z_v')
test = count_count(test,'event_id','z_v')



train = count_mean(train,'event_id','mul_energy_mass')
train = count_sum(train,'event_id','mul_energy_mass')
train = count_std(train,'event_id','mul_energy_mass')
train = count_count(train,'event_id','mul_energy_mass')




test = count_mean(test,'event_id','mul_energy_mass')
test = count_sum(test,'event_id','mul_energy_mass')
test = count_std(test,'event_id','mul_energy_mass')
test = count_count(test,'event_id','mul_energy_mass')

train = count_mean(train,'event_id','energy_every')
train = count_sum(train,'event_id','energy_every')
train = count_std(train,'event_id','energy_every')
train = count_count(train,'event_id','energy_every')


test = count_mean(test,'event_id','energy_every')
test = count_sum(test,'event_id','energy_every')
test = count_std(test,'event_id','energy_every')
test = count_count(test,'event_id','energy_every')





train = count_mean(train,'event_id','energy_sum')
train = count_sum(train,'event_id','energy_sum')
train = count_std(train,'event_id','energy_sum')
train = count_count(train,'event_id','energy_sum')

test = count_mean(test,'event_id','energy_sum')
test = count_sum(test,'event_id','energy_sum')
test = count_std(test,'event_id','energy_sum')
test = count_count(test,'event_id','energy_sum')

train=count_mean(train,'event_id','number_of_particles_in_this_jet')
train=count_sum(train,'event_id','number_of_particles_in_this_jet')
train=count_std(train,'event_id','number_of_particles_in_this_jet')
train=count_count(train,'event_id','number_of_particles_in_this_jet')

train=count_mean(train,'event_id','jet_mass')
train=count_sum(train,'event_id','jet_mass')
train=count_std(train,'event_id','jet_mass')
train=count_count(train,'event_id','jet_mass')

train=count_mean(train,'event_id','jet_energy')
train=count_sum(train,'event_id','jet_energy')
train=count_std(train,'event_id','jet_energy')
train=count_count(train,'event_id','jet_energy')

train['mean_energy']=train['jet_energy']/train['number_of_particles_in_this_jet']
train['mean_jet_mass']=train['jet_mass']/train['number_of_particles_in_this_jet']
train=count_mean(train,'event_id','mean_energy')
train=count_sum(train,'event_id','mean_energy')
train=count_std(train,'event_id','mean_energy')
train=count_count(train,'event_id','mean_energy')



train=count_mean(train,'event_id','mean_jet_mass')
train=count_sum(train,'event_id','mean_jet_mass')
train=count_std(train,'event_id','mean_jet_mass')
train=count_count(train,'event_id','mean_jet_mass')


train=count_mean(train,'event_id','abs')
train=count_sum(train,'event_id','abs')
train=count_std(train,'event_id','abs')
train=count_count(train,'event_id','abs')


train=count_mean(train,'event_id','energy')
train=count_sum(train,'event_id','energy')
train=count_std(train,'event_id','energy')
train=count_count(train,'event_id','energy')






test=count_mean(test,'event_id','number_of_particles_in_this_jet')
test=count_sum(test,'event_id','number_of_particles_in_this_jet')
test=count_std(test,'event_id','number_of_particles_in_this_jet')
test=count_count(test,'event_id','number_of_particles_in_this_jet')


test=count_mean(test,'event_id','jet_mass')
test=count_sum(test,'event_id','jet_mass')
test=count_std(test,'event_id','jet_mass')
test=count_count(test,'event_id','jet_mass')



test=count_mean(test,'event_id','jet_energy')
test=count_sum(test,'event_id','jet_energy')
test=count_std(test,'event_id','jet_energy')
test=count_count(test,'event_id','jet_energy')



test['mean_energy']=test['jet_energy']/test['number_of_particles_in_this_jet']
test['mean_jet_mass']=test['jet_mass']/test['number_of_particles_in_this_jet']




test=count_mean(test,'event_id','mean_energy')
test=count_sum(test,'event_id','mean_energy')
test=count_std(test,'event_id','mean_energy')
test=count_count(test,'event_id','mean_energy')


test=count_mean(test,'event_id','mean_jet_mass')
test=count_sum(test,'event_id','mean_jet_mass')
test=count_std(test,'event_id','mean_jet_mass')
test=count_count(test,'event_id','mean_jet_mass')


test=count_mean(test,'event_id','abs')
test=count_sum(test,'event_id','abs')
test=count_std(test,'event_id','abs')
test=count_count(test,'event_id','abs')

test=count_mean(test,'event_id','energy')
test=count_sum(test,'event_id','energy')
test=count_std(test,'event_id','energy')
test=count_count(test,'event_id','energy')



#d={1:[1,0,0,0,],4:[0,1,0,0],5:[0,0,1,0],21:[0,0,0,1]}
d={1:0,4:1,5:2,21:3}
def label_process(x):
    x=d[x]
    return x
train['label']=train['label'].apply(label_process)
#train_y=train.pop('label').values
#train_y=np.array(list(train_y))
_=train.pop('jet_id')
test_id=test.pop('jet_id')
_=train.pop('event_id')
_=test.pop('event_id')
#train=train.values
#test=test.values


#train.to_csv('train_xy.csv')
train_target_21 = train[train.label==3]
train_target_1 = train[train.label==0]
train_target_4 = train[train.label==1]
train_target_5 = train[train.label==2]
print(train_target_21.shape)
print(train_target_1.shape)
print(train_target_4.shape)
print(train_target_5.shape)
#test_data = pd.read_csv(test_path,index_col="id")


xgb = XGBClassifier(nthread=16)

res = pd.DataFrame(index=test.index,columns=['id','label'])
#train_p = pd.DataFrame(index=train_data.index)
'''
21    358600
1     261207
4     260186
5     254562
Name: label, dtype: int64

'''


estimator = LGBMRegressor(num_leaves=31)
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}
#xgb = GridSearchCV(xgb, param_grid, scoring='roc_auc')


xgb_param_dist = {'n_estimators':range(80,200,4),'max_depth':range(2,15,1),'learning_rate':np.linspace(0.01,2,20),'subsample':np.linspace(0.7,0.9,20),'colsample_bytree':np.linspace(0.5,0.98,10),'min_child_weight':range(1,9,1)}
gbm = GridSearchCV(estimator, param_grid)
gbdt = GradientBoostingClassifier(random_state=10)
catboost = CatBoostClassifier(
        iterations=2000,
        od_type='Iter',
        od_wait=120,
        max_depth=10,
        learning_rate=0.02,
        l2_leaf_reg=9,
        random_seed=2019,
        metric_period=50,
        fold_len_multiplier=1.1,
        loss_function='MultiClass',
        logging_level='Verbose'

    )
rfc = RandomForestClassifier(random_state=0)

def train(ite):
    print(i)
    data = train_target_21.sample(254562)#数据显示1 ：0 = 17：2（》0.5）
    data = data.append(train_target_1.sample(254562))
    data = data.append(train_target_4.sample(254562))
    data = data.append(train_target_5.sample(254562))
    
    y_ = data.label
    del data['label']
    if ite%3== 0:
        arg = xgb
    if ite%3==1:
        arg = gbm
    if ite % 3 == 2:
        arg = catboost
#    if ite % 4 == 3:
#        arg = rfc
    arg.fit(data,y_)
#    train_p[ite] = xgb.predict(train_data)
    res[ite] = arg.predict(test)
    del arg
    gc.collect()
#import threading
#print('start')
#for i in range(3):
#    for j in range(4):
#       threading.Thread(target=run, args=(i*3+j,)).start()
#print('end')
#    


for i in range(30):
   train(i)
sub=pd.DataFrame()
res = res.apply(lambda x: x.value_counts().index[0],axis =1)
res = res.apply(lambda x: int(x))
dd={0:1,1:4,2:5,3:21}
def sub_process(x):
    x=dd[x]
    return x
sub['label']=list(res)
sub['label']=sub['label'].apply(sub_process)
sub['id']=list(test_id)
#a = resT.apply(sum)
#a/3
#b= a/3
#c=pd.DataFrame(b)
res.to_csv('xgb_balence.csv')
sub.to_csv('sub_baseline.csv')