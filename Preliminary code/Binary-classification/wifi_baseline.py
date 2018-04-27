"""
Created on Sun Nov 12 06:19:41 2017
@author: long
"""
import pandas as pd
import numpy as np
from sklearn import  preprocessing
import lightgbm as lgb

offline_trainset = pd.read_csv('data/offline_trainset.csv')
offline_validset = pd.read_csv('data/offline_validset.csv')

shop_info = pd.read_csv('data/shop_info.csv')
offline_train_shop_feature = pd.read_csv('data/offline_train_shop_feature.csv')
offline_valid_shop_feature = pd.read_csv('data/offline_valid_shop_feature.csv')
offline_train_shop_wifi_feature = pd.read_csv('data/offline_train_shop_wifi_feature.csv')
offline_valid_shop_wifi_feature = pd.read_csv('data/offline_valid_shop_wifi_feature.csv')
shop_info.drop(["mall_id"], inplace=True, axis=1)  

offline_trainset = offline_trainset.merge(shop_info, how='left', on='shop_id')  
offline_trainset = offline_trainset.merge(offline_train_shop_feature, how='left', on='shop_id')  
offline_trainset["longitude_cha"] = offline_trainset["longitude"] - offline_trainset["slongitude"]
offline_trainset["latitude_cha"] = offline_trainset["latitude"] - offline_trainset["slatitude"]
offline_trainset["wifi_bssid_maxsignal_cha"] = offline_trainset["wifi1_bssid"] - offline_trainset["swifi_bssid_maxsignal"]
offline_trainset["wifi_signal_maxsignal_cha"] = offline_trainset["wifi1_signal"] - offline_trainset["swifi_signal_maxsignal"]
offline_trainset["dist"] = offline_trainset["longitude_cha"]**2 + offline_trainset["latitude_cha"]**2

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi1_bssid','wifi_signal':'sw_wifi1_signal','sw_number':'sw_number1'})
offline_trainset = offline_trainset.merge(sss, how='left', on=['shop_id','wifi1_bssid'])  
offline_trainset['sw_wifi1_signal_cha'] = offline_trainset['wifi1_signal'] - offline_trainset['sw_wifi1_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi2_bssid','wifi_signal':'sw_wifi2_signal','sw_number':'sw_number2'})
offline_trainset = offline_trainset.merge(sss, how='left', on=['shop_id','wifi2_bssid'])  
offline_trainset['sw_wifi2_signal_cha'] = offline_trainset['wifi2_signal'] - offline_trainset['sw_wifi2_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi3_bssid','wifi_signal':'sw_wifi3_signal','sw_number':'sw_number3'})
offline_trainset = offline_trainset.merge(sss, how='left', on=['shop_id','wifi3_bssid'])  
offline_trainset['sw_wifi3_signal_cha'] = offline_trainset['wifi3_signal'] - offline_trainset['sw_wifi3_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi4_bssid','wifi_signal':'sw_wifi4_signal','sw_number':'sw_number4'})
offline_trainset = offline_trainset.merge(sss, how='left', on=['shop_id','wifi4_bssid'])  
offline_trainset['sw_wifi4_signal_cha'] = offline_trainset['wifi4_signal'] - offline_trainset['sw_wifi4_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi5_bssid','wifi_signal':'sw_wifi5_signal','sw_number':'sw_number5'})
offline_trainset = offline_trainset.merge(sss, how='left', on=['shop_id','wifi5_bssid'])  
offline_trainset['sw_wifi5_signal_cha'] = offline_trainset['wifi5_signal'] - offline_trainset['sw_wifi5_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi6_bssid','wifi_signal':'sw_wifi6_signal','sw_number':'sw_number6'})
offline_trainset = offline_trainset.merge(sss, how='left', on=['shop_id','wifi6_bssid'])  
offline_trainset['sw_wifi6_signal_cha'] = offline_trainset['wifi6_signal'] - offline_trainset['sw_wifi6_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi7_bssid','wifi_signal':'sw_wifi7_signal','sw_number':'sw_number7'})
offline_trainset = offline_trainset.merge(sss, how='left', on=['shop_id','wifi7_bssid'])  
offline_trainset['sw_wifi7_signal_cha'] = offline_trainset['wifi7_signal'] - offline_trainset['sw_wifi7_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi8_bssid','wifi_signal':'sw_wifi8_signal','sw_number':'sw_number8'})
offline_trainset = offline_trainset.merge(sss, how='left', on=['shop_id','wifi8_bssid'])  
offline_trainset['sw_wifi8_signal_cha'] = offline_trainset['wifi8_signal'] - offline_trainset['sw_wifi8_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi9_bssid','wifi_signal':'sw_wifi9_signal','sw_number':'sw_number9'})
offline_trainset = offline_trainset.merge(sss, how='left', on=['shop_id','wifi9_bssid'])  
offline_trainset['sw_wifi9_signal_cha'] = offline_trainset['wifi9_signal'] - offline_trainset['sw_wifi9_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi10_bssid','wifi_signal':'sw_wifi10_signal','sw_number':'sw_number10'})
offline_trainset = offline_trainset.merge(sss, how='left', on=['shop_id','wifi10_bssid'])  
offline_trainset['sw_wifi10_signal_cha'] = offline_trainset['wifi10_signal'] - offline_trainset['sw_wifi10_signal']

###################################
import re
def swifi_split_for_bssid(line):
    line_splt=re.split(":|\|",line)
    bssid_tmp=[x for x in line_splt[0::1]]
    return bssid_tmp

def wifi_split_for_bssid(line):
    line_splt=re.split("_|;|\|",line)
    bssid_tmp=[x for x in line_splt[1::4]]
    return bssid_tmp

bssid=offline_trainset[['wifi_infos','row_id','swifi_infos']]
bssid['wifi_bssid_all']=bssid['wifi_infos'].astype('str').apply(wifi_split_for_bssid)
bssid['swifi_bssid_all']=bssid['swifi_infos'].astype('str').apply(swifi_split_for_bssid)

wifi_in_swifi=[]
for i in range(len(bssid)):
    same_set=set(bssid.at[i,'wifi_bssid_all'])&set(bssid.at[i,'swifi_bssid_all']) #common part
    ratio=float(len(same_set))/float(len(set(bssid.at[i,'wifi_bssid_all'])))
    wifi_in_swifi.append(ratio)
offline_trainset['wifi_in_swifi']=pd.Series(wifi_in_swifi)
offline_trainset["user_id"] = offline_trainset.user_id.astype('str').apply(lambda x:x.replace('u_','')).astype('int')      
offline_trainset["mall_id"] = offline_trainset.mall_id.astype('str').apply(lambda x:x.replace('m_','')).astype('int') 
offline_trainset["shop_id"] = offline_trainset.shop_id.astype('str').apply(lambda x:x.replace('s_','')).astype('int') 
#############################################################

offline_validset = offline_validset.merge(shop_info, how='left', on='shop_id')  
offline_validset = offline_validset.merge(offline_valid_shop_feature, how='left', on='shop_id')  
offline_validset["longitude_cha"] = offline_validset["longitude"] - offline_validset["slongitude"]
offline_validset["latitude_cha"] = offline_validset["latitude"] - offline_validset["slatitude"]
offline_validset["wifi_bssid_maxsignal_cha"] = offline_validset["wifi1_bssid"] - offline_validset["swifi_bssid_maxsignal"]
offline_validset["wifi_signal_maxsignal_cha"] = offline_validset["wifi1_signal"] - offline_validset["swifi_signal_maxsignal"]
offline_validset["dist"] = offline_validset["longitude_cha"]**2 + offline_validset["latitude_cha"]**2
sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi1_bssid','wifi_signal':'sw_wifi1_signal','sw_number':'sw_number1'})
offline_validset = offline_validset.merge(sss, how='left', on=['shop_id','wifi1_bssid'])  
offline_validset['sw_wifi1_signal_cha'] = offline_validset['wifi1_signal'] - offline_validset['sw_wifi1_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi2_bssid','wifi_signal':'sw_wifi2_signal','sw_number':'sw_number2'})
offline_validset = offline_validset.merge(sss, how='left', on=['shop_id','wifi2_bssid'])  
offline_validset['sw_wifi2_signal_cha'] = offline_validset['wifi2_signal'] - offline_validset['sw_wifi2_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi3_bssid','wifi_signal':'sw_wifi3_signal','sw_number':'sw_number3'})
offline_validset = offline_validset.merge(sss, how='left', on=['shop_id','wifi3_bssid'])  
offline_validset['sw_wifi3_signal_cha'] = offline_validset['wifi3_signal'] - offline_validset['sw_wifi3_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi4_bssid','wifi_signal':'sw_wifi4_signal','sw_number':'sw_number4'})
offline_validset = offline_validset.merge(sss, how='left', on=['shop_id','wifi4_bssid'])  
offline_validset['sw_wifi4_signal_cha'] = offline_validset['wifi4_signal'] - offline_validset['sw_wifi4_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi5_bssid','wifi_signal':'sw_wifi5_signal','sw_number':'sw_number5'})
offline_validset = offline_validset.merge(sss, how='left', on=['shop_id','wifi5_bssid'])  
offline_validset['sw_wifi5_signal_cha'] = offline_validset['wifi5_signal'] - offline_validset['sw_wifi5_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi6_bssid','wifi_signal':'sw_wifi6_signal','sw_number':'sw_number6'})
offline_validset = offline_validset.merge(sss, how='left', on=['shop_id','wifi6_bssid'])  
offline_validset['sw_wifi6_signal_cha'] = offline_validset['wifi6_signal'] - offline_validset['sw_wifi6_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi7_bssid','wifi_signal':'sw_wifi7_signal','sw_number':'sw_number7'})
offline_validset = offline_validset.merge(sss, how='left', on=['shop_id','wifi7_bssid'])  
offline_validset['sw_wifi7_signal_cha'] = offline_validset['wifi7_signal'] - offline_validset['sw_wifi7_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi8_bssid','wifi_signal':'sw_wifi8_signal','sw_number':'sw_number8'})
offline_validset = offline_validset.merge(sss, how='left', on=['shop_id','wifi8_bssid'])  
offline_validset['sw_wifi8_signal_cha'] = offline_validset['wifi8_signal'] - offline_validset['sw_wifi8_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi9_bssid','wifi_signal':'sw_wifi9_signal','sw_number':'sw_number9'})
offline_validset = offline_validset.merge(sss, how='left', on=['shop_id','wifi9_bssid'])  
offline_validset['sw_wifi9_signal_cha'] = offline_validset['wifi9_signal'] - offline_validset['sw_wifi9_signal']

sss = offline_train_shop_wifi_feature.rename(columns={'wifi_bssid':'wifi10_bssid','wifi_signal':'sw_wifi10_signal','sw_number':'sw_number10'})
offline_validset = offline_validset.merge(sss, how='left', on=['shop_id','wifi10_bssid'])  
offline_validset['sw_wifi10_signal_cha'] = offline_validset['wifi10_signal'] - offline_validset['sw_wifi10_signal']

###################################
bssid=offline_validset[['wifi_infos','row_id','swifi_infos']]
bssid['wifi_bssid_all']=bssid['wifi_infos'].astype('str').apply(wifi_split_for_bssid)
bssid['swifi_bssid_all']=bssid['swifi_infos'].astype('str').apply(swifi_split_for_bssid)

wifi_in_swifi=[]
for i in range(len(bssid)):
    same_set=set(bssid.at[i,'wifi_bssid_all'])&set(bssid.at[i,'swifi_bssid_all']) #common part
    ratio=float(len(same_set))/float(len(set(bssid.at[i,'wifi_bssid_all'])))
    wifi_in_swifi.append(ratio)
offline_validset['wifi_in_swifi']=pd.Series(wifi_in_swifi)
offline_validset["user_id"] = offline_validset.user_id.astype('str').apply(lambda x:x.replace('u_','')).astype('int')                
offline_validset["mall_id"] = offline_validset.mall_id.astype('str').apply(lambda x:x.replace('m_','')).astype('int') 
offline_validset["shop_id"] = offline_validset.shop_id.astype('str').apply(lambda x:x.replace('s_','')).astype('int') 
#############################################################
del bssid
le = preprocessing.LabelEncoder()
le.fit(offline_trainset.mall_id.append(offline_validset.mall_id))
offline_trainset['mall_id']=le.transform(offline_trainset.mall_id)
offline_validset['mall_id']=le.transform(offline_validset.mall_id)

le.fit(offline_trainset.wifi1_bssid.append(offline_validset.wifi1_bssid))
offline_trainset['wifi1_bssid']=le.transform(offline_trainset.wifi1_bssid)
offline_validset['wifi1_bssid']=le.transform(offline_validset.wifi1_bssid)

le.fit(offline_trainset.wifi2_bssid.append(offline_validset.wifi2_bssid))
offline_trainset['wifi2_bssid']=le.transform(offline_trainset.wifi2_bssid)
offline_validset['wifi2_bssid']=le.transform(offline_validset.wifi2_bssid)

le.fit(offline_trainset.wifi3_bssid.append(offline_validset.wifi3_bssid))
offline_trainset['wifi3_bssid']=le.transform(offline_trainset.wifi3_bssid)
offline_validset['wifi3_bssid']=le.transform(offline_validset.wifi3_bssid)

le.fit(offline_trainset.wifi4_bssid.append(offline_validset.wifi4_bssid))
offline_trainset['wifi4_bssid']=le.transform(offline_trainset.wifi4_bssid)
offline_validset['wifi4_bssid']=le.transform(offline_validset.wifi4_bssid)

le.fit(offline_trainset.wifi5_bssid.append(offline_validset.wifi5_bssid))
offline_trainset['wifi5_bssid']=le.transform(offline_trainset.wifi5_bssid)
offline_validset['wifi5_bssid']=le.transform(offline_validset.wifi5_bssid)

le.fit(offline_trainset.wifi6_bssid.append(offline_validset.wifi6_bssid))
offline_trainset['wifi6_bssid']=le.transform(offline_trainset.wifi6_bssid)
offline_validset['wifi6_bssid']=le.transform(offline_validset.wifi6_bssid)

le.fit(offline_trainset.wifi8_bssid.append(offline_validset.wifi8_bssid))
offline_trainset['wifi8_bssid']=le.transform(offline_trainset.wifi8_bssid)
offline_validset['wifi8_bssid']=le.transform(offline_validset.wifi8_bssid)

le.fit(offline_trainset.wifi7_bssid.append(offline_validset.wifi7_bssid))
offline_trainset['wifi7_bssid']=le.transform(offline_trainset.wifi7_bssid)
offline_validset['wifi7_bssid']=le.transform(offline_validset.wifi7_bssid)

le.fit(offline_trainset.wifi9_bssid.append(offline_validset.wifi9_bssid))
offline_trainset['wifi9_bssid']=le.transform(offline_trainset.wifi9_bssid)
offline_validset['wifi9_bssid']=le.transform(offline_validset.wifi9_bssid)

le.fit(offline_trainset.wifi10_bssid.append(offline_validset.wifi10_bssid))
offline_trainset['wifi10_bssid']=le.transform(offline_trainset.wifi10_bssid)
offline_validset['wifi10_bssid']=le.transform(offline_validset.wifi10_bssid)
    
params = {
'task': 'train',
'boosting_type': 'gbdt',
'objective': 'binary',
'metric': 'binary_logloss',
'metric_freq':1,
'is_training_metric':'false',
'max_bin':255,
'num_leaves': 128,
'learning_rate': 0.08,
'tree_learner':'serial',
'feature_fraction': 0.8,
'feature_fraction_seed':2,
'bagging_fraction': 0.8,
'bagging_freq': 5,
'min_date_in_leaf':100,
'min_sum_hessian_in_leaf':100,
'max_depth':7,
'early_stopping_round':50,
'verbose':0,
'is_unbanlance':'true' ,
}      
 
feature=[x for x in offline_trainset.columns if x not in ['label','shop_ID','time_stamp','row_id','wifi_infos','swifi_infos','wifi_infos_number','transaction_day', 'transaction_minute',
                                                          'wifi1_flag','wifi2_flag','wifi3_flag','wifi4_flag','wifi5_flag','wifi6_flag','wifi7_flag','wifi8_flag','wifi9_flag','wifi10_flag']]  
X_train = offline_trainset[feature]
y_train = offline_trainset['label']
X_test = offline_validset[feature]

lgb_train = lgb.Dataset(X_train, y_train)
gbm = lgb.train(params,
        lgb_train,
        num_boost_round=3000,
        valid_sets={lgb_train},
        )
del X_train,y_train
pred = gbm.predict(X_test)
offline_validset['label'] = np.array(pred)

offline_validset['label'] = offline_validset['shop_id'].astype('str')+':'+offline_validset['label'].astype('str')
val = offline_validset[['row_id','label']]
val = val.groupby(val['row_id'])['label'].apply(lambda x:'|'.join(x)).reset_index()
val["label"] = val.label.apply(lambda x:x.split('|')).apply(lambda x:sorted(x,key=lambda y:float(y.split(':')[1]),reverse=True)).apply(lambda x:x[0])
val["label"] = val["label"].astype('str').apply(lambda x:x.split(':'))
val["label"] = val["label"].apply(lambda x:x[0]).astype('int')

result = val[['row_id','label']]
result.rename(columns={'label':'shop_id'},inplace=True)
result.shop_id = 's_'+result.shop_id.astype('str')
result.row_id = result.row_id.astype('str')
result.to_csv('result.csv',index=False)

improtance = list(gbm.feature_importance())
gbm.save_model('data/model.txt')


