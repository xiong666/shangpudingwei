"""
Created on Mon Nov 13 03:45:16 2017

@author: long
"""

import pandas as pd
import numpy as np
from sklearn import  preprocessing
from collections import Counter

shop = pd.read_csv('data/ccf_first_round_shop_info.csv')
df = pd.read_csv('data/trainset.csv')
test = pd.read_csv('data/testset.csv')
offline_trainset_multi = pd.read_csv('data/offline_trainset_multi.csv')
offline_validset_multi = pd.read_csv('data/offline_testset_multi.csv')

offline_train_feature = df[df.transaction_day<=24]
offline_trainset =  df[df.transaction_day>=25]
offline_valid_feature = df
offline_validset = test

#############################################################################################################################################
shop_wifiall = offline_train_feature[['shop_id','wifi_infos']]
shop_wifiall = shop_wifiall.drop('wifi_infos', axis=1).join(shop_wifiall['wifi_infos'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('wifi_infos'))
shop_wifiall["wifi_infos"] = shop_wifiall["wifi_infos"].astype('str').apply(lambda x:x.split('|'))
shop_wifiall["wifi_bssid"] = shop_wifiall["wifi_infos"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
shop_wifiall["wifi_signal"] = shop_wifiall["wifi_infos"].apply(lambda x:x[1]).astype('int')
shop_wifiall.drop(["wifi_infos"], inplace=True, axis=1)
wifi_counts = pd.DataFrame(shop_wifiall.wifi_bssid.value_counts())
wifi_counts.rename(columns={'wifi_bssid' : 'wifi_counts'},inplace=True)
wifi_counts['wifi_bssid'] = wifi_counts.index
shop_wifiall = shop_wifiall.merge(wifi_counts, how='left', on=['wifi_bssid'])
shop_wifiall = shop_wifiall[shop_wifiall.wifi_counts>5]
shop_wifiall = shop_wifiall.drop_duplicates()

shop_wifiall['shop_id'] = shop_wifiall['shop_id'].astype('str')+':'+shop_wifiall['wifi_signal'].astype('str')
shop_wifiall = shop_wifiall.groupby(shop_wifiall['wifi_bssid'])['shop_id'].apply(lambda x:'|'.join(x)).reset_index()
shop_wifiall["shop_id"] = shop_wifiall.shop_id.apply(lambda x:x.split('|')).apply(lambda x:sorted(x,key=lambda y:float(y.split(':')[1]),reverse=True)).apply(lambda x:x[0:12]).agg(lambda x:';'.join(x))
shop_wifiall=shop_wifiall.drop('shop_id', axis=1).join(shop_wifiall['shop_id'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('shop_id'))
shop_wifiall["shop_id"] = shop_wifiall["shop_id"].astype('str').apply(lambda x:x.split(':'))
shop_wifiall["shop_id"] = shop_wifiall["shop_id"].apply(lambda x:x[0])
shop_wifiall = shop_wifiall.drop_duplicates()

offline_trainset['row_id'] = np.array(range(len(offline_trainset)))
offline_trainset_1 = offline_trainset.copy()
offline_trainset_1['shop_ID'] = offline_trainset_1['shop_id']
offline_trainset_1['label'] = 1
offline_trainset_2 = offline_trainset.copy()
offline_trainset_2['shop_ID'] = offline_trainset_2['shop_id']
offline_trainset_2.drop(['shop_id'], inplace=True, axis=1)
offline_trainset['wifi_infos2'] = offline_trainset['wifi_infos']
offline_trainset = offline_trainset.drop('wifi_infos2', axis=1).join(offline_trainset['wifi_infos2'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('wifi_infos2'))
offline_trainset["wifi_infos2"] = offline_trainset["wifi_infos2"].astype('str').apply(lambda x:x.split('|'))
offline_trainset["wifi_bssid"] = offline_trainset["wifi_infos2"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
offline_trainset["wifi_signal"] = offline_trainset["wifi_infos2"].apply(lambda x:x[1]).astype('int')
offline_trainset.drop(["wifi_infos2",'wifi_signal'], inplace=True, axis=1)


offline_trainset.rename(columns={'shop_id' : 'shop_ID'},inplace=True)
offline_trainset = offline_trainset.merge(shop_wifiall[['shop_id','wifi_bssid']], how='left', on=['wifi_bssid'])
offline_trainset.drop(["wifi_bssid"], inplace=True, axis=1)
offline_trainset = offline_trainset.drop_duplicates()
offline_trainset = offline_trainset[offline_trainset.shop_id.notnull()]
offline_trainset['label'] = offline_trainset.shop_id.astype('str').apply(lambda x:x.replace('s_','')).astype('int') - offline_trainset.shop_ID.astype('str').apply(lambda x:x.replace('s_','')).astype('int')
offline_trainset['label'] = offline_trainset.label.apply(lambda x:1 if x==0 else 0)

offline_trainset = pd.concat([offline_trainset_1,offline_trainset])
offline_trainset = offline_trainset.drop_duplicates()

multi = offline_trainset_multi[offline_trainset_multi.shop_index==0][['row_id','shop_id','label']]
offline_trainset_2 = offline_trainset_2.merge(multi, how='left', on=['row_id'])
offline_trainset = pd.concat([offline_trainset_2,offline_trainset])
offline_trainset = offline_trainset.drop_duplicates()
offline_trainset.to_csv('data/offline_trainset.csv',index=None)

#############################################################################################################################################
shop_wifiall = offline_valid_feature[['shop_id','wifi_infos']]
shop_wifiall = shop_wifiall.drop('wifi_infos', axis=1).join(shop_wifiall['wifi_infos'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('wifi_infos'))
shop_wifiall["wifi_infos"] = shop_wifiall["wifi_infos"].astype('str').apply(lambda x:x.split('|'))
shop_wifiall["wifi_bssid"] = shop_wifiall["wifi_infos"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
shop_wifiall["wifi_signal"] = shop_wifiall["wifi_infos"].apply(lambda x:x[1]).astype('int')
shop_wifiall.drop(["wifi_infos"], inplace=True, axis=1)

def cc(shop):
    return pd.Series(Counter(shop).most_common(3)).apply(lambda x:x[0])

shop_wifiallmode = shop_wifiall['shop_id'].groupby(shop_wifiall['wifi_bssid']).apply(cc).reset_index()
shop_wifiall = shop_wifiall.drop_duplicates()

shop_wifiall['shop_id'] = shop_wifiall['shop_id'].astype('str')+':'+shop_wifiall['wifi_signal'].astype('str')
shop_wifiall = shop_wifiall.groupby(shop_wifiall['wifi_bssid'])['shop_id'].apply(lambda x:'|'.join(x)).reset_index()
shop_wifiall["shop_id"] = shop_wifiall.shop_id.apply(lambda x:x.split('|')).apply(lambda x:sorted(x,key=lambda y:float(y.split(':')[1]),reverse=True)).apply(lambda x:x[0:15]).agg(lambda x:';'.join(x))
shop_wifiall=shop_wifiall.drop('shop_id', axis=1).join(shop_wifiall['shop_id'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('shop_id'))
shop_wifiall["shop_id"] = shop_wifiall["shop_id"].astype('str').apply(lambda x:x.split(':'))
shop_wifiall["shop_id"] = shop_wifiall["shop_id"].apply(lambda x:x[0])
shop_wifiall = pd.concat([shop_wifiall,shop_wifiallmode[['shop_id','wifi_bssid']]])
shop_wifiall = shop_wifiall.drop_duplicates()


offline_validset_2 = offline_validset.copy()
offline_validset['wifi_infos2'] = offline_validset['wifi_infos']
offline_validset = offline_validset.drop('wifi_infos2', axis=1).join(offline_validset['wifi_infos2'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('wifi_infos2'))
offline_validset["wifi_infos2"] = offline_validset["wifi_infos2"].astype('str').apply(lambda x:x.split('|'))
offline_validset["wifi_bssid"] = offline_validset["wifi_infos2"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
offline_validset["wifi_signal"] = offline_validset["wifi_infos2"].apply(lambda x:x[1]).astype('int')
offline_validset.drop(["wifi_infos2",'wifi_signal'], inplace=True, axis=1)


offline_validset = offline_validset.merge(shop_wifiall[['shop_id','wifi_bssid']], how='left', on=['wifi_bssid'])
offline_validset.drop(["wifi_bssid"], inplace=True, axis=1)
offline_validset = offline_validset.drop_duplicates()
offline_validset = offline_validset[offline_validset.shop_id.notnull()]

multi = offline_validset_multi[offline_validset_multi.shop_index==0][['row_id','shop_id']]
offline_validset_2 = offline_validset_2.merge(multi, how='left', on=['row_id'])
offline_validset = pd.concat([offline_validset_2,offline_validset])

offline_validset = offline_validset.drop_duplicates()

offline_validset.to_csv('data/offline_validset.csv',index=None)