 
"""
Created on Sun Nov 12 21:05:07 2017
@author: long
"""
import pandas as pd
import numpy as np

shop_info = pd.read_csv('data/ccf_first_round_shop_info.csv')
df = pd.read_csv('data/trainset.csv')

#deal with the shop_info
shop_info["category_id"] = shop_info.category_id.astype('str').apply(lambda x:x.replace('c_','')).astype('int')
shop_info.rename(columns={'longitude' : 'slongitude'},inplace=True)
shop_info.rename(columns={'latitude' : 'slatitude'},inplace=True)
shop_info.to_csv('data/shop_info.csv',index=None)

offline_train_shop_feature = df[df.transaction_day<=24]
offline_valid_shop_feature = df

#############################################################for offline_train_shop_feature########################################################################################
#count for all shop-wifi
shop_wifiall = offline_train_shop_feature[['shop_id','wifi_infos']]
shop_wifiall = shop_wifiall.drop('wifi_infos', axis=1).join(shop_wifiall['wifi_infos'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('wifi_infos'))
shop_wifiall["wifi_infos"] = shop_wifiall["wifi_infos"].astype('str').apply(lambda x:x.split('|'))
shop_wifiall["wifi_bssid"] = shop_wifiall["wifi_infos"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_',''))
shop_wifiall["wifi_signal"] = shop_wifiall["wifi_infos"].apply(lambda x:x[1]).astype('int')
shop_wifiall["wifi_flag"] = shop_wifiall["wifi_infos"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
shop_wifiall.drop(["wifi_infos"], inplace=True, axis=1)
#num of wifi under 5 is drop
wifi_counts = pd.DataFrame(shop_wifiall.wifi_bssid.value_counts())
wifi_counts.rename(columns={'wifi_bssid' : 'wifi_counts'},inplace=True)
wifi_counts['wifi_bssid'] = wifi_counts.index
shop_wifiall = shop_wifiall.merge(wifi_counts, how='left', on=['wifi_bssid'])
shop_wifiall = shop_wifiall[shop_wifiall.wifi_counts>0]

#feature1:shop signal use all wifi
shop_signal = shop_wifiall[['shop_id','wifi_signal']].groupby(['shop_id']).agg('mean').reset_index()
shop_signal.rename(columns={'wifi_signal' : 'shop_wifi_signal'},inplace=True)

#feature swifi_infos
swifi_infos = shop_wifiall.drop_duplicates(['shop_id','wifi_bssid'])
swifi_infos = swifi_infos[['shop_id','wifi_bssid']]
swifi_infos = swifi_infos.groupby(swifi_infos['shop_id'])['wifi_bssid'].apply(lambda x:'|'.join(x)).reset_index()
swifi_infos.rename(columns={'wifi_bssid' : 'swifi_infos'},inplace=True)

#shop-wifi feature
shop_wifi_f = shop_wifiall[['shop_id','wifi_bssid','wifi_signal']].groupby(['shop_id','wifi_bssid']).agg('mean').reset_index()
shop_wifiall['sw_number'] =1
shop_wifi_num = shop_wifiall[['shop_id','wifi_bssid','sw_number']].groupby(['shop_id','wifi_bssid']).agg('sum').reset_index()
shop_wifi_num['sw_number'] = shop_wifi_num['sw_number']/24
shop_wifi_f = shop_wifi_f.merge(shop_wifi_num, how='left', on=['shop_id','wifi_bssid'])
shop_wifi_f.to_csv('data/offline_train_shop_wifi_feature.csv',index=None)

#feature2:shop wififeature use all wifi
shop_wifiall_signal = shop_wifiall.groupby(['shop_id','wifi_bssid']).agg('mean').reset_index()
shop_wifiall1 = shop_wifiall['wifi_bssid'].groupby(shop_wifiall['shop_id']).apply(lambda x:x.mode()).reset_index()
shop_wifiall1 = shop_wifiall1[['shop_id','wifi_bssid']].merge(shop_wifiall_signal, how='left', on=['shop_id','wifi_bssid'])
shop_wifiall1['wifi_bssid'] = shop_wifiall1['wifi_bssid'].astype('str')+':'+shop_wifiall1['wifi_signal'].astype('str')
val = shop_wifiall1.groupby(shop_wifiall1['shop_id'])['wifi_bssid'].apply(lambda x:'|'.join(x)).reset_index()
val["wifi_bssid"] = val.wifi_bssid.apply(lambda x:x.split('|')).apply(lambda x:sorted(x,key=lambda y:float(y.split(':')[1]),reverse=True)).apply(lambda x:x[0])
val["wifi_bssid"] = val["wifi_bssid"].astype('str').apply(lambda x:x.split(':'))
val["swifi_bssid_maxsignal"] = val["wifi_bssid"].apply(lambda x:x[0]).astype('int')
val["swifi_signal_maxsignal"] = val["wifi_bssid"].apply(lambda x:x[1]).astype('float')
val.drop(["wifi_bssid"], inplace=True, axis=1)
shop_wifiall = val

#feature3:shop signal use wifi1
shop_wifi1 = offline_train_shop_feature[['shop_id','wifi1_bssid','wifi1_signal']]
shop_signal_wifi1 = shop_wifi1.groupby(shop_wifi1['shop_id'])['wifi1_signal'].agg('mean').reset_index()
shop_signal_wifi1.rename(columns={'wifi1_signal' : 'swifi1_signal_all'},inplace=True)

#feature4:shop wifinum use wifi1
shop_wifi1_num = shop_wifi1.drop_duplicates(['shop_id','wifi1_bssid'])
shop_wifi1_num = shop_wifi1_num[['shop_id','wifi1_bssid']].groupby(shop_wifi1_num['shop_id']).apply(lambda x:len(x)).reset_index()
shop_wifi1_num.rename(columns={0:'shop_wifi_num'},inplace=True)
shop_wifi1_num['shop_wifi_num'] = shop_wifi1_num['shop_wifi_num'].astype('float')/24

#feature5:shop wififeature use wifi1
shop_wifi_signal = shop_wifi1.groupby(['shop_id','wifi1_bssid']).agg('mean').reset_index()
shop_wifi1 = shop_wifi1.groupby(shop_wifi1['shop_id'])['wifi1_bssid'].apply(lambda x:x.mode()).reset_index()
shop_wifi1 = shop_wifi1[['shop_id','wifi1_bssid']].merge(shop_wifi_signal, how='left', on=['shop_id','wifi1_bssid'])
shop_wifi1['wifi1_bssid'] = shop_wifi1['wifi1_bssid'].astype('str')+':'+shop_wifi1['wifi1_signal'].astype('str')
val = shop_wifi1.groupby(shop_wifi1['shop_id'])['wifi1_bssid'].apply(lambda x:'|'.join(x)).reset_index()
val["wifi1_bssid"] = val.wifi1_bssid.apply(lambda x:x.split('|')).apply(lambda x:sorted(x,key=lambda y:float(y.split(':')[1]),reverse=True)).apply(lambda x:x[0])
val["wifi1_bssid"] = val["wifi1_bssid"].astype('str').apply(lambda x:x.split(':'))
val["swifi1_bssid_maxsignal"] = val["wifi1_bssid"].apply(lambda x:x[0]).astype('int')
val["swifi1_signal_maxsignal"] = val["wifi1_bssid"].apply(lambda x:x[1]).astype('float')
val.drop(["wifi1_bssid"], inplace=True, axis=1)
shop_wifi1 = val

#merge all features
shop_wifi_feature = shop_wifiall.merge(shop_signal, how='left', on=['shop_id'])
shop_wifi_feature = shop_wifi_feature.merge(shop_signal_wifi1, how='left', on=['shop_id'])
shop_wifi_feature = shop_wifi_feature.merge(shop_wifi1_num, how='left', on=['shop_id'])
shop_wifi_feature = shop_wifi_feature.merge(shop_wifi1, how='left', on=['shop_id'])
shop_wifi_feature = shop_wifi_feature.merge(swifi_infos, how='left', on=['shop_id'])
shop_wifi_feature.to_csv('data/offline_train_shop_feature.csv',index=None)

#####################################################################for offline_valid_shop_feature##############################################################################
#count for all shop-wifi
shop_wifiall = offline_valid_shop_feature[['shop_id','wifi_infos']]
shop_wifiall = shop_wifiall.drop('wifi_infos', axis=1).join(shop_wifiall['wifi_infos'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('wifi_infos'))
shop_wifiall["wifi_infos"] = shop_wifiall["wifi_infos"].astype('str').apply(lambda x:x.split('|'))
shop_wifiall["wifi_bssid"] = shop_wifiall["wifi_infos"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_',''))
shop_wifiall["wifi_signal"] = shop_wifiall["wifi_infos"].apply(lambda x:x[1]).astype('int')
shop_wifiall["wifi_flag"] = shop_wifiall["wifi_infos"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
shop_wifiall.drop(["wifi_infos"], inplace=True, axis=1)
#num of wifi under 5 is drop
wifi_counts = pd.DataFrame(shop_wifiall.wifi_bssid.value_counts())
wifi_counts.rename(columns={'wifi_bssid' : 'wifi_counts'},inplace=True)
wifi_counts['wifi_bssid'] = wifi_counts.index
shop_wifiall = shop_wifiall.merge(wifi_counts, how='left', on=['wifi_bssid'])
shop_wifiall = shop_wifiall[shop_wifiall.wifi_counts>0]

#feature1:shop signal use all wifi
shop_signal = shop_wifiall[['shop_id','wifi_signal']].groupby(['shop_id']).agg('mean').reset_index()
shop_signal.rename(columns={'wifi_signal' : 'shop_wifi_signal'},inplace=True)

#feature swifi_infos
swifi_infos = shop_wifiall.drop_duplicates(['shop_id','wifi_bssid'])
swifi_infos = swifi_infos[['shop_id','wifi_bssid']]
swifi_infos = swifi_infos.groupby(swifi_infos['shop_id'])['wifi_bssid'].apply(lambda x:'|'.join(x)).reset_index()
swifi_infos.rename(columns={'wifi_bssid' : 'swifi_infos'},inplace=True)

#shop-wifi feature
shop_wifi_f = shop_wifiall[['shop_id','wifi_bssid','wifi_signal']].groupby(['shop_id','wifi_bssid']).agg('mean').reset_index()
shop_wifiall['sw_number'] =1
shop_wifi_num = shop_wifiall[['shop_id','wifi_bssid','sw_number']].groupby(['shop_id','wifi_bssid']).agg('sum').reset_index()
shop_wifi_num['sw_number'] = shop_wifi_num['sw_number']/31
shop_wifi_f = shop_wifi_f.merge(shop_wifi_num, how='left', on=['shop_id','wifi_bssid'])
shop_wifi_f.to_csv('data/offline_valid_shop_wifi_feature.csv',index=None)

#feature2:shop wififeature use all wifi
shop_wifiall_signal = shop_wifiall.groupby(['shop_id','wifi_bssid']).agg('mean').reset_index()
shop_wifiall1 = shop_wifiall['wifi_bssid'].groupby(shop_wifiall['shop_id']).apply(lambda x:x.mode()).reset_index()
shop_wifiall1 = shop_wifiall1[['shop_id','wifi_bssid']].merge(shop_wifiall_signal, how='left', on=['shop_id','wifi_bssid'])
shop_wifiall1['wifi_bssid'] = shop_wifiall1['wifi_bssid'].astype('str')+':'+shop_wifiall1['wifi_signal'].astype('str')
val = shop_wifiall1.groupby(shop_wifiall1['shop_id'])['wifi_bssid'].apply(lambda x:'|'.join(x)).reset_index()
val["wifi_bssid"] = val.wifi_bssid.apply(lambda x:x.split('|')).apply(lambda x:sorted(x,key=lambda y:float(y.split(':')[1]),reverse=True)).apply(lambda x:x[0])
val["wifi_bssid"] = val["wifi_bssid"].astype('str').apply(lambda x:x.split(':'))
val["swifi_bssid_maxsignal"] = val["wifi_bssid"].apply(lambda x:x[0]).astype('int')
val["swifi_signal_maxsignal"] = val["wifi_bssid"].apply(lambda x:x[1]).astype('float')
val.drop(["wifi_bssid"], inplace=True, axis=1)
shop_wifiall = val

#feature3:shop signal use wifi1
shop_wifi1 = offline_valid_shop_feature[['shop_id','wifi1_bssid','wifi1_signal']]
shop_signal_wifi1 = shop_wifi1.groupby(shop_wifi1['shop_id'])['wifi1_signal'].agg('mean').reset_index()
shop_signal_wifi1.rename(columns={'wifi1_signal' : 'swifi1_signal_all'},inplace=True)

#feature4:shop wifinum use wifi1
shop_wifi1_num = shop_wifi1.drop_duplicates(['shop_id','wifi1_bssid'])
shop_wifi1_num = shop_wifi1_num[['shop_id','wifi1_bssid']].groupby(shop_wifi1_num['shop_id']).apply(lambda x:len(x)).reset_index()
shop_wifi1_num.rename(columns={0:'shop_wifi_num'},inplace=True)
shop_wifi1_num['shop_wifi_num'] = shop_wifi1_num['shop_wifi_num'].astype('float')/31

#feature5:shop wififeature use wifi1
shop_wifi_signal = shop_wifi1.groupby(['shop_id','wifi1_bssid']).agg('mean').reset_index()
shop_wifi1 = shop_wifi1.groupby(shop_wifi1['shop_id'])['wifi1_bssid'].apply(lambda x:x.mode()).reset_index()
shop_wifi1 = shop_wifi1[['shop_id','wifi1_bssid']].merge(shop_wifi_signal, how='left', on=['shop_id','wifi1_bssid'])
shop_wifi1['wifi1_bssid'] = shop_wifi1['wifi1_bssid'].astype('str')+':'+shop_wifi1['wifi1_signal'].astype('str')
val = shop_wifi1.groupby(shop_wifi1['shop_id'])['wifi1_bssid'].apply(lambda x:'|'.join(x)).reset_index()
val["wifi1_bssid"] = val.wifi1_bssid.apply(lambda x:x.split('|')).apply(lambda x:sorted(x,key=lambda y:float(y.split(':')[1]),reverse=True)).apply(lambda x:x[0])
val["wifi1_bssid"] = val["wifi1_bssid"].astype('str').apply(lambda x:x.split(':'))
val["swifi1_bssid_maxsignal"] = val["wifi1_bssid"].apply(lambda x:x[0]).astype('int')
val["swifi1_signal_maxsignal"] = val["wifi1_bssid"].apply(lambda x:x[1]).astype('float')
val.drop(["wifi1_bssid"], inplace=True, axis=1)
shop_wifi1 = val

#merge all features
shop_wifi_feature = shop_wifiall.merge(shop_signal, how='left', on=['shop_id'])
shop_wifi_feature = shop_wifi_feature.merge(shop_signal_wifi1, how='left', on=['shop_id'])
shop_wifi_feature = shop_wifi_feature.merge(shop_wifi1_num, how='left', on=['shop_id'])
shop_wifi_feature = shop_wifi_feature.merge(shop_wifi1, how='left', on=['shop_id'])
shop_wifi_feature = shop_wifi_feature.merge(swifi_infos, how='left', on=['shop_id'])
shop_wifi_feature.to_csv('data/offline_valid_shop_feature.csv',index=None)
