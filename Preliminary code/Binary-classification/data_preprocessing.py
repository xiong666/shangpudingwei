"""
Created on Thu Oct 19 08:29:24 2017
@author: wuguilong
"""

import pandas as pd
import numpy as np
from datetime import date

evaluation = pd.read_csv('data/evaluation_public.csv', parse_dates=['time_stamp'], low_memory=False)
shop_info = pd.read_csv('data/ccf_first_round_shop_info.csv', low_memory=False)
user_shop_behavior = pd.read_csv('data/ccf_first_round_user_shop_behavior.csv', parse_dates=['time_stamp'], low_memory=False)


#first creat other features:mall_features;user_features;user_mall_features
t = user_shop_behavior[["user_id","shop_id"]]
t = t.merge(shop_info, how='left', on='shop_id')

#creat the mall features
mall = t['price'].groupby(t['mall_id']).agg('mean').reset_index()
mall.rename(columns={'price':'mall_price'},inplace=True)
t1 = t['longitude'].groupby(t['mall_id']).agg('mean').reset_index()
t2 = t['latitude'].groupby(t['mall_id']).agg('mean').reset_index()
mall = mall.merge(t1, how='left', on='mall_id')
mall = mall.merge(t2, how='left', on='mall_id')
mall.rename(columns={'longitude':'mall_longitude','latitude':'mall_latitude'},inplace=True)

#deal with the user_shop_behavior data ,creat the user_shop_behavior features
# get the datetime from time_stamp
def add_datetime_features(df):
#    df["transaction_year"] = df["time_stamp"].dt.year
#    df["transaction_month"] = df["time_stamp"].dt.month
    df["transaction_day"] = df["time_stamp"].dt.day
    df["transaction_hour"] = df["time_stamp"].dt.hour
    df["transaction_minute"] = df["time_stamp"].dt.minute
    df["transaction_isoweekday"] = df["time_stamp"].dt.date

    df.drop(["time_stamp"], inplace=True, axis=1)
    return df

user_shop_behavior = add_datetime_features(user_shop_behavior)

user_shop_behavior["transaction_isoweekday"] = user_shop_behavior.transaction_isoweekday.astype('str').apply(lambda x:x.split('-'))
user_shop_behavior["transaction_isoweekday"] = user_shop_behavior.transaction_isoweekday.apply(lambda x:date(int(x[0]),int(x[1]),int(x[2])).weekday()+1)

from math import radians, cos, sin, asin, sqrt  
def haversine(s):
    lon1, lat1, lon2, lat2 = s.split(':')
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * asin(sqrt(a))  
    r=6371
    return c*r*1000

user_shop_behavior = user_shop_behavior.merge(shop_info[["shop_id","mall_id"]], how='left', on=['shop_id'])
user_shop_behavior = user_shop_behavior.merge(mall, how='left', on=['mall_id'])

user_shop_behavior['distance'] = user_shop_behavior['longitude'].astype('str')+':'+user_shop_behavior['latitude'].astype('str')+':'+user_shop_behavior['mall_longitude'].astype('str')+':'+user_shop_behavior['mall_latitude'].astype('str')
user_shop_behavior['distance'] = user_shop_behavior['distance'].apply(haversine)
user_shop_behavior['longitude1'] = user_shop_behavior['longitude'].astype('str')+':'+'0'+':'+user_shop_behavior['mall_longitude'].astype('str')+':'+'0'
user_shop_behavior['longitude1'] = user_shop_behavior['longitude1'].apply(haversine)
user_shop_behavior['latitude1'] = user_shop_behavior['latitude'].astype('str')+':'+'0'+':'+user_shop_behavior['mall_latitude'].astype('str')+':'+'0'
user_shop_behavior['latitude1'] = user_shop_behavior['latitude1'].apply(haversine)
user_shop_behavior.drop(["mall_longitude"], inplace=True, axis=1)
user_shop_behavior.drop(["mall_latitude"], inplace=True, axis=1)
user_shop_behavior.drop(["mall_price"], inplace=True, axis=1)

#get the wifi features
user_shop_behavior["wifi_infosf"] = user_shop_behavior.wifi_infos.astype('str').apply(lambda x:x.split(';'))
user_shop_behavior["wifi_infos_number"] = user_shop_behavior.wifi_infosf.apply(lambda x:len(x))
user_shop_behavior["wifi_infosf"] = user_shop_behavior.wifi_infosf.apply(lambda x:sorted(x,key=lambda y:int(y.split('|')[1]),reverse=True))

# get the wififeature from time_stamp
def add_wififeature_features(df):
    
    df = df+["b_-999|-999|false"]*abs(10-len(df))
    return df

user_shop_behavior["wifi_infosf"] = user_shop_behavior.wifi_infosf.apply(add_wififeature_features)

user_shop_behavior["wifi1"] = user_shop_behavior["wifi_infosf"].apply(lambda x:x[0]).apply(lambda x:x.split('|'))
user_shop_behavior["wifi2"] = user_shop_behavior["wifi_infosf"].apply(lambda x:x[1]).apply(lambda x:x.split('|'))
user_shop_behavior["wifi3"] = user_shop_behavior["wifi_infosf"].apply(lambda x:x[2]).apply(lambda x:x.split('|'))
user_shop_behavior["wifi4"] = user_shop_behavior["wifi_infosf"].apply(lambda x:x[3]).apply(lambda x:x.split('|'))
user_shop_behavior["wifi5"] = user_shop_behavior["wifi_infosf"].apply(lambda x:x[4]).apply(lambda x:x.split('|'))
user_shop_behavior["wifi6"] = user_shop_behavior["wifi_infosf"].apply(lambda x:x[5]).apply(lambda x:x.split('|'))
user_shop_behavior["wifi7"] = user_shop_behavior["wifi_infosf"].apply(lambda x:x[6]).apply(lambda x:x.split('|'))
user_shop_behavior["wifi8"] = user_shop_behavior["wifi_infosf"].apply(lambda x:x[7]).apply(lambda x:x.split('|'))
user_shop_behavior["wifi9"] = user_shop_behavior["wifi_infosf"].apply(lambda x:x[8]).apply(lambda x:x.split('|'))
user_shop_behavior["wifi10"] = user_shop_behavior["wifi_infosf"].apply(lambda x:x[9]).apply(lambda x:x.split('|'))

user_shop_behavior.drop(["wifi_infosf"], inplace=True, axis=1)
user_shop_behavior["wifi1_bssid"] = user_shop_behavior["wifi1"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
user_shop_behavior["wifi1_signal"] = user_shop_behavior["wifi1"].apply(lambda x:x[1])
user_shop_behavior["wifi1_flag"] = user_shop_behavior["wifi1"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
user_shop_behavior.drop(["wifi1"], inplace=True, axis=1)

user_shop_behavior["wifi2_bssid"] = user_shop_behavior["wifi2"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
user_shop_behavior["wifi2_signal"] = user_shop_behavior["wifi2"].apply(lambda x:x[1])
user_shop_behavior["wifi2_flag"] = user_shop_behavior["wifi2"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
user_shop_behavior.drop(["wifi2"], inplace=True, axis=1)

user_shop_behavior["wifi3_bssid"] = user_shop_behavior["wifi3"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
user_shop_behavior["wifi3_signal"] = user_shop_behavior["wifi3"].apply(lambda x:x[1])
user_shop_behavior["wifi3_flag"] = user_shop_behavior["wifi3"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
user_shop_behavior.drop(["wifi3"], inplace=True, axis=1)

user_shop_behavior["wifi4_bssid"] = user_shop_behavior["wifi4"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
user_shop_behavior["wifi4_signal"] = user_shop_behavior["wifi4"].apply(lambda x:x[1])
user_shop_behavior["wifi4_flag"] = user_shop_behavior["wifi4"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
user_shop_behavior.drop(["wifi4"], inplace=True, axis=1)

user_shop_behavior["wifi5_bssid"] = user_shop_behavior["wifi5"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
user_shop_behavior["wifi5_signal"] = user_shop_behavior["wifi5"].apply(lambda x:x[1])
user_shop_behavior["wifi5_flag"] = user_shop_behavior["wifi5"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
user_shop_behavior.drop(["wifi5"], inplace=True, axis=1)

user_shop_behavior["wifi6_bssid"] = user_shop_behavior["wifi6"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
user_shop_behavior["wifi6_signal"] = user_shop_behavior["wifi6"].apply(lambda x:x[1])
user_shop_behavior["wifi6_flag"] = user_shop_behavior["wifi6"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
user_shop_behavior.drop(["wifi6"], inplace=True, axis=1)

user_shop_behavior["wifi7_bssid"] = user_shop_behavior["wifi7"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
user_shop_behavior["wifi7_signal"] = user_shop_behavior["wifi7"].apply(lambda x:x[1])
user_shop_behavior["wifi7_flag"] = user_shop_behavior["wifi7"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
user_shop_behavior.drop(["wifi7"], inplace=True, axis=1)

user_shop_behavior["wifi8_bssid"] = user_shop_behavior["wifi8"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
user_shop_behavior["wifi8_signal"] = user_shop_behavior["wifi8"].apply(lambda x:x[1])
user_shop_behavior["wifi8_flag"] = user_shop_behavior["wifi8"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
user_shop_behavior.drop(["wifi8"], inplace=True, axis=1)

user_shop_behavior["wifi9_bssid"] = user_shop_behavior["wifi9"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
user_shop_behavior["wifi9_signal"] = user_shop_behavior["wifi9"].apply(lambda x:x[1])
user_shop_behavior["wifi9_flag"] = user_shop_behavior["wifi9"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
user_shop_behavior.drop(["wifi9"], inplace=True, axis=1)

user_shop_behavior["wifi10_bssid"] = user_shop_behavior["wifi10"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
user_shop_behavior["wifi10_signal"] = user_shop_behavior["wifi10"].apply(lambda x:x[1])
user_shop_behavior["wifi10_flag"] = user_shop_behavior["wifi10"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
user_shop_behavior.drop(["wifi10"], inplace=True, axis=1)


#deal with the evaluation data
evaluation = add_datetime_features(evaluation)
evaluation["transaction_isoweekday"] = evaluation.transaction_isoweekday.astype('str').apply(lambda x:x.split('-'))
evaluation["transaction_isoweekday"] = evaluation.transaction_isoweekday.apply(lambda x:date(int(x[0]),int(x[1]),int(x[2])).weekday()+1)

evaluation = evaluation.merge(mall, how='left', on=['mall_id'])

evaluation['distance'] = evaluation['longitude'].astype('str')+':'+evaluation['latitude'].astype('str')+':'+evaluation['mall_longitude'].astype('str')+':'+evaluation['mall_latitude'].astype('str')
evaluation['distance'] = evaluation['distance'].apply(haversine)
evaluation['longitude1'] = evaluation['longitude'].astype('str')+':'+'0'+':'+evaluation['mall_longitude'].astype('str')+':'+'0'
evaluation['longitude1'] = evaluation['longitude1'].apply(haversine)
evaluation['latitude1'] = evaluation['latitude'].astype('str')+':'+'0'+':'+evaluation['mall_latitude'].astype('str')+':'+'0'
evaluation['latitude1'] = evaluation['latitude1'].apply(haversine)
evaluation.drop(["mall_longitude"], inplace=True, axis=1)
evaluation.drop(["mall_latitude"], inplace=True, axis=1)
evaluation.drop(["mall_price"], inplace=True, axis=1)

#get the wifi features
evaluation["wifi_infosf"] = evaluation.wifi_infos.astype('str').apply(lambda x:x.split(';'))
evaluation["wifi_infos_number"] = evaluation.wifi_infosf.apply(lambda x:len(x))
evaluation["wifi_infosf"] = evaluation.wifi_infosf.apply(lambda x:sorted(x,key=lambda y:int(y.split('|')[1]),reverse=True))

# get the wififeature from time_stamp
def add_wififeature_features(df):
    
    df = df+["b_-999|-999|false"]*abs(10-len(df))
    return df

evaluation["wifi_infosf"] = evaluation.wifi_infosf.apply(add_wififeature_features)

evaluation["wifi1"] = evaluation["wifi_infosf"].apply(lambda x:x[0]).apply(lambda x:x.split('|'))
evaluation["wifi2"] = evaluation["wifi_infosf"].apply(lambda x:x[1]).apply(lambda x:x.split('|'))
evaluation["wifi3"] = evaluation["wifi_infosf"].apply(lambda x:x[2]).apply(lambda x:x.split('|'))
evaluation["wifi4"] = evaluation["wifi_infosf"].apply(lambda x:x[3]).apply(lambda x:x.split('|'))
evaluation["wifi5"] = evaluation["wifi_infosf"].apply(lambda x:x[4]).apply(lambda x:x.split('|'))
evaluation["wifi6"] = evaluation["wifi_infosf"].apply(lambda x:x[5]).apply(lambda x:x.split('|'))
evaluation["wifi7"] = evaluation["wifi_infosf"].apply(lambda x:x[6]).apply(lambda x:x.split('|'))
evaluation["wifi8"] = evaluation["wifi_infosf"].apply(lambda x:x[7]).apply(lambda x:x.split('|'))
evaluation["wifi9"] = evaluation["wifi_infosf"].apply(lambda x:x[8]).apply(lambda x:x.split('|'))
evaluation["wifi10"] = evaluation["wifi_infosf"].apply(lambda x:x[9]).apply(lambda x:x.split('|'))

evaluation.drop(["wifi_infosf"], inplace=True, axis=1)
evaluation["wifi1_bssid"] = evaluation["wifi1"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
evaluation["wifi1_signal"] = evaluation["wifi1"].apply(lambda x:x[1])
evaluation["wifi1_flag"] = evaluation["wifi1"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
evaluation.drop(["wifi1"], inplace=True, axis=1)

evaluation["wifi2_bssid"] = evaluation["wifi2"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
evaluation["wifi2_signal"] = evaluation["wifi2"].apply(lambda x:x[1])
evaluation["wifi2_flag"] = evaluation["wifi2"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
evaluation.drop(["wifi2"], inplace=True, axis=1)

evaluation["wifi3_bssid"] = evaluation["wifi3"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
evaluation["wifi3_signal"] = evaluation["wifi3"].apply(lambda x:x[1])
evaluation["wifi3_flag"] = evaluation["wifi3"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
evaluation.drop(["wifi3"], inplace=True, axis=1)

evaluation["wifi4_bssid"] = evaluation["wifi4"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
evaluation["wifi4_signal"] = evaluation["wifi4"].apply(lambda x:x[1])
evaluation["wifi4_flag"] = evaluation["wifi4"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
evaluation.drop(["wifi4"], inplace=True, axis=1)

evaluation["wifi5_bssid"] = evaluation["wifi5"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
evaluation["wifi5_signal"] = evaluation["wifi5"].apply(lambda x:x[1])
evaluation["wifi5_flag"] = evaluation["wifi5"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
evaluation.drop(["wifi5"], inplace=True, axis=1)

evaluation["wifi6_bssid"] = evaluation["wifi6"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
evaluation["wifi6_signal"] = evaluation["wifi6"].apply(lambda x:x[1])
evaluation["wifi6_flag"] = evaluation["wifi6"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
evaluation.drop(["wifi6"], inplace=True, axis=1)

evaluation["wifi7_bssid"] = evaluation["wifi7"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
evaluation["wifi7_signal"] = evaluation["wifi7"].apply(lambda x:x[1])
evaluation["wifi7_flag"] = evaluation["wifi7"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
evaluation.drop(["wifi7"], inplace=True, axis=1)

evaluation["wifi8_bssid"] = evaluation["wifi8"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
evaluation["wifi8_signal"] = evaluation["wifi8"].apply(lambda x:x[1])
evaluation["wifi8_flag"] = evaluation["wifi8"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
evaluation.drop(["wifi8"], inplace=True, axis=1)

evaluation["wifi9_bssid"] = evaluation["wifi9"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
evaluation["wifi9_signal"] = evaluation["wifi9"].apply(lambda x:x[1])
evaluation["wifi9_flag"] = evaluation["wifi9"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
evaluation.drop(["wifi9"], inplace=True, axis=1)

evaluation["wifi10_bssid"] = evaluation["wifi10"].apply(lambda x:x[0]).apply(lambda x:x.replace('b_','')).astype('int')
evaluation["wifi10_signal"] = evaluation["wifi10"].apply(lambda x:x[1])
evaluation["wifi10_flag"] = evaluation["wifi10"].apply(lambda x:x[2]).apply(lambda x:x.replace('false','0')).apply(lambda x:x.replace('true','1')).astype('int')
evaluation.drop(["wifi10"], inplace=True, axis=1)

user_shop_behavior.to_csv('data/trainset.csv',index=None)
evaluation.to_csv('data/testset.csv',index=None)


