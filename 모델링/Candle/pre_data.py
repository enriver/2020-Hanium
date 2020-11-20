# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 22:17:44 2020

@author: seungjun
"""

import FinanceDataReader as fdr
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import numpy as np
import os 
import datetime
import sqlite3
import datetime
from tqdm import tqdm
def naver():
    nav= pd.DataFrame(columns = ['1_kd','2_kd','3_kd','4_kd','5_kd','1_kp','2_kp','3_kp','4_kp','5_kp'])
    
    url_set={'상승' : 'rise', '보합' : 'steady', '하락' : 'fall',
             '급등' : 'low_up',
             '급락' : 'high_down', '거래상위' : 'quant', '거래증가' : 'quant_high', '거래감소' : 'quant_low'}
    market={'코스피' : '0', '코스닥' : '1'}
    
    url_set=['low_up','high_down','quant','quant_high','quant_low']
    market = ['0','1']
    
    for i in range(len(nav.columns)):
        if i <5:
            
            count1 = i
            count2 = 0
        else:
            count1 = i-5
            count2 = 1
    
        
        #테이블 불러오기
        try:
            url = 'https://finance.naver.com/sise/sise_'+url_set[count1]+'.nhn?sosok='+market[count2]
        
            source_code = requests.get(url).text
            html = BeautifulSoup(source_code, "html.parser")
    
            table = html.select('table.type_2')[0]
            #aa=table.select('tr')
        
            #이름 불러오기
            title = table.select('a.tltle')
            #price = table.select('td.number')
            index=[]
            num=0
            names=[]
            for j in title:
                num+=1
                index.append(num)
                names.append(j.text)
            
            nav[nav.columns[i]]=names
    
            print(nav.head(10))
        except:
            print('cant connect to finance data by wifi')
            print('plz try again')
            exit()
    return nav.head(100)


def bollinger(period, pb, pre, min_per,start_date, df_result):
    #before_df = df_result
    #df_krx = fdr.StockListing('KRX')
    #top=pd.merge(df_krx, before_df, on='Name')
    #bu=top['Symbol']
    bu = list(df_result)
    top_list=[]
    for i in tqdm(range(len(bu))):
        df=[]
        df=fdr.DataReader(bu[i], '2020-09-01')
        df['Moving Average']=df['Close'].rolling(window=period, min_periods = min_per).mean()
        df['Std'] = df['Close'].rolling(window=period, min_periods =min_per).std()
        
        df['Upper Band'] = df['Moving Average'] + (df['Std']*pb)
        df['Lower Band'] = df['Moving Average'] - (df['Std']*pb)
        df['Bollinger']=''
        for j in range(pre+min_per,len(df.index)):
            a = df.loc[df.index[j-pre:j],('Close')] > df.loc[df.index[j-pre:j],('Upper Band')]
            b = df.loc[df.index[j-pre:j],('Close')] < df.loc[df.index[j-pre:j],('Lower Band')]
            
            if df.loc[df.index[j],('Close')]<df.loc[df.index[j],('Upper Band')] and a.all()==True:
                df['Bollinger'][df.index[j]]='Sell'
            if df.loc[df.index[j],('Close')]>df.loc[df.index[j],('Lower Band')] and b.all()==True:
                df['Bollinger'][df.index[j]]='Buy'
        top_list.append(df)
    return top_list

def second_check(top_list, top):
    Sale_com=[]
    Buy_com=[]
    tops = list(top)
    for i in range(len(tops)):
        last = top_list[i].tail(1)['Bollinger']
        ss = last.values=='Sell'
        kk = last.values=='Buy'
        if ss.any():
            Sale_com.append(tops[i])
        if kk.any():
            Buy_com.append(tops[i])
           
    return Sale_com, Buy_com


    
    