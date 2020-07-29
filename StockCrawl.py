# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:39:42 2020

@author: river
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
import zipfile
import xml.etree.ElementTree as et
import json

   
"""
# 네이버 금융에서 종목 가격정보와 거래량을 가져오는 함수: get_price

def get_price(company_code):
    # count=1000에서 1000은 과거 1,000 영업일간의 데이터를 의미. 사용자가 조절 가능
    url = "https://fchart.stock.naver.com/sise.nhn?symbol={}&timeframe=day&count=1000&requestType=0".format(company_code)
    get_result = requests.get(url)
    bs_obj = BeautifulSoup(get_result.content, "html.parser")
    
    # information
    inf = bs_obj.select('item')
    columns = ['Date', 'Open' ,'High', 'Low', 'Close', 'Volume']
    df_inf = pd.DataFrame([], columns = columns, index = range(len(inf)))
    
    for i in range(len(inf)):
        df_inf.iloc[i] = str(inf[i]['data']).split('|')
    
    df_inf.index = pd.to_datetime(df_inf['Date'])
    
    return df_inf.drop('Date', axis=1).astype(float)
 
    
stock=pd.DataFrame()
stock=get_price('000720')
stock=stock.reset_index(drop=False, inplace=False)
"""



#url='https://opendart.fss.or.kr/api/fnlttSinglAcnt.json?crtfc_key='+api_key+'&corp_code=000720&bsns_year=2016&reprt_code=11011&fs_div=OFS'

def get_corpcode(crtfc_key):
    """ OpenDART 기업 고유번호 받아오기 return 값:주식코드를 가진 업체의 DataFrame """
    params = {'crtfc_key':crtfc_key}
    items = ["corp_code","corp_name","stock_code","modify_date"]
    item_names = ["고유번호","회사명","종목코드","수정일"]
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    res = requests.get(url,params=params)
    zfile = zipfile.ZipFile(io.BytesIO(res.content))
    fin = zfile.open(zfile.namelist()[0])
    root = et.fromstring(fin.read().decode('utf-8'))
    data = []
    for child in root:
        if len(child.find('stock_code').text.strip()) > 1: # 종목코드가 있는 경우
            data.append([])
            for item in items:
                data[-1].append(child.find(item).text)
    df = pd.DataFrame(data, columns=item_names)
    return df


def convertFnltt(url, items, item_names, params):
    res = requests.get(url, params)
    json_dict = json.loads(res.text)
    data = []
    if json_dict['status']=="000":
        for line in json_dict['list']:
            data.append([])
            for itm in items:
                if itm in line.keys():
                    data[-1].append(line[itm])
                else: data[-1].append('')
    df = pd.DataFrame(data,columns=item_names)
    return df



def get_fnlttSinglAcntAll(crtfc_key, corp_code, bsns_year, reprt_code, fs_div = "CFS"):
    items = ["rcept_no","reprt_code","bsns_year","corp_code","sj_div","sj_nm",
             "account_id","account_nm","account_detail","thstrm_nm", "thstrm_amount",
             "thstrm_add_amount","frmtrm_nm","frmtrm_amount", "frmtrm_q_nm",
             "frmtrm_q_amount","frmtrm_add_amount","bfefrmtrm_nm", "bfefrmtrm_amount","ord"]
    item_names = ["접수번호","보고서코드","사업연도","고유번호","재무제표구분", "재무제표명",
                  "계정ID","계정명","계정상세","당기명","당기금액", "당기누적금액","전기명",
                  "전기금액","전기명(분/반기)", "전기금액(분/반기)","전기누적금액","전전기명",
                  "전전기금액", "계정과목정렬순서"]
    params = {'crtfc_key':crtfc_key, 'corp_code':corp_code, 'bsns_year':bsns_year, 'reprt_code':reprt_code, 'fs_div':fs_div}
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json?"
    return convertFnltt(url,items,item_names,params)





api_key='f41c1f1e770dd7404dfb83cfb1fc00139eb9ea15'

corpCode=get_corpcode(api_key) # 기업코드 DataFrame

hyundaiArchi=corpCode[corpCode['회사명']=='현대건설']
hdArchi_code=hyundaiArchi['고유번호'][2229]

finan16=get_fnlttSinglAcntAll(api_key,hdArchi_code,2016, 11011,fs_div="OFS")
finan17=get_fnlttSinglAcntAll(api_key,hdArchi_code,2017, 11011,fs_div="OFS")
finan18=get_fnlttSinglAcntAll(api_key,hdArchi_code,2018, 11011,fs_div="OFS")
finan19=get_fnlttSinglAcntAll(api_key,hdArchi_code,2019, 11011,fs_div="OFS")

Total_finan=pd.concat([finan16,finan17,finan18,finan19])

corpCode.to_csv("C:\\Users\\river\\Desktop\\Hanium\\HaniumCode\\corpCode.csv",encoding='ms949')
Total_finan.to_csv("C:\\Users\\river\\Desktop\\Hanium\\HaniumCode\\현대건설_재무제표.csv",encoding='ms949')
