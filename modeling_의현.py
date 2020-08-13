# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:12:13 2020

@author: river
"""


import requests
from bs4 import BeautifulSoup
import pandas as pd
import zipfile
import io
import xml.etree.ElementTree as et

def get_rank():
    web=requests.get("https://finance.naver.com/sise/lastsearch2.nhn")
    soup=BeautifulSoup(web.content,"html.parser")
    
    no_list=[]
    name_list=[]
    
    fi_rank={}
    
    a=soup.find_all("td")
    print(a)
    
    for x in range(0,29):
        name_list.append(soup.select(".tltle")[x].get_text())
        no_list.append(int(soup.select(".no")[x].get_text()))
        fi_rank[no_list[x]]=name_list[x]
        
    return fi_rank

def get_outlier(old,new):
    outlier_list=[]
    
    old2={value:key for key, value in old.items()}
    new2={value:key for key, value in new.items()}
    
    
    for value in new.values():
        if value not in old.values():
            outlier_list.append(value)

    [outlier_list.append(k) for k in new2 if k in old2 and abs(old2[k]-new2[k])>=5]
    
    return outlier_list

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



old=get_rank()
new=get_rank()

outlier=get_outlier(old,new)

api_key='f41c1f1e770dd7404dfb83cfb1fc00139eb9ea15'
corpCode=get_corpcode(api_key) # 기업코드 DataFrame

corpCode[corpCode['회사명']==outlier[5]]['종목코드']
