import pandas as pd
# 필요한 모듈 import 하기 
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import download_plotlyjs, plot
from plotly.graph_objs import *

url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)
res = requests.get(url)
res.encoding = 'utf-8'
res.status_code

soap = BeautifulSoup(res.text, 'lxml')

el_table_navi = soap.find("table", class_="Nnavi")
el_td_last = el_table_navi.find("td", class_="pgRR")
pg_last = el_td_last.a.get('href').rsplit('&')[1]
pg_last = pg_last.split('=')[1]
pg_last = int(pg_last)
pg_last

df = pd.DataFrame()
for page in range(1,pg_last):
    url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)     
    url = '{url}&page={page}'.format(url=url, page=page)
    print(url)
    df = df.append(pd.read_html(url, header=0)[0], ignore_index=True)
    
    
# df.dropna()를 이용해 결측값 있는 행 제거 
df = df.dropna() 

# 데이터의 타입을 int형으로 바꿔줌 
df[['종가', '전일비', '시가', '고가', '저가', '거래량']] = df[['종가', '전일비', '시가', '고가', '저가', '거래량']].astype(int) 

# 컬럼명 'date'의 타입을 date로 바꿔줌 
df['날짜'] = pd.to_datetime(df['날짜']) 

# 일자(date)를 기준으로 오름차순 정렬 
df = df.sort_values(by=['날짜'], ascending=True) 

# 상위 5개 데이터 확인 
df.head()



fig = px.line(df, x='날짜', y='종가', title='{}의 종가(close) Time Series'.format(company))

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=30, label="1주일", step="day", stepmode="backward"),
            dict(count=3, label="3개월", step="month", stepmode="backward"),
            dict(count=1, label="1년", step="year", stepmode="backward"),
            dict(count=3, label="3년", step="year", stepmode="backward"),
            dict(count=5, label="5년", step="year", stepmode="backward"),
            dict(label = "전체", step="all")
        ])
    )
)

fig.show()
fig.write_html("stocks.html")