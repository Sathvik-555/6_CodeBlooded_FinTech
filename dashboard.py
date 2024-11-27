import streamlit as st, pandas as pd
import requests
from bs4 import BeautifulSoup
import time
st.header('Indian Stock Dashboard')

ticker = st.sidebar.text_input('Symbol code','INFY')
exchange = st.sidebar.text_input('Exchange','NSE')
url = f'https://www.google.com/finance/quote/{ticker}:NSE'


response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
price = float(soup.find(class_='YMlKec fxKbKc').text.strip()[1:].replace(",",""))
previous_close = float(soup.find(class_='P6K39c').text.strip()[1:].replace(",",""))
revenue = soup.find(class_='QXDnM').text
news = soup.find(class_='Yfwt5').text
about = soup.find(class_='bLLb2d').text

dict1 = {'Price':price,
         'Previous price':previous_close,
         'Revenue':revenue,
         'News':news,
         'About':about}

df = pd.DataFrame(dict1,index = ['Extracted data']).T
st.write(df)

