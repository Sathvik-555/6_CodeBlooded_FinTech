import requests
from bs4 import BeautifulSoup
import time
stock_ip = []

ticker = 'M%26M'
url = f'https://www.google.com/finance/quote/{ticker}:NSE'

while True:
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')
  class1 = "YMlKec fxKbKc"
  price = float(soup.find(class_=class1).text.strip()[1:].replace(",",""))
  stock_ip.append(price)
  print(price)
  time.sleep(3)
