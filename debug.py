import requests
from bs4 import BeautifulSoup

url = "https://finance.naver.com/item/news_news.naver?code=005930&page=1"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://finance.naver.com"
}
resp = requests.get(url, headers=headers, timeout=10)
resp.encoding = "euc-kr"
soup = BeautifulSoup(resp.text, "html.parser")

table = soup.find("table", class_="type5")
rows = table.find_all("tr")
print("tr 총 개수:", len(rows))

for i, row in enumerate(rows[:5]):
    print(f"\n[row {i}]", row)