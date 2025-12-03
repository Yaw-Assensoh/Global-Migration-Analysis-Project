import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://www.worldometers.info/world-population/population-by-country/"
response = requests.get(url)

soup = BeautifulSoup(response.text, "lxml")

table = soup.find("table")
rows = table.find_all("tr")

data = []
for row in rows[1:]:
    cols = [col.text.strip() for col in row.find_all("td")]
    data.append(cols)

headers = [th.text.strip() for th in table.find_all("th")]

df = pd.DataFrame(data, columns=headers)
df.to_csv("data/supply_raw.csv", index=False)

print("scrape_complete")

