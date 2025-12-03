import requests
from bs4 import BeautifulSoup
import pandas as pd

# STEP 1: Choose a page with a simple HTML table to scrape.
# (For exercise purposes we use a generic example table)
url = "https://www.worldometers.info/world-population/population-by-country/"

response = requests.get(url)
soup = BeautifulSoup(response.text, "lxml")

table = soup.find("table")

# Extract headers
headers = [th.text.strip() for th in table.find_all("th")]

# Extract rows
rows = []
for tr in table.find_all("tr")[1:]:
    tds = tr.find_all("td")
    if len(tds) == 0:
        continue
    rows.append([td.text.strip() for td in tds])

df = pd.DataFrame(rows, columns=headers)

# Save raw data into your project folder
df.to_csv("data/raw/supply_raw.csv", index=False)

print("Scrape complete! Raw file saved to data/raw/supply_raw.csv")

