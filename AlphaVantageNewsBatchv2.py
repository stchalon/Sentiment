import requests
import json
import time
import pandas as pd
from datetime import datetime, timedelta

# Alpha Vantage API Key
api_key = "PM3TEJGFH2WHWPIG"

# Date range
# 2024-09-01 to 2024-12-31
# 2025-01-01 to 2025-04-17
#
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 4, 17)

# Max requests per day
max_requests_per_day = 25

def format_date(date):
    return date.strftime("%Y%m%dT0130")

def append_line(filepath, text):
    with open(filepath, "a") as f:
        f.write(text + "\n")

# Batch processing
current_date = start_date
requests_made = 0
all_news = []

while current_date < end_date and requests_made < max_requests_per_day:
    next_date = current_date + timedelta(days=7)

    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=TSLA" \
          f"&time_from={format_date(current_date)}&time_to={format_date(next_date)}" \
          f"&limit=1000&apikey={api_key}"

    print(f"Fetching news from {current_date.date()} to {next_date.date()}")

    response = requests.get(url)
    data = response.json()

    if "feed" in data:
        for i, item in enumerate(data["feed"], start=1):
            print(f"{i:03d}: {item['time_published'][:8]} {float(item['overall_sentiment_score']):.3f} "
                  f"[{item['overall_sentiment_label']}] [{item['source'][:4]}] {item['title'][:50]}...")
            all_news.append(item)
    else:
        print("No news data available for this period.")

    current_date = next_date
    requests_made += 1

    if requests_made >= max_requests_per_day:
        print(f"Reached {max_requests_per_day} requests. Exiting loop.")
        break

# Save all news at once
if all_news:
    with open("c:/ml/AlphaVantageNewsBatchPrintFull.json", "w", encoding="utf-8") as f:
        json.dump(all_news, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(all_news)} news entries to AlphaVantageNewsBatchPrintFull.json")

print("Finished retrieving historical news.")
