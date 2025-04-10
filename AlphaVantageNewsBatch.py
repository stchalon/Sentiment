import requests
import json
import time
import pandas as pd
from datetime import datetime, timedelta

# Alpha Vantage API Key
api_key = "<your API key>"

# Start date for news retrieval (50 news maximum)
# [2025-01-01, 2025-08-04] done
# [2024-09-01, 2024-12-31] done
start_date = datetime(2024, 9, 1)  # to update to starting date
end_date = datetime(2025, 12, 31)    # to update to ending date
#end_date = datetime.today()     # to update to ending date

# Max requests per day (for free API plan)
max_requests_per_day = 25

# Function to format date for API request
def format_date(date):
    return date.strftime("%Y%m%dT0130")

def append_line(filepath, text):
    with open(filepath, "a") as f:
        f.write(text + "\n")

# Batch processing
current_date = start_date
requests_made = 0

while current_date < end_date and requests_made < max_requests_per_day:
    next_date = current_date + timedelta(days=7)  # 7-day batches

    # Construct API request
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=TSLA" \
          f"&time_from={format_date(current_date)}&time_to={format_date(next_date)}" \
          f"&limit=1000&apikey={api_key}"

    print(f"Fetching news from {current_date.date()} to {next_date.date()}")

    # Make request
    response = requests.get(url)
    data = response.json()

    if "feed" in data:
        for i, item in enumerate(data["feed"], start=1):
            print(f"{i:03d}: {item['time_published'][:8]} {float(item['overall_sentiment_score']):.3f} "
                  f"[{item['overall_sentiment_label']}] [{item['source'][:4]}] {item['title'][:50]}...")
            append_line("c:/ml/AlphaVantageNewsBatchPrint.txt",f"{i:03d}: {item['time_published'][:8]} {float(item['overall_sentiment_score']):.3f} "
                  f"[{item['overall_sentiment_label']}] [{item['source'][:20]}] {item['title'][:400]}...")
    
    else:
        print("No news data available for this period.")

    # Move to the next batch
    current_date = next_date
    requests_made += 1

    # Avoid exceeding daily limit
    if requests_made >= max_requests_per_day:
        print(f"Reached {max_requests_per_day} requests. Waiting until the next day (or abort ^C)...")
        time.sleep(86400)  # Sleep for 24 hours
        requests_made = 0  # Reset request count

print("Finished retrieving historical news.")

