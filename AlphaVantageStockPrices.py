import time
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

def get_daily_stock_prices(api_key, symbol, start_date, end_date):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    
    # Convert index to datetime and filter by date range
    data.index = pd.to_datetime(data.index)
    filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
    
    return filtered_data

def append_line(filepath, text):
    with open(filepath, "a") as f:
        f.write(text + "\n")

def main():
    API_KEY = "PM3TEJGFH2WHWPIG"  # Replace with your actual Alpha Vantage API key
    SYMBOL = "TSLA"
    START_DATE = "2024-09-01"
    END_DATE = "2025-04-17"
    # [2025-01-01, 2025-08-04]
    # [2024-01-01, 2024-12-31]

    stock_data = get_daily_stock_prices(API_KEY, SYMBOL, START_DATE, END_DATE)
    stock_data = stock_data.sort_index(ascending=True)
    stock_data.reset_index(drop=False, inplace=True)
    stock_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    print(stock_data)
    append_line("c:/ml/AlphaVantageStockPricesPrint.txt",stock_data.to_string(index=True))

if __name__ == "__main__":
    main()