import pandas as pd
import json

# --- Load saved JSON lines ---------------------------------------------
# AlphaVantageNewsBatchPrintFull-<start>-<end>.json
#with open("c:/ml/AlphaVantageNewsBatchPrintFull-20240901-20241231.json", "r", encoding="utf-8") as f:
#    lines = f.readlines()
#    records = [json.loads(line) for line in lines]

# Convert to DataFrame
#full_df1 = pd.DataFrame(records)

# Open and load the JSON file
with open('c:/ml/AlphaVantageNewsBatchPrintFull-20240901-20241231.json', 'r', encoding="utf-8") as f:
    data = json.load(f)

# Convert to DataFrame
full_df1 = pd.DataFrame(data)


# Load saved JSON lines
# AlphaVantageNewsBatchPrintFull-<start>-<end>.json
#with open("c:/ml/AlphaVantageNewsBatchPrintFull-20250101-20250417.json", "r", encoding="utf-8") as f:
#    lines = f.readlines()
#    records = [json.loads(line) for line in lines]

# Convert to DataFrame
#full_df2 = pd.DataFrame(records)
#full_df2['date'] = pd.to_datetime(full_df2['time_published']).dt.date

# Open and load the JSON file
with open('c:/ml/AlphaVantageNewsBatchPrintFull-20250101-20250417.json', 'r', encoding="utf-8") as f:
    data = json.load(f)

# Convert to DataFrame
full_df2 = pd.DataFrame(data)


# Concatenate full_df1 and full_df2
#
# Concatenate the DataFrames
full_df = pd.concat([full_df1, full_df2], ignore_index=True)
# Convert a specific 'date' column to string
#full_df['date'] = full_df['date'].astype(str)

# Convert to a list of dictionaries
data = full_df.to_dict(orient='records')

# Save to xx.txt using json.dump
with open('c:/ml/AlphaVantageNewsBatchPrintFull.json', 'w') as f:
    json.dump(data, f)      # 'indent=2' makes it more readable

# Preview
print(full_df.head())

# df.dtypes
# title: object
# url: url format
# time_published: object -> date
# authors: list
# summary: object
# banner_image: object
# source: object
#  { 'Motley Fool', 'Zacks Commentary', 'South China Morning Post', 'Decrypt.co', 'CNBC', 'Benzinga', ... }
# category_within_source: object
# source_domain: url
# topics: object
# overall_sentiment_score: float64
# overall_sentiment_label: object
# ticker_sentiment: object
#
