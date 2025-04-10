import requests
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import time
import datetime
import re           # for regex

##########################################################################
# --- Read AlphaVantageNewsBatchPrint.txt file ---------------------------
# 1.  The data frame has sentiment_score from Alpha Vantage

# Define regex once, outside the function for efficiency
line_pattern = re.compile(
    r'^(?P<index>\d+):\s+(?P<date>\d{8})\s+(?P<sentiment_score>[-+]?\d*\.\d+|\d+)\s+\[(?P<sentiment_label>[^\]]+)\]\s+\[(?P<source>[^\]]+)\]\s+(?P<title>.+)$'
)

def parse_line_to_dict(line):
    match = line_pattern.match(line.strip())
    if match:
        data = match.groupdict()
        # Optional: convert data types
        data['index'] = int(data['index'])
        data['sentiment_score'] = float(data['sentiment_score'])

        return data
    else:
        return None

records = []
with open("c:/ml/AlphaVantageNewsBatchPrint.txt", "r", encoding="utf-8-sig", errors="replace") as f:
    for line in f:
        parsed = parse_line_to_dict(line)
        if parsed:
            records.append(parsed)

df = pd.DataFrame(records)
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

##########################################################################
# 2.  Add sentiment scores from Bing Lexicon
# --- Load Bing lexicon and compute bing-based sentiment scores ----------
def load_bing_lexicon():
    pos_url = "https://gist.githubusercontent.com/mkulakowski2/4289437/raw/positive-words.txt"
    neg_url = "https://gist.githubusercontent.com/mkulakowski2/4289441/raw/negative-words.txt"

    def load_words(url, sentiment):
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download {sentiment} words")
        lines = response.text.splitlines()
        words = [line.strip() for line in lines if line and not line.startswith(";")]
        return pd.DataFrame({'word': words, 'sentiment': sentiment})

    pos_df = load_words(pos_url, 'positive')
    neg_df = load_words(neg_url, 'negative')
    return pd.concat([pos_df, neg_df], ignore_index=True)

bing_lexicon = load_bing_lexicon()

##########################################################################
# ---------- Tokenize News Titles ----------
tokenized_df = df.copy()
tokenized_df['word'] = tokenized_df['title'].str.lower().apply(word_tokenize)
tokenized_df = tokenized_df.explode('word')

# ---------- Join with Bing Lexicon ----------
sentiment_words_df = tokenized_df.merge(bing_lexicon, on='word', how='inner')

# ---------- Group By index and sentiment ----------
pivot_df = sentiment_words_df.groupby(['index', 'sentiment']).size().unstack(fill_value=0)
pivot_df.reset_index(inplace=True)

# ---------- Merge df with pivot_df ----------
merged_df = pd.merge(df, pivot_df, on='index', how='left')
merged_df = merged_df.fillna(0)

merged_df = merged_df.rename(columns={
    'negative': 'bg_neg',
    'positive': 'bg_pos'
})
merged_df['bg_sentiment_score']= merged_df['bg_pos'] - merged_df['bg_neg']

##########################################################################
# 3.  Add sentiment scores from VADER
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create an instance of the Vader sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Loop through the texts and get the sentiment scores for each one
for text in df['title']:
    # Use the sentiment analyzer to get the scores for each text
    scores = analyzer.polarity_scores(text)

compound_scores = []
neutral_scores = []
positive_scores = []
negative_scores = []

for text in merged_df['title']:
    scores = analyzer.polarity_scores(text)
    compound_scores.append(scores['compound'])
    neutral_scores.append(scores['neu'])
    positive_scores.append(scores['pos'])
    negative_scores.append(scores['neg'])

# Add the results as a new column to df
merged_df['va_sentiment_score'] = compound_scores
merged_df['va_neutral'] = neutral_scores
merged_df['va_positive'] = positive_scores    
merged_df['va_negative'] = negative_scores

##########################################################################
# --- Group by date and compute mean sentiment score ---------------------
#
ddf = merged_df.groupby('date')[['sentiment_score','bg_sentiment_score','va_sentiment_score']].mean().reset_index()
ddf.columns = ['date', 'av_sentiment_score','bg_sentiment_score','va_sentiment_score']

##########################################################################
# --- Load stock prices from Alpha Vantage -------------------------------
# Load the data
file_path = "c:/ml/AlphaVantageStockPricesPrint.txt"
stock_df = pd.read_csv(file_path, delim_whitespace=True)

# Convert 'date' column to datetime
stock_df['date'] = pd.to_datetime(stock_df['date'])

import seaborn as sns
import matplotlib.pyplot as plt

# Create a figure and a set of subplots with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot 'close' on the first y-axis (left)
ax1.plot(stock_df['date'], stock_df['close'], color='tab:blue', label='Close Price')
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis (right)
ax2 = ax1.twinx()
ax2.bar(stock_df['date'], stock_df['volume'], color='tab:gray', alpha=0.3, label='Volume')
ax2.set_ylabel('Volume', color='tab:gray')
ax2.tick_params(axis='y', labelcolor='tab:gray')

# Improve layout
fig.tight_layout()
plt.title("Stock Close Price and Volume Over Time")
plt.show()

# ---------- Merge ddf with stock_df ----------
merged_ddf = pd.merge(ddf, stock_df, on='date', how='left')
merged_ddf = merged_ddf.fillna(0)

###############################################################
# --- Correlation and heatmap ---------------------------------
#
# Select the relevant columns
cols = ['av_sentiment_score', 'bg_sentiment_score','va_sentiment_score','close','volume']  # Add more if needed

# Compute the full correlation matrix
corr_matrix = merged_ddf[cols].corr()

# Plot correlation matrix
plt.figure(figsize=(12, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix base')
plt.tight_layout()
plt.show()

###############################################################
import pandas as pd

# Assuming merged_ddf is your DataFrame and cols is the list of columns you want to include
##cols = ['av_score', 'bg_score', 'va_score', 'close', 'volume']

# Define the number of lags you want to consider
n_lags = 3  # For example, you can change this to any number of lags you need

# Create a dictionary to store the correlation matrices for each lag
corr_matrices = {}

for n in range(1, n_lags + 1):
    # Create lagged columns for 'close' and 'volume'
    lagged_close = merged_ddf['close'].shift(n)
    lagged_volume = merged_ddf['volume'].shift(n)

    # Create a new DataFrame with the original and lagged columns
    lagged_df = merged_ddf[cols].copy()
    lagged_df[f'close_lag_{n}'] = lagged_close
    lagged_df[f'volume_lag_{n}'] = lagged_volume

    # Compute the correlation matrix for the lagged DataFrame
    corr_matrix = lagged_df.corr()

    # Store the correlation matrix in the dictionary
    corr_matrices[f'lag_{n}'] = corr_matrix

    # Display the correlation matrix for the current lag
    print(f"Correlation Matrix for Lag {n}:\n", corr_matrix)

    # corr_matrices[f'lag_{n}']
    plt.figure(figsize=(12, 5))
    sns.heatmap(corr_matrices[f'lag_{n}'], annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)
    plt.title(f'Correlation Matrix for Lag {n}')
    plt.tight_layout()
    plt.show()

# Access a specific correlation matrix if needed
# For example, to get the correlation matrix for lag 1:
# lag_1_corr_matrix = corr_matrices['lag_1']
# ...
# lag_3_corr_matrix = corr_matrices['lag_3']

###################################################################
# Plotting the positive and negative word counts over time
#
grouped_df = merged_df.groupby('date')[['sentiment_score', 'va_sentiment_score','bg_sentiment_score']].sum().reset_index()
print(grouped_df)

# below : use merged_df (by index -> area of all items with same date) or use grouped_df.
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax1 = plt.subplots(figsize=(12, 5))

# First line: Alpha Vantage (green)
line1 = sns.lineplot(x='date', y='sentiment_score', data=grouped_df, marker='o', color='green', ax=ax1)
line1.set_label("Alpha Vantage sentiment score")
ax1.set_ylabel("Alpha Vantage sentiment score", color='green')
ax1.tick_params(axis='y', labelcolor='green')

# Second y-axis: VABER (red)
ax2 = ax1.twinx()
line2 = sns.lineplot(x='date', y='va_sentiment_score', data=grouped_df, marker='o', color='red', ax=ax2)
line2.set_label("VABER sentiment score")
ax2.set_ylabel("VABER sentiment score", color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.spines["right"].set_position(("axes", 1.0))  # Position ax2 on the right

# Third y-axis: Bing Lexicon (blue)
ax3 = ax1.twinx()
line3 = sns.lineplot(x='date', y='bg_sentiment_score', data=grouped_df, marker='o', color='blue', ax=ax3)
line3.set_label("Bing Lexicon sentiment score")
ax3.set_ylabel("Bing Lexicon sentiment score", color='blue')
ax3.tick_params(axis='y', labelcolor='blue')
ax3.spines["right"].set_position(("axes", 1.1))  # Move ax3 further out

# X-axis and title
ax1.set_xlabel('Date')
ax1.set_title('Sentiment Scores Over Time')
plt.xticks(rotation=45)

# Combine all lines for legend
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left')

plt.tight_layout()
plt.show()

