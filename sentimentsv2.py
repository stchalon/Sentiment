import requests
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import time
import datetime

########################################################################
# --- Read AlphaVantageNewsBatchPrintFull.json file ---------------------------
# 1.  The data frame has sentiment_score from Alpha Vantage
# candidate for replacement - read JSON file
import json

# Open and load the JSON file
with open('c:/ml/AlphaVantageNewsBatchPrintFull.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)
df['index'] = range(len(df))
df['sentiment_score'] = df['overall_sentiment_score']

df['date'] = pd.to_datetime(df['time_published']).dt.date


# Display the DataFrame
print(df.head())
# df.dtypes: title, url, time_published, authors, summary, banner_image, source, category_within_source, source_domain, topics, overall_sentiment_score, overall_sentiment_label, ticker_sentiment

# --- Filter on source ------------------------
#  { 'Motley Fool', 'Zacks Commentary', 'South China Morning Post', 'Decrypt.co', 'CNBC', 'Benzinga', ... }
# df = df[df['source'].isin(['A', 'D', 'E'])]
# df = df[df['source'] == 'Motley Fool']

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
#
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
# 4. Group by date and compute mean sentiment score
#
ddf = merged_df.groupby('date')[['sentiment_score','bg_sentiment_score','va_sentiment_score']].mean().reset_index()
ddf.columns = ['date', 'av_sentiment_score','bg_sentiment_score','va_sentiment_score']


##########################################################################
# 5. Load stock prices from Alpha Vantage 
#
# Load the data
file_path = "c:/ml/AlphaVantageStockPricesPrint.txt"
stock_df = pd.read_csv(file_path, delim_whitespace=True)

# Convert 'date' column to datetime
stock_df['date'] = pd.to_datetime(stock_df['date'])


import seaborn as sns
import matplotlib.pyplot as plt

# --- Plot close & volume
# Create a figure and a set of subplots with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot 'close' on the first y-axis (left)
ax1.plot(stock_df['date'], stock_df['close'], color='blue', label='Close Price')
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis (right)
ax2 = ax1.twinx()
ax2.bar(stock_df['date'], stock_df['volume'], color='red', alpha=0.3, label='Volume')
ax2.set_ylabel('Volume', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Improve layout
fig.tight_layout()
plt.title("Stock Close Price and Volume Over Time")
plt.show()
fig.savefig('c:/ml/figCloseVolume.pdf')

##########################################################################
# 6. Merge in ddf Sentiments analysis and stock prices

# ---------- Merge ddf with stock_df ----------
merged_ddf = pd.merge(ddf, stock_df, on='date', how='left')
merged_ddf = merged_ddf.fillna(0)

# --- Filter null close values (week-end) ----------------------
merged_ddf = merged_ddf[merged_ddf['close'] >0]


###############################################################
# 7. Correlation and heatmap
#

# Adding change-rate
merged_ddf['change-rate']=(merged_ddf['close']-merged_ddf['open'])/merged_ddf['close']

# Adding daily_return, sentiment_return
#
merged_ddf['daily-return'] = merged_ddf['close'].pct_change()
merged_ddf['av_sentiment_score-return'] = merged_ddf['av_sentiment_score'].pct_change()

# Check
columns_to_display = ['date', 'av_sentiment_score', 'close', 'av_sentiment_score', 'daily-return']
print(merged_ddf[columns_to_display].head())

# Save the merged DataFrame to a text file
#
with open("c:/ml/merged_ddf.txt", "w") as f:
    for index, row in merged_ddf.iterrows():
        f.write(f"{row['date']}, {row['av_sentiment_score']}, {row['close']}, {row['av_sentiment_score-return']}, {row['daily-return']}\n") 

###############################################################
# --- Stationary test - ADF Augmented Dickey-Fuller ----------
#
from statsmodels.tsa.stattools import adfuller

def ADFtest(series, title=''):
    """
    Perform Augmented Dickey-Fuller test and print the results.
    """
    result = adfuller(series)
    print(f"**ADF test results for {title}:")
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', '#Observations Used']
    for label, value in zip(labels, result):
        print(f'{label}: {value}')
    if result[1] <= 0.05:
        print("Reject H0: Series is stationary\n")
    else:
        print("Fail to reject H0: Series is non-stationary\n")

s1 = merged_ddf['av_sentiment_score'].dropna()
s2 = merged_ddf['close'].dropna()
s3 = merged_ddf['av_sentiment_score-return'].dropna()
s4 = merged_ddf['daily-return'].dropna()

ADFtest(s1, 'av_sentiment_score')
ADFtest(s2, 'close')
ADFtest(s3, 'av_sentiment_score-return')    # is stationary  !!
ADFtest(s4, 'daily-return')                 # is stationary  !! 


# Select the relevant columns
cols = ['av_sentiment_score', 'bg_sentiment_score','va_sentiment_score','close','volume','change-rate','av_sentiment_score-return','daily-return']  # Add more if needed

# --- Plot daily-close & sentiment_score-return
# Create a figure and a set of subplots with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot 'close' on the first y-axis (left)
ax1.plot(merged_ddf['date'], merged_ddf['daily-return'], color='blue', label='Daily return')
ax1.set_xlabel('Date')
ax1.set_ylabel('Daily return', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis (right)
ax2 = ax1.twinx()
ax2.plot(merged_ddf['date'], merged_ddf['av_sentiment_score-return'], color='red', label='Sentiment Score return', alpha=0.3)
ax2.set_ylabel('Sentiment_score return', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Improve layout
fig.tight_layout()
plt.title("Stock Daily-return and Sentiment_score return Time")
plt.show()
fig.savefig('c:/ml/figCloseSentiment.pdf')

# --- Plot daily-return & Sentiment score
# Create a figure and a set of subplots with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot 'close' on the first y-axis (left)
ax1.plot(merged_ddf['date'], merged_ddf['daily-return'], color='blue', label='Daily return', alpha=0.3)
ax1.set_xlabel('Date')
ax1.set_ylabel('daily-return', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis (right)
ax2 = ax1.twinx()
ax2.plot(merged_ddf['date'], merged_ddf['av_sentiment_score'], color='red', label='Sentiment Score', alpha=0.3)
ax2.set_ylabel('Sentiment', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Improve layout
fig.tight_layout()
plt.title("Daily-return and Sentiment over Time")
plt.show()
fig.savefig('c:/ml/figDailyreturnSentiment.pdf')


# --- Compute the full correlation matrix ------------
corr_matrix = merged_ddf[cols].corr()

# Plot correlation matrix
plt.figure(figsize=(12, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix base')
plt.tight_layout()
plt.show()
fig.savefig('c:/ml/figCorrelationBase.pdf')

###############################################################
import pandas as pd

# Define the number of lags you want to consider
n_lags = 3

# Create a dictionary to store the correlation matrices for each lag
corr_matrices = {}

for n in range(1, n_lags + 1):
    # Create lagged columns for 'close' and 'volume'
    lagged_close = merged_ddf['close'].shift(n)
    lagged_volume = merged_ddf['volume'].shift(n)
    lagged_dailyreturn = merged_ddf['daily-return'].shift(n)

    # Create a new DataFrame with the original and lagged columns
    lagged_df = merged_ddf[cols].copy()
    lagged_df[f'close_lag_{n}'] = lagged_close
    lagged_df[f'volume_lag_{n}'] = lagged_volume
    lagged_df[f'dailyreturn_lag_{n}'] = lagged_dailyreturn

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
    fig.savefig(f'c:/ml/figCorrelationLag{n}.pdf')

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
fig.savefig('c:/ml/figSentimentsAvVaBg.pdf')

###########################################################################
# --- Scatter plot of price and buy/sell signals based on sentiment scores
#
merged_ddf['buy_signal'] = 0
merged_ddf['sell_signal'] = 0
merged_ddf['buy_signal'] = (merged_ddf['av_sentiment_score'] > 0.2) & (merged_ddf['close'] > 0)
merged_ddf['sell_signal'] = (merged_ddf['av_sentiment_score'] < -0.2) & (merged_ddf['close'] > 0)

# Create the base plot
fig, ax = plt.subplots(figsize=(12, 6))

close_signals = merged_ddf[merged_ddf['close'] > 0 ]
ax.plot(close_signals['date'], close_signals['close'], label='Stock Price', linewidth=1)

# Add buy signals (green triangles)
buy_signals = merged_ddf[merged_ddf['buy_signal'] == True ]
ax.scatter(buy_signals['date'], buy_signals['close'],
           marker='^', color='green', s=100, label='Buy Signal')

# Add sell signals (red triangles)
sell_signals = merged_ddf[merged_ddf['sell_signal'] == True]
ax.scatter(sell_signals['date'], sell_signals['close'],
           marker='v', color='red', s=100, label='Sell Signal')

plt.tight_layout()
plt.show()
fig.savefig('c:/ml/figCloseSellBuy.pdf')

####################################################################
# 8. Granger causality test
# 8.1 av_sentiment_score => close
# 8.2 av_sentiment_score => daily-return
# From ritvikmath (https://www.youtube.com/@ritvikmath)
# https://www.youtube.com/watch?v=4TkNZviNJC0   Example
# https://www.youtube.com/watch?v=b8hzDzGWyGM   Theory
#
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###################################################################################
# --- Granger causality test (av_sentiment_score => close) ---
t1 = merged_ddf['av_sentiment_score']
t2 = merged_ddf['close']                # Expected to be t2 source

d = merged_ddf['date']

# Create a figure and a set of subplots with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 8))
ax2 = ax1.twinx()

# plot and store handles
p1, = ax1.plot(d, t1, color='blue', label='t1')
p2, = ax2.plot(d, t2, color='red', label='t2')

# combine legend handles and labels from both axes
lines = [p1, p2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=10)

# add title
fig.suptitle('Granger Causality Test: av_sentiment_score => Close', fontsize=14)

plt.show()
fig.savefig('c:/ml/figAvSentimentClose.pdf')

# 1st columnn is t2, the serie you think is the cause of the 2nd argument t1
ts_df = pd.DataFrame(columns=['t2', 't1'], data=zip(t2,t1))

# Granger causality test, 1 lag, 2 lags and 3 lags
# checking p-value vs 0.0 (very strong evidence that t2 causes t1 with 3 lags
print("Granger causality test (av_sentiment_score => close)\n")
gc_res = grangercausalitytests(ts_df, maxlag=4, verbose=True)

###################################################################################
# --- Granger causality test (av_sentiment_score => Daily-return) ---
t1 = merged_ddf['av_sentiment_score']
t2 = merged_ddf['daily-return'].fillna(0)

d = merged_ddf['date']

# Create a figure and a set of subplots with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 8))
ax2 = ax1.twinx()

# plot and store handles
p1, = ax1.plot(d, t1, color='blue', label='sentiment_score')
p2, = ax2.plot(d, t2, color='red', label='daily-return')

# combine legend handles and labels from both axes
lines = [p1, p2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=10)

# add title
fig.suptitle('Granger Causality Test: av_sentiment_score => daily-return', fontsize=14)

plt.show()
fig.savefig('c:/ml/figAvSentimentDailyreturn.pdf')

# 1st columnn is t2, the serie you think is the cause of the 2nd argument t1
ts_df = pd.DataFrame(columns=['t2', 't1'], data=zip(t2,t1))

# Granger causality test, 1 lag, 2 lags and 3 lags
# checking p-value vs 0.0 (very strong evidence that t2 causes t1 with 3 lags
print("Granger causality test (av_sentiment_score => Daily-return)\n")
gc_res = grangercausalitytests(ts_df, maxlag=4, verbose=True)


###################################################################################
# --- Granger causality test (av_sentiment_score-return => Daily-return) ---
# Drop rows with any NaNs
merged_ddf = merged_ddf.dropna()
t1 = merged_ddf['av_sentiment_score-return']
t2 = merged_ddf['daily-return'].fillna(0)

d = merged_ddf['date']

# Create a figure and a set of subplots with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 8))
ax2 = ax1.twinx()

# plot and store handles
p1, = ax1.plot(d, t1, color='blue', label='sentiment_score-return')
p2, = ax2.plot(d, t2, color='red', label='daily-return')

# combine legend handles and labels from both axes
lines = [p1, p2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=10)

# add title
fig.suptitle('Granger Causality Test: av_sentiment_score-return => daily-return', fontsize=14)

plt.show()
fig.savefig('c:/ml/figAvSentimentreturnDailyreturn.pdf')

# 1st columnn is t2, the serie you think is the cause of the 2nd argument t1
ts_df = pd.DataFrame(columns=['t2', 't1'], data=zip(t2,t1))

# Granger causality test, 1 lag, 2 lags and 3 lags
# checking p-value vs 0.0 (very strong evidence that t2 causes t1 with 3 lags
print("Granger causality test (av_sentiment_score-return => Daily-return)\n")
gc_res = grangercausalitytests(ts_df, maxlag=4, verbose=True)


############################################################################
# Granger causality test matrix
#
from statsmodels.tsa.stattools import grangercausalitytests

df = merged_ddf
df.set_index('date', inplace=True)

# Drop rows with NaN (like first daily-return)
df.dropna(inplace=True)

# Granger causality test
def granger_matrix(data, maxlag=2):
    variables = data.columns
    pvals = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)

    for col_x in variables:
        for col_y in variables:
            if col_x != col_y:
                try:
                    result = grangercausalitytests(data[[col_y, col_x]], maxlag=maxlag, verbose=False)
                    pval = min([result[i + 1][0]['ssr_ftest'][1] for i in range(maxlag)])
                    pvals.loc[col_y, col_x] = round(pval, 4)
                except:
                    pvals.loc[col_y, col_x] = np.nan
            else:
                pvals.loc[col_y, col_x] = np.nan
    return pvals

# Apply test
selected = ['av_sentiment_score', 'av_sentiment_score-return', 'close', 'daily-return']
pval_matrix = granger_matrix(df[selected])

# Plot
plt.figure(figsize=(7, 5))
sns.heatmap(pval_matrix, annot=True, cmap="coolwarm_r", fmt=".4f", cbar_kws={"label": "p-value"})
plt.title("Granger Causality Test Heatmap")
plt.xlabel("Causing variable")
plt.ylabel("Affected variable")
plt.tight_layout()
plt.show()
fig.savefig('c:/ml/figGrangerCausalityHeatmap.pdf')

print("end\n")
