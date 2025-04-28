import requests
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import time
import datetime

import seaborn as sns
import matplotlib.pyplot as plt

########################################################################
# --- Read AlphaVantageNewsBatchPrintFull.json file ---------------------------
# 1.  The data frame has sentiment_score from Alpha Vantage
#
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

print(df['source'].value_counts())
print(df['category_within_source'].value_counts())

pivot_table = pd.crosstab(df['source'], df['category_within_source'])
print(pivot_table)

# df.dtypes: title, url, time_published, authors, summary, banner_image, source, category_within_source, source_domain, topics, overall_sentiment_score, overall_sentiment_label, ticker_sentiment

df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

# --- Optional Filter on source, category_within_source
#

filtersCategories = ['n/a','News', 'Markets', 'Trading', 'General', 'Top News', 'Companies', 'Business',
    'RSS', 'Economy', 'Investing', 'Top Stories', 'Finance', 'Money', 'Mergers', 'Earnings']

filterSources = ['Benzinga', 'Motley Fool', 'Zacks Commentary', 'CNBC', 'South China Morning Post',
           'Decrypt.co', 'GlobeNewswire', 'Cointelegraph', 'Financial Times', 'Business Insider']


##df = df[ (df['source'] == 'Financial Times') & (df['category_within_source'] == 'Markets') ]
df = df[ (df['source'] == 'Benzinga') & (df['category_within_source'] == 'News') ]



##########################################################################
# 4. Group by date and compute mean sentiment score
#
ddf = df.groupby('date')[['sentiment_score']].mean().reset_index()
ddf.columns = ['date', 'av_sentiment_score']

##########################################################################
# 5. Load stock prices from Alpha Vantage 
#
# Load the data
file_path = "c:/ml/AlphaVantageStockPricesPrint.txt"
stock_df = pd.read_csv(file_path, delim_whitespace=True)

# Convert 'date' column to datetime
stock_df['date'] = pd.to_datetime(stock_df['date'])

##########################################################################
# 6. Merge in ddf Sentiments analysis and stock prices

# ---------- Merge ddf with stock_df ----------
merged_ddf = pd.merge(ddf, stock_df, on='date', how='left')
merged_ddf = merged_ddf.fillna(0)

# --- Filter null close values (week-end) ----------------------
merged_ddf = merged_ddf[merged_ddf['close'] >0]

###############################################################

# Adding change-rate
merged_ddf['change-rate']=(merged_ddf['close']-merged_ddf['open'])/merged_ddf['close']

# Adding daily_return, sentiment_return
#
merged_ddf['daily-return'] = merged_ddf['close'].pct_change()
merged_ddf['av_sentiment_score-return'] = merged_ddf['av_sentiment_score'].pct_change()

# Check
columns_to_display = ['date', 'av_sentiment_score', 'close', 'av_sentiment_score', 'daily-return']
print(merged_ddf[columns_to_display].head())


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

#s1 = merged_ddf['av_sentiment_score'].dropna()
#s2 = merged_ddf['close'].dropna()
s3 = merged_ddf['av_sentiment_score-return'].dropna()
s4 = merged_ddf['daily-return'].dropna()

#ADFtest(s1, 'av_sentiment_score')          # is non-stationary !!
#ADFtest(s2, 'close')                       # is non-stationary !!
ADFtest(s3, 'av_sentiment_score-return')    # is stationary  !!
ADFtest(s4, 'daily-return')                 # is stationary  !! 


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
# --- Granger causality test (av_sentiment_score-return => Daily-return) ---
#
# Drop rows with any NaNs
merged_ddf = merged_ddf.dropna()
t1 = merged_ddf['av_sentiment_score-return']
t2 = merged_ddf['daily-return'].fillna(0)

d = merged_ddf['date']

# 1st columnn is t2, the serie you think is the cause of the 2nd argument t1
ts_df = pd.DataFrame(columns=['t2', 't1'], data=zip(t2,t1))

# Granger causality test, 1 lag, 2 lags and 3 lags
# checking p-value vs 0.0 (very strong evidence that t2 causes t1 with 3 lags
print("Granger causality test (av_sentiment_score-return => Daily-return)\n")
gc_res = grangercausalitytests(ts_df, maxlag=4, verbose=True)

###################################################################################
# --- Granger causality test (Daily-return => av_sentiment_score-return) ---
#
# Drop rows with any NaNs
merged_ddf = merged_ddf.dropna()
t1 = merged_ddf['daily-return'].fillna(0)
t2 = merged_ddf['av_sentiment_score-return']

d = merged_ddf['date']

# 1st columnn is t2, the serie you think is the cause of the 2nd argument t1
ts_df = pd.DataFrame(columns=['t2', 't1'], data=zip(t2,t1))

# Granger causality test, 1 lag, 2 lags and 3 lags
# checking p-value vs 0.0 (very strong evidence that t2 causes t1 with 3 lags
print("Granger causality test (Daily-return => av_sentiment_score-return\n")
gc_res = grangercausalitytests(ts_df, maxlag=4, verbose=True)


############################################################################
# Granger causality test matrix
#
from statsmodels.tsa.stattools import grangercausalitytests

df = merged_ddf
df.set_index('date', inplace=True)

# Drop rows with NaN (like first daily-return)
df.dropna(inplace=True)

import warnings

# Granger causality test
def granger_matrix(data, maxlag=4): 
    variables = data.columns
    pvals = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)

    for col_x in variables:
        for col_y in variables:
            if col_x != col_y:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=FutureWarning)
                        result = grangercausalitytests(data[[col_y, col_x]], maxlag=maxlag, verbose=False)
                    pval = min([result[i + 1][0]['ssr_ftest'][1] for i in range(maxlag)])
                    pvals.loc[col_y, col_x] = round(pval, 4)
                except:
                    pvals.loc[col_y, col_x] = np.nan
            else:
                pvals.loc[col_y, col_x] = np.nan
    return pvals

# Apply test
selected = ['av_sentiment_score-return', 'daily-return']
pval_matrix = granger_matrix(df[selected])

# Plot Granger Causality Test Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(pval_matrix, annot=True, cmap="coolwarm_r", fmt=".4f", cbar_kws={"label": "p-value"})
plt.title("Granger Causality Test Heatmap")
plt.xlabel("Causing variable")
plt.ylabel("Affected variable")
plt.tight_layout()
plt.savefig('c:/ml/figGrangerCausalityHeatmap.pdf')
plt.show()

print("end\n")
