import requests
import pandas as pd

import numpy as np

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import time
import datetime

import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import grangercausalitytests

import warnings

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


##############################################################################
# --- Load stock prices from Alpha Vantage ------------------------------------
#
# Load the data
file_path = "c:/ml/AlphaVantageStockPricesPrint.txt"
stock_df = pd.read_csv(file_path, delim_whitespace=True)

# Convert 'date' column to datetime
stock_df['date'] = pd.to_datetime(stock_df['date'])

###############################################################################
# --- Read AlphaVantageNewsBatchPrintFull.json file ---------------------------
#
import json

# Open and load the JSON file
with open('c:/ml/AlphaVantageNewsBatchPrintFull.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
news_df = pd.DataFrame(data)

news_df['index'] = range(len(news_df))
news_df['sentiment_score'] = news_df['overall_sentiment_score']

news_df['date'] = pd.to_datetime(news_df['time_published']).dt.date
news_df['date'] = pd.to_datetime(news_df['date'], format="%Y%m%d", errors="coerce")

# Display the DataFrame
print(news_df.head())

print(news_df['source'].value_counts())
print(news_df['category_within_source'].value_counts())

# df.dtypes: title, url, time_published, authors, summary, banner_image, source, category_within_source, source_domain, topics, overall_sentiment_score, overall_sentiment_label, ticker_sentiment

# --- Optional Filter on source ------------------------
#  { 'Motley Fool', 'Zacks Commentary', 'South China Morning Post', 'Decrypt.co', 'CNBC', 'Benzinga', ... }
# df = df[df['source'].isin(['A', 'D', 'E'])]
# df = df[df['source'] == 'Motley Fool']

# df["category_within_source"] == 'Trading'

filtersCategories = ['n/a','News', 'Markets', 'Trading', 'General', 'Top News', 'Companies', 'Business',
    'RSS', 'Economy', 'Investing', 'Top Stories', 'Finance', 'Money', 'Mergers', 'Earnings']

filterSources = ['Benzinga', 'Motley Fool', 'Zacks Commentary', 'CNBC', 'South China Morning Post',
           'Decrypt.co', 'GlobeNewswire', 'Cointelegraph', 'Financial Times', 'Business Insider']

#variables = filter
#pvals = pd.DataFrame(np.zeros((len(variables), len(variables))),
#                     columns=variables, index=variables)

# Function to apply
def gc_function(Category, Source):

    df = news_df.copy()
    df = df[(df['category_within_source'] == Category) & (df['source'] == Source)]
    if df.empty:
        return np.nan, np.nan
    else:
        ##########################################################################
        # Compute mean sentiment score per date and Group News by date
        #
        ddf = df.groupby('date')[['sentiment_score']].mean().reset_index()
        ddf.columns = ['date', 'av_sentiment_score']

        ##########################################################################
        # Merge in ddf Sentiments analysis and stock prices

        # ---------- Merge ddf with stock_df ----------
        merged_ddf = pd.merge(ddf, stock_df, on='date', how='left')
        merged_ddf = merged_ddf.fillna(0)

        # --- Filter null close values (week-end) ----------------------
        merged_ddf = merged_ddf[merged_ddf['close'] >0]

        # Adding change-rate (close-open)/close
        merged_ddf['change-rate']=(merged_ddf['close']-merged_ddf['open'])/merged_ddf['close']

        # Adding daily_return, sentiment_return
        #
        merged_ddf['daily-return'] = merged_ddf['close'].pct_change()
        merged_ddf['av_sentiment_score-return'] = merged_ddf['av_sentiment_score'].pct_change()

        ############################################################################
        # Granger causality test matrix
        #
        gcdf = merged_ddf
        gcdf.set_index('date', inplace=True)

        # Drop rows with NaN (like first daily-return)
        gcdf.dropna(inplace=True)

        # Apply test
        selected = ['av_sentiment_score-return', 'daily-return']
        print(f'### source: {filterSource}, category: {filterCategory}')
        pval_matrix = granger_matrix(gcdf[selected])

        # 1/ cause 'av_sentiment_score-return', effect 'daily-return'
        # 2/ cause 'daily-return', effect 'av_sentiment_score-return'
        print(f'source {filterSource}, category: {filterCategory} cause sentimentscore-return {pval_matrix["daily-return"]["av_sentiment_score-return"]}')
        print(f'source {filterSource}, category: {filterCategory} cause daily-return {pval_matrix["av_sentiment_score-return"]["daily-return"]}')
        return pval_matrix["daily-return"]["av_sentiment_score-return"], pval_matrix["av_sentiment_score-return"]["daily-return"]
    

# Build the matrix
McauseSentimentsEffectDaily = []
McauseDailyEffectSentiments = []

for filterSource in filterSources:

    rowCauseSentimentsEffectDaily = []
    rowCauseDailyEffectSentiments = []

    for filterCategory in filtersCategories:
        # 1/ cause 'av_sentiment_score-return', effect 'daily-return'
        # 2/ cause 'daily-return', effect 'av_sentiment_score-return'
        gc_causeSentimentsEffectDaily, gc_causeDailyEffectSentiments = gc_function(filterCategory, filterSource)
        rowCauseSentimentsEffectDaily.append(gc_causeSentimentsEffectDaily)
        rowCauseDailyEffectSentiments.append(gc_causeDailyEffectSentiments)

    McauseSentimentsEffectDaily.append(rowCauseSentimentsEffectDaily)
    McauseDailyEffectSentiments.append(rowCauseDailyEffectSentiments)

# Convert Matrixes to a DataFrames for better handling
p_matrix_causeSentimentsEffectDaily = pd.DataFrame(McauseSentimentsEffectDaily)
p_matrix_causeDailyEffectSentiments = pd.DataFrame(McauseDailyEffectSentiments)

# Plot the heatmap 
plt.figure(figsize=(10, 5))
sns.heatmap(p_matrix_causeSentimentsEffectDaily, annot=True, cmap="coolwarm_r", fmt=".2f", cbar_kws={"label": "value"})

plt.title("Granger Causality Test - Sentiment_score-return => Daily-return")
plt.xlabel("Categories")
plt.ylabel("Sources")

# Set custom x-axis and y-axis labels
plt.xticks(ticks=np.arange(len(filtersCategories)) + 0.5, labels=filtersCategories, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(filterSources)) + 0.5, labels=filterSources, rotation=0)

plt.tight_layout()
plt.savefig('c:/ml/figGrangerCausalityCauseSentimentEffectDaily.pdf')
plt.show()


# Plot the heatmap 
plt.figure(figsize=(10, 5))
sns.heatmap(p_matrix_causeDailyEffectSentiments, annot=True, cmap="coolwarm_r", fmt=".2f", cbar_kws={"label": "value"})

plt.title("Granger Causality Test - Daily-return => Sentiment_score-return")
plt.xlabel("Categories")
plt.ylabel("Sources")

# Set custom x-axis and y-axis labels
plt.xticks(ticks=np.arange(len(filtersCategories)) + 0.5, labels=filtersCategories, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(filterSources)) + 0.5, labels=filterSources, rotation=0)

plt.tight_layout()
plt.savefig('c:/ml/figGrangerCausalityCauseDailyEffectSentiment.pdf')
plt.show()



print("end\n")