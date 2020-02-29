import os
from datetime import datetime
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation as LDA

from dataset_statistics import plot_n_most_common_words, EXT_TYPES
  
random.seed(42)
np.random.seed(42)
 
def sentiment_scores(sid_obj: SentimentIntensityAnalyzer, sentence: str) -> int: 
    sentiment_dict = sid_obj.polarity_scores(sentence) 
    if sentiment_dict['compound'] >= 0.05 : 
        # print("positive", sentence)
        return 1
    elif sentiment_dict['compound'] <= - 0.05 : 
        # print("negative", sentence)
        return -1
    else:
        # print("neutral", sentence) 
        return 0


def get_sentiment(df, data_name):
    sid_obj = SentimentIntensityAnalyzer() 
    sentiments = np.array([sentiment_scores(sid_obj, sentence) for sentence in df["joke"].dropna().values])
    results = dict(zip(*np.unique(sentiments, return_counts=True)))
    pos_prop = results[-1] / len(sentiments)
    neutral_prop = results[0] / len(sentiments)
    neg_prop = results[1] / len(sentiments)

    # std for a proportion is sqrt(p*q / n)
    pos_std = np.sqrt((pos_prop * (1 - pos_prop)) / len(sentiments))
    neutral_std = np.sqrt((neutral_prop * (1 - neutral_prop)) / len(sentiments))
    neg_std = np.sqrt((neg_prop * (1 - neg_prop)) / len(sentiments))

    results = {"positive_prop": pos_prop, "positive_std": pos_std, "neutral_prop": neutral_prop, "neutral_std": neutral_std, "negative_prop": neg_prop,
                "negative_std": neg_std}

    if "date" not in df.columns:
        return results
    else:
        list_of_results = []
        for prop_type in ["positive", "neutral", "negative"]:
            name = prop_type + "_prop"
            name_std = prop_type + "_std"
            list_of_results.append({"prop": results[name], "sentiment": prop_type, "std": results[name_std], "date": df.date.iloc[0]})
        return pd.DataFrame(list_of_results)


def get_scores(df, data_name):
    return df[["date", "score"]]

def get_jokes(df, data_name):
    return df[["date", "joke"]]

def split_by_year(df: pd.DataFrame, func, data_name="all") -> pd.DataFrame:
    df["date"] = df["date"].map(lambda x: pd.to_datetime(datetime.utcfromtimestamp(x)).year)
    date_df = df.groupby("date").apply(lambda x: func(x, data_name))
    return date_df

def plot_lines_by_year(date_df: pd.DataFrame, data_name: str, name: str = "sentiment"):
    for ext_type in EXT_TYPES:
        ax = sns.lineplot(x="date", y="prop", hue="sentiment", data=date_df, palette=["C2", "C7", "C3"]) 
        ax.errorbar(date_df["date"], date_df["prop"], yerr=date_df["std"], fmt='o', ecolor='black')
        plt.xlabel('Year')
        plt.ylabel('Proportion of Sentiment')
        plt.savefig(os.path.join(os.path.realpath('..'), "plots", data_name, "{}.{}".format(name, ext_type)))
        plt.close()

def plot_distribution_by_year(date_df: pd.DataFrame, data_name: str, name: str = "score"):
    date_df = date_df[date_df["score"] != 0] # zeros skew the distribution
    for ext_type in EXT_TYPES:
        ax = sns.violinplot(x="date", y="score", data=date_df)
        plt.xlabel('Year')
        plt.ylabel('Distribution of Jokes')
        plt.savefig(os.path.join(os.path.realpath('..'), "plots", data_name, "{}.{}".format(name, ext_type)))
        plt.close()

def plot_bar_by_year(df: pd.DataFrame, data_name: str, name: str):
    if name == "mean":
        ave = df.groupby('date').mean().reset_index(drop=True)
        ave.columns = ["average"]
        max = df.groupby('date').max().reset_index()
        max.columns = ["date", "score"]
        data = pd.concat([max, ave], axis=1)
    elif name == "count":
        data = df.groupby('date').count().reset_index()
    elif name == "max":
        data = df.groupby('date').max().reset_index()
    else:
        raise NotImplementedError()

    for ext_type in EXT_TYPES:
        ax = sns.barplot(x="date", y="score", data=data)
        for index, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    # will use average for numbering in max case
                    '{}'.format(int(data["score"].iloc[index]) if name in ["count", "max"] else int(data["average"].iloc[index])),
                    ha="center") 
        plt.xlabel('Year')
        plt.ylabel(name)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.realpath('..'), "plots", data_name, "{}.{}".format(name, ext_type)))
        plt.close()


if __name__ == "__main__" : 
    df = pd.read_csv(os.path.join(os.path.realpath('..'), "data", "preprocessed.csv"), index_col=None, encoding="UTF-8", keep_default_na=False)
    df["date"] = pd.to_numeric(df["date"])
    df["score"] = pd.to_numeric(df["score"])
    df = df[df["date"].isna() == False]

    plot_bar_by_year(split_by_year(df.copy(deep=True), get_scores), "all", "max")
    plot_bar_by_year(split_by_year(df.copy(deep=True), get_scores), "all", "count")
    plot_lines_by_year(split_by_year(df.copy(deep=True), get_sentiment, "all"), "all", "sentiment")
    plot_distribution_by_year(split_by_year(df.copy(deep=True), get_scores, "all"), "all", "score_distributions")
    # plot top entities per year
    jokes_by_year = split_by_year(df.copy(deep=True), get_jokes, "all")
    for year in jokes_by_year["date"].unique():
        plot_n_most_common_words(jokes_by_year[jokes_by_year["date"] == year]["joke"].tolist(), data_name="yearbyyear", specific_year=year)
