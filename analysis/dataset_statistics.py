import os
import json
import string
import random

import pandas as pd
import numpy as np
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

from sklearn.decomposition import LatentDirichletAllocation as LDA

random.seed(42)
np.random.seed(42)

EXT_TYPES = ["pdf", "png"]
REMOVE_TOKENS = set(stopwords.words('english') + ["n't", "ve"]).union(set(string.punctuation))
sns.set_style('whitegrid')

def get_continuous_chunks(text):
        chunked = ne_chunk(pos_tag(word_tokenize(text)))
        continuous_chunk = []
        current_chunk = []
        for i in chunked:
                if type(i) == Tree:
                        current_chunk.append(" ".join([token for token, pos in i.leaves()]))
                elif current_chunk:
                        named_entity = " ".join(current_chunk)
                        if named_entity not in continuous_chunk:
                                continuous_chunk.append(named_entity)
                                current_chunk = []
                else:
                        continue
        return continuous_chunk


def plot_n_most_common_words(text: pd.Series, data_name: str, n: int = 10, specific_year: str = None):
    entities = []
    for sentence in text:
        entities.extend(get_continuous_chunks(sentence))
    stemmer = PorterStemmer()
    filtered_words = [stemmer.stem(w) for w in entities if not w in REMOVE_TOKENS]
    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words='english')
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(filtered_words)
    # Visualise the 10 most common words
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]

    get_lda(count_data, count_vectorizer)

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:n]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    for ext_type in EXT_TYPES:
        plt.figure(2, figsize=(15, 15/1.6180))
        plt.subplot(title='{} most common words'.format(n))
        sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
        sns.barplot(x_pos, counts, palette='husl')
        plt.xticks(x_pos, words, rotation=90) 
        plt.xlabel('words')
        plt.ylabel('counts')
        plt.savefig(os.path.join(os.path.realpath('..'), "plots", data_name, 
                    "{}top_wordcounts.{}".format("" if specific_year is None else str(specific_year) + "_", ext_type)))
        plt.close()


def run_wordcloud(text: list, data_name: str):
    word_tokens = []
    try:
        for index, sentence in enumerate(text):
            if pd.isnull(sentence):
                print("index is null", index)
                continue
            word_tokens.extend(word_tokenize(sentence))
    except Exception as e:
        import pdb; pdb.set_trace()
        print(e, sentence)
    filtered_words = [w for w in word_tokens if not w in REMOVE_TOKENS] 
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", width=800, height=800, max_words=150, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(" ".join(filtered_words))
    # Visualize the word cloud
    for ext_type in EXT_TYPES:
        wordcloud.to_file(os.path.join(os.path.realpath('..'), "plots", data_name, 'wordCloud.{}'.format(ext_type)))
        plt.close()


def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

def get_lda(count_data, count_vectorizer): 
    # Tweak the two parameters below
    number_topics = 5
    number_words = 10
    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)
    print("Topics found via LDA:")
    print_topics(lda, count_vectorizer, number_words)

def get_statistics(df: pd.DataFrame, data_name: str):
    distrib = df.copy(deep=True)
    distrib = distrib[distrib["score"] != 0]
    for ext_type in EXT_TYPES:
        sns.distplot(distrib["score"])
        plt.savefig(os.path.join(os.path.realpath('..'), "plots", data_name, 'total_score_distribution.{}'.format(ext_type)))
        plt.close()

    punch = np.array([len(sentence) for sentence in df["punchline"].values])
    body = np.array([len(sentence) for sentence in df["body"].values])
    joke = np.array([len(sentence) for sentence in df["joke"].values])

    average_punch = np.mean(punch)
    average_body = np.mean(body)
    average_joke = np.mean(joke) - 1.0 # Joke Token, `AND` to join body and punchline together

    std_punch = np.nanstd(punch)
    std_body = np.nanstd(body)
    std_joke = np.nanstd(joke)

    ave_tokens_punch = np.nanmean(np.array([len(nltk.word_tokenize(sentence)) for sentence in df["punchline"].dropna().values]))
    ave_tokens_body = np.nanmean(np.array([len(nltk.word_tokenize(sentence)) for sentence in df["body"].dropna().values]))
    ave_tokens_joke = np.nanmean(np.array([len(nltk.word_tokenize(sentence)) for sentence in df["joke"].dropna().values]))

    tokens = []
    [tokens.extend(nltk.word_tokenize(joke)) for joke in df["joke"].dropna().values]
    total_tokens = len(set(tokens))

    stat_df = pd.DataFrame([{"ave_punchline_len": average_punch, "ave_body_len": average_body, "ave_joke_len": average_joke, "std_punch": std_punch,
                                "std_body": std_body, "std_joke": std_joke, "total_tokens": total_tokens}])
    stat_df.to_csv(os.path.join(os.path.realpath('..'), "plots", data_name, 'statistics.txt'))


def plot_sentiment(df: pd.DataFrame, data_name: str):
    for ext_type in EXT_TYPES:
        ax = sns.lineplot(x="date", y="prop", hue="sentiment", data=df, ci=False)
        # Find the x,y coordinates for each point
        x_coords = []
        y_coords = []
        for point_pair in ax.collections:
            for x, y in point_pair.get_offsets():
                x_coords.append(x)
                y_coords.append(y)
        # create the custom error bars
        colors = ['steelblue']*2 + ['coral']*2
        ax.errorbar(x_coords, y_coords, yerr=df["std"],
            ecolor=colors, fmt=' ', zorder=-1)
        ax.savefig(os.path.join(os.path.realpath('..'), "data", data_name, "sentiment_plot.{}".format(ext_type)))



def gather_data(df: pd.DataFrame, data_name: str):
    if not os.path.isdir(os.path.join(os.path.realpath('..'), "plots", data_name)):
        os.mkdir(os.path.join(os.path.realpath('..'), "plots", data_name))
    run_wordcloud(df["joke"].tolist(), data_name)
    plot_n_most_common_words(df["joke"].tolist(), data_name)
    get_statistics(df, data_name)


def percentiles_upvotes(df: pd.DataFrame, data_name: str) -> pd.DataFrame:
    list_of_percentiles = []
    for percentile in [0, 10, 25, 50, 75, 90, 100]:
        cur_per = np.percentile(df["score"], percentile)
        list_of_percentiles.append({"percentile": percentile, "value": cur_per})
    percent_df = pd.DataFrame(list_of_percentiles)
    percent_df.to_csv(os.path.join(os.path.realpath('..'), "plots", data_name, "percentiles.csv"))
    return percent_df


if __name__ == "__main__":
    # NOTE: log-distribution plots are found in the preprocess.py script
    df = pd.read_csv(os.path.join(os.path.realpath('..'), "data", "preprocessed.csv"), index_col=None, encoding="UTF-8", keep_default_na=False)
    df["date"] = pd.to_numeric(df["date"])
    df["score"] = pd.to_numeric(df["score"])
    df = df[df["date"].isna() == False]
    assert df.shape == df.dropna().shape, "was nans that are unaccounted for"
    percentiles_upvotes(df, "all")
    gather_data(df, "all")










