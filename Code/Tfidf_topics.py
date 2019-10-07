# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 08:15:30 2019

@author: thaddea.chua
"""
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

raw_df = pd.read_csv(r'..\Data\reddit_df_topics.csv')


def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def get_keywords(bag_of_words, num_keywords):
    tf_idf_vector = tfidf_transformer.transform(cv.transform([bag_of_words]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(feature_names, sorted_items,num_keywords)
    return keywords


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples,key = lambda x: (x[1],x[0]), reverse = True)

def extract_topn_from_vector(feature_names,sorted_items,topn = 10):
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score,3))
        feature_vals.append(feature_names[idx])
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return pd.DataFrame([score_vals, feature_vals]).transpose()


thresholds = np.arange(0,1,0.1).tolist()

cv = CountVectorizer()
comments_list = [' '.join(ast.literal_eval(word)) for word in raw_df['tokenized_list']]
word_count_vector = cv.fit_transform(comments_list)
feature_names = cv.get_feature_names()


tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)


for threshold in thresholds:
    filtered_df = raw_df[raw_df['topic_pct'] > threshold]
    print(threshold)
    writer = pd.ExcelWriter('tfdif_multiple_'+str(threshold)+'.xlsx', engine = 'xlsxwriter')
    for i in range(0,9):
        subset_df = filtered_df[filtered_df['topic'] == i]
        comments_list_subset = [' '.join(ast.literal_eval(word)) for word in subset_df['tokenized_list']]
        bag_of_words = ' '.join([' '.join(ast.literal_eval(word)) for word in subset_df['tokenized_list']])
        get_keywords(bag_of_words, 100000).to_excel(writer, sheet_name = "Topic_" + str(i))
        print(i)
    writer.save()
    
