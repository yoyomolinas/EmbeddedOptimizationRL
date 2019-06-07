#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:04:11 2018

@author: apple
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

"""
Returns t-sne embedding for given data, perplexity and iteration. 
Data consists of n rows of arbitrary-dimensional vectors. Is a numpy array or list.
For preferred perplexity and iteration values, refer to gridSearchTSNE function.
"""
def getScaledTsneEmbedding(data, perplexity, iteration):
    embedding = TSNE(n_components=2,
                     perplexity=perplexity,
                     n_iter=iteration).fit_transform(data)
    scaler = MinMaxScaler()
    scaler.fit(embedding)
    return scaler.transform(embedding)

"""
Returns a dictionary in the following format: {"(perp, n_iter)" : embeddings}
"""
def gridSearchTSNE(X):
    perplexities = [30, 50, 100]
    iterations =  [1000, 2000, 5000]

    i = 1
    rets = {}
    for perp in perplexities:
        for n_iter in iterations:
            print('perplexity:', perp, ', n_iter:', n_iter)
            embeddings = getScaledTsneEmbedding(X, perp, n_iter)
            # red_data = init.create_dataset(str(i), data = embedding)
            # red_data.attrs['title'], red_data.attrs['perplexity'], red_data.attrs['iteration'] = i, perp, n_iter
            # i += 1
            rets[str((perp, n_iter))] = embeddings.tolist()

    return rets