import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

# get prototype from all embedding from database
def get_prototype(db):
    return db.reshape(-1,3,512).mean(axis=1)

# get conditional probability based on evaluation result
def cond_prob(dist1, dist2, distance, same=True, step=100):
    lower = distance - distance%(2/step)
    upper = lower + 2/step
    same_count = dist1[(dist1 >= lower) & (dist1 < upper)].shape[0]
    diff_count = dist2[(dist2 >= lower) & (dist2 < upper)].shape[0]
    if same_count==0 and diff_count==0:
        return None
    if same:
        return same_count / (same_count + diff_count)
    else:
        return diff_count / (same_count + diff_count)

# comparing against the base, could be a prototype (k * 512) or all embedding from database (k * 3 * 512)
def compare_embed(base, embed, metrics='euclidean'):
    if len(embed.shape) == 1:
        embed = embed.reshape(1,-1)
    if metrics == 'euclidean':
        return euclidean_distances(base, embed)
    if metrics == 'cosine':
        return cosine_distances(base, embed)
    if metrics == 'cond_prob':
        distances = euclidean_distances(base, embed)
        return np.array([cond_prob(dist1, dist2, i, same=False, step=100) for i in distances])

# check comparison result against prototype with threshold
def prototype_compare_result(arr, threshold=1):
    if arr.min() > threshold:
        return None
    else:
        return arr.argmin()

# check comparison result against embedding of each class with threshold
def each_compare_result(arr, threshold=1, img_threshold=1):
    if arr.min() > threshold:
        return None
    else:
        arr = arr.reshape(-1,3)
        valid_count = (arr < threshold).sum(axis=1)
        max_count = valid_count.max()
        if max_count < img_threshold:
            return None
        else:
            temp = (valid_count==max_count).astype(float)
            temp[temp==0] = np.Inf
            valid_arr = arr * temp.reshape(-1,1)
            return valid_arr.mean(axis=1).argmin()

# db is the all embeddings from the database (default 3 for each person)
# return lable if successfully recognized (valid), otherwise None (invalid)
def recognize(db, embed):
    mode = 'prototype' # prototype or each
    metrics = 'euclidean' # euclidean, cosine, cond_prob
    threshold = 0.7
    img_threshold = 2 # minimum number of images rquired that satisfy the threshold (only used with mode 'each')

    # get evaluation dist result
    if metrics == 'cond_prob':
        with open('eval.npy', 'rb') as f:
            dist1 = np.load(f)
            dist2 = np.load(f)
        dist1[dist1>dist2.max()] = 0 # remove outliers in same comparison (too large distance calculated)

    if mode == 'prototype':
        prototype = get_prototype(db)
        return prototype_compare_result(compare_embed(prototype, embed, metrics='euclidean'), threshold=threshold)
    elif mode == 'each':
        return each_compare_result(compare_embed(db, embed, metrics='euclidean'), threshold=threshold, img_threshold=img_threshold)
