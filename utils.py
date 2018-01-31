from collections import defaultdict

import numpy as np
from scipy.sparse import dok_matrix, lil_matrix
from tqdm import tqdm
import os
import time

path_to_data="/citeulike-t"
path_to_data_movie="/dataConverter"

def citeulike(tag_occurence_thres=10):
    user_dict = defaultdict(set)
    for u, item_list in enumerate(open(path_to_data+"/users.dat").readlines()):
        items = item_list.strip().split(" ")
        for item in items:
            user_dict[u].add(int(item))

    n_users = len(user_dict)
    n_items = max([item for items in user_dict.values() for item in items]) + 1

    #Enter all items 1 if user liked it
    user_item_matrix = dok_matrix((n_users, n_items), dtype=np.int32)
    for u, item_list in enumerate(open(path_to_data+"/users.dat").readlines()):
        items = item_list.strip().split(" ")
        for item in items:
            user_item_matrix[u, int(item)] = 1

    n_features = 0
    for l in open(path_to_data+"/tag-item.dat").readlines():
        items = l.strip().split(" ")
        if len(items) >= tag_occurence_thres:
            n_features += 1
    print("{} features over tag_occurence_thres ({})".format(n_features, tag_occurence_thres))

    features = dok_matrix((n_items, n_features), dtype=np.int32)
    feature_index = 0
    for l in open(path_to_data+"/tag-item.dat").readlines():
        items = l.strip().split(" ")
        if len(items) >= tag_occurence_thres:
            features[[int(i) for i in items], feature_index] = 1
            feature_index += 1

    return user_item_matrix, features

def movie100k(tag_occurence_thres=10):
    """
    smaples the data for the movie100k dataset
    :return:
    :param tag_occurence_thres: threshold for the tag_occurence
    """
    user_dict = defaultdict(set)
    for u, item_list in enumerate(open(path_to_data_movie+"/userMlAboveAvg.dat").readlines()):
        items = item_list.strip().split(" ")
        for item in items:
            user_dict[u].add(int(item))

    n_users = len(user_dict)
    #Since the way @Sandro processed movie100k data it might be that some movies exist which have no positive upvote which might lead to index out of range error to fix this change max size of movie
    # n_items = max([item for items in user_dict.values() for item in items]) + 1
    n_items=1683 #we know there are 1682 movies in db +1 for matrix TODO imporove this logic if necessary

    #Enter all items 1 if user liked it
    user_item_matrix = dok_matrix((n_users, n_items), dtype=np.int32)
    for u, item_list in enumerate(open(path_to_data_movie+"/userMlAboveAvg.dat").readlines()):
        items = item_list.strip().split(" ")
        for item in items:
            user_item_matrix[u, int(item)] = 1
    #######################################################
    # Sample Negatives @Sandro
    # create a user dict for explicitly negative item
    user_dict_exp_neg = defaultdict(set)
    for u, item_list in enumerate(open(path_to_data_movie+"/userMlBelowAvg.dat").readlines()):
        items = item_list.strip().split(" ")
        for item in items:
            user_dict_exp_neg[u].add(int(item))
    n_users_exp_neg = len(user_dict_exp_neg)

    #Enter all items 1 if user DID rate the items below his average, aka disliked the item
    user_item_exp_neg_matrix = dok_matrix((n_users_exp_neg, n_items), dtype=np.int32)
    for u, item_list in enumerate(open(path_to_data_movie+"/userMlBelowAvg.dat").readlines()):
        items = item_list.strip().split(" ")
        for item in items:
            user_item_exp_neg_matrix[u, int(item)] = 1


    #######################################################
    print 5 * "\n"
    print "Dim User likes:" + str(n_users)
    print "Dim User dislikes:" + str(n_users_exp_neg)
    print 5 * "\n"




    n_features = 0
    for l in open(path_to_data_movie+"/PlotTagItemCMl.dat").readlines():
        items = l.strip().split(" ")
        if len(items) >= tag_occurence_thres:
            n_features += 1
    print("{} features over tag_occurence_thres ({})".format(n_features, tag_occurence_thres))

    print "new features:",n_features
    features = dok_matrix((n_items, n_features), dtype=np.int32)
    feature_index = 0
    for l in open(path_to_data_movie+"/PlotTagItemCMl.dat").readlines():
        items = l.strip().split(" ")
        if len(items) >= tag_occurence_thres:
            features[[int(i) for i in items], feature_index] = 1
            feature_index += 1

    return user_item_matrix, features, user_item_exp_neg_matrix

def movie20m(tag_occurence_thres=30,ratio_to_take=0.1):
    """
    smaples the data for the movie20m dataset
    :return:
    :param tag_occurence_thres: threshold for the tag_occurence
    """
    start = time.time()

    print "Current Working directory:", os.getcwd()
    user_dict = defaultdict(set)
    lis=open(os.getcwd()+path_to_data_movie+"/uAboveDense.dat").readlines()
    for u, item_list in enumerate(lis[:int(len(lis)*ratio_to_take)]):
        items = item_list.strip().split(" ")
        for item in items[1:]:
            user_dict[u].add(int(item))
    print("Made matrix")

    end = time.time()
    print(end - start," for aboveAvg")
    start = time.time()


    n_users = int(len(user_dict)*ratio_to_take)
    #Since the way @Sandro processed movie100k data it might be that some movies exist which have no positive upvote which might lead to index out of range error to fix this change max size of movie
    # n_items = max([item for items in user_dict.values() for item in items]) + 1
    #131264=20m 1683=100k
    n_items=int(20683*ratio_to_take) #we know there are 131264 movies in db +1 for matrix TODO imporove this logic if necessary
    #Enter all items 1 if user liked it
    user_item_matrix = dok_matrix((n_users, n_items), dtype=np.int32)
    lis=open(os.getcwd()+path_to_data_movie+"/uAboveDense.dat").readlines()
    for u, item_list in enumerate(lis[:int(len(lis)*ratio_to_take)]):
        items = item_list.strip().split(" ")
        for item in items[1:]:
            if int(item)<n_items and u<n_users:
                user_item_matrix[u, int(item)] = 1
    print("Made matrix")
    end = time.time()
    print(end - start," for aboveAvg 2nd")
    start = time.time()
    #######################################################
    # Sample Negatives @Sandro
    # create a user dict for explicitly negative item
    user_dict_exp_neg = defaultdict(set)
    lis=open(os.getcwd()+path_to_data_movie+"/uBelowDense.dat").readlines()
    for u, item_list in enumerate(lis[:int(len(lis)*ratio_to_take)]):
        items = item_list.strip().split(" ")
        for item in items[1:]:
            user_dict_exp_neg[u].add(int(item))
    n_users_exp_neg = len(user_dict_exp_neg)
    print("Made matrix")
    end = time.time()
    print(end - start," for belowAvg")
    start = time.time()

    #Enter all items 1 if user DID rate the items below his average, aka disliked the item
    user_item_exp_neg_matrix = dok_matrix((n_users_exp_neg, n_items), dtype=np.int32)
    lis=open(os.getcwd()+path_to_data_movie+"/uBelowDense.dat").readlines()
    for u, item_list in enumerate(lis[:int(len(lis)*ratio_to_take)]):
        items = item_list.strip().split(" ")
        for item in items[1:]:
            if int(item)<n_items and u<n_users:
                user_item_exp_neg_matrix[u, int(item)] = 1

    end = time.time()
    print(end - start," for belowAvg 2nd")
    start = time.time()

    n_features = 0
    for l in open(os.getcwd()+path_to_data_movie+"/PlotTagDense.dat").readlines():
        items = l.strip().split(" ")
        if len([i for i in items[1:] if int(i)<n_items]) >= tag_occurence_thres:
            n_features += 1
    print("{} features over tag_occurence_thres ({})".format(n_features, tag_occurence_thres))

    print "new features:",n_features
    features = dok_matrix((n_items, n_features), dtype=np.int32)
    feature_index = 0
    for l in open(os.getcwd()+path_to_data_movie+"/PlotTagDense.dat").readlines()[:]:

            items = l.strip().split(" ")
            if len([i for i in items[1:] if int(i)<n_items]) >= tag_occurence_thres:
                features[[int(i) for i in items if int(i)<n_items], feature_index] = 1
                feature_index += 1
    end = time.time()
    print(end - start," for plotTags")
    start = time.time()

    return user_item_matrix, features, user_item_exp_neg_matrix

def books6m(tag_occurence_thres=30,ratio_to_take=1.0):
    """
    smaples the data for the Goodbooks datase
    :return:
    :param tag_occurence_thres: threshold for the tag_occurence
    """
    start = time.time()

    print "Current Working directory:", os.getcwd()
    user_dict = defaultdict(set)
    lis=open(os.getcwd()+path_to_data_movie+"/uAboveAvgBooks.dat").readlines()
    for u, item_list in enumerate(lis[:int(len(lis)*ratio_to_take)]):
        items = item_list.strip().split(" ")
        for item in items[1:]:
            user_dict[u].add(int(item))
    print("Made matrix")

    end = time.time()
    print(end - start," for aboveAvg")
    start = time.time()


    n_users = int(len(user_dict)*ratio_to_take)
    #Since the way @Sandro processed movie100k data it might be that some movies exist which have no positive upvote which might lead to index out of range error to fix this change max size of movie
    # n_items = max([item for items in user_dict.values() for item in items]) + 1
    #131264=20m 1683=100k
    n_items=int(10000*ratio_to_take) #we know there are 131264 movies in db +1 for matrix TODO imporove this logic if necessary
    #Enter all items 1 if user liked it
    user_item_matrix = dok_matrix((n_users, n_items), dtype=np.int32)
    lis=open(os.getcwd()+path_to_data_movie+"/uAboveAvgBooks.dat").readlines()
    for u, item_list in enumerate(lis[:int(len(lis)*ratio_to_take)]):
        items = item_list.strip().split(" ")
        for item in items[1:]:
            if int(item)<n_items and u<n_users:
                user_item_matrix[u, int(item)] = 1
    print("Made matrix")
    end = time.time()
    print(end - start," for aboveAvg 2nd")
    start = time.time()
    #######################################################
    # Sample Negatives @Sandro
    # create a user dict for explicitly negative item
    user_dict_exp_neg = defaultdict(set)
    lis=open(os.getcwd()+path_to_data_movie+"/uBelowAvgBooks.dat").readlines()
    for u, item_list in enumerate(lis[:int(len(lis)*ratio_to_take)]):
        items = item_list.strip().split(" ")
        for item in items[1:]:
            user_dict_exp_neg[u].add(int(item))
    n_users_exp_neg = len(user_dict_exp_neg)
    print("Made matrix")
    end = time.time()
    print(end - start," for belowAvg")
    start = time.time()

    #Enter all items 1 if user DID rate the items below his average, aka disliked the item
    user_item_exp_neg_matrix = dok_matrix((n_users_exp_neg, n_items), dtype=np.int32)
    lis=open(os.getcwd()+path_to_data_movie+"/uBelowAvgBooks.dat").readlines()
    for u, item_list in enumerate(lis[:int(len(lis)*ratio_to_take)]):
        items = item_list.strip().split(" ")
        for item in items[1:]:
            if int(item)<n_items and u<n_users:
                user_item_exp_neg_matrix[u, int(item)] = 1

    end = time.time()
    print(end - start," for belowAvg 2nd")
    start = time.time()

    n_features = 0
    for l in open(os.getcwd()+path_to_data_movie+"/book_tag_cml.dat").readlines():
        items = l.strip().split(" ")
        if len([i for i in items[1:] if int(i)<n_items]) >= tag_occurence_thres:
            n_features += 1
    print("{} features over tag_occurence_thres ({})".format(n_features, tag_occurence_thres))

    print "new features:",n_features
    features = dok_matrix((n_items, n_features), dtype=np.int32)
    feature_index = 0
    for l in open(os.getcwd()+path_to_data_movie+"/book_tag_cml.dat").readlines()[:]:

            items = l.strip().split(" ")
            if len([i for i in items[1:] if int(i)<n_items]) >= tag_occurence_thres:
                features[[int(i) for i in items if int(i)<n_items], feature_index] = 1
                feature_index += 1
    end = time.time()
    print(end - start," for plotTags")
    start = time.time()

    return user_item_matrix, features, user_item_exp_neg_matrix

def ml1m(tag_occurence_thres=2,ratio_to_take=1.0):
    """
    smaples the data for the movielens 1 million dataset
    :return:
    :param tag_occurence_thres: threshold for the tag_occurence
    """
    start = time.time()

    print "Current Working directory:", os.getcwd()
    user_dict = defaultdict(set)
    lis=open(os.getcwd()+path_to_data_movie+"/mlm1uabove.dat").readlines()
    for u, item_list in enumerate(lis[:int(len(lis)*ratio_to_take)]):
        items = item_list.strip().split(" ")
        for item in items[1:]:
            user_dict[u].add(int(item))
    print("Made matrix")

    end = time.time()
    print(end - start," for aboveAvg")
    start = time.time()


    n_users = int(len(user_dict)*ratio_to_take)
    #Since the way @Sandro processed movie100k data it might be that some movies exist which have no positive upvote which might lead to index out of range error to fix this change max size of movie
    # n_items = max([item for items in user_dict.values() for item in items]) + 1
    #131264=20m 1683=100k
    n_items=int(3950*ratio_to_take) #we know there are 131264 movies in db +1 for matrix TODO imporove this logic if necessary
    #Enter all items 1 if user liked it
    user_item_matrix = dok_matrix((n_users, n_items), dtype=np.int32)
    lis=open(os.getcwd()+path_to_data_movie+"/mlm1uabove.dat").readlines()
    for u, item_list in enumerate(lis[:int(len(lis)*ratio_to_take)]):
        items = item_list.strip().split(" ")
        for item in items[1:]:
            if int(item)<n_items and u<n_users:
                user_item_matrix[u, int(item)] = 1
    print("Made matrix")
    end = time.time()
    print(end - start," for aboveAvg 2nd")
    start = time.time()
    #######################################################
    # Sample Negatives @Sandro
    # create a user dict for explicitly negative item
    user_dict_exp_neg = defaultdict(set)
    lis=open(os.getcwd()+path_to_data_movie+"/mlm1ubelow.dat").readlines()
    for u, item_list in enumerate(lis[:int(len(lis)*ratio_to_take)]):
        items = item_list.strip().split(" ")
        for item in items[1:]:
            user_dict_exp_neg[u].add(int(item))
    n_users_exp_neg = len(user_dict_exp_neg)
    print("Made matrix")
    end = time.time()
    print(end - start," for belowAvg")
    start = time.time()

    #Enter all items 1 if user DID rate the items below his average, aka disliked the item
    user_item_exp_neg_matrix = dok_matrix((n_users_exp_neg, n_items), dtype=np.int32)
    lis=open(os.getcwd()+path_to_data_movie+"/mlm1ubelow.dat").readlines()
    for u, item_list in enumerate(lis[:int(len(lis)*ratio_to_take)]):
        items = item_list.strip().split(" ")
        for item in items[1:]:
            if int(item)<n_items and u<n_users:
                user_item_exp_neg_matrix[u, int(item)] = 1

    end = time.time()
    print(end - start," for belowAvg 2nd")
    start = time.time()

    n_features = 0
    for l in open(os.getcwd()+path_to_data_movie+"/mlm1tag.dat").readlines():
        items = l.strip().split(" ")
        if len([i for i in items[1:] if int(i)<n_items]) >= tag_occurence_thres:
            n_features += 1
    print("{} features over tag_occurence_thres ({})".format(n_features, tag_occurence_thres))

    print "new features:",n_features
    features = dok_matrix((n_items, n_features), dtype=np.int32)
    feature_index = 0
    for l in open(os.getcwd()+path_to_data_movie+"/mlm1tag.dat").readlines()[:]:

            items = l.strip().split(" ")
            if len([i for i in items[1:] if int(i)<n_items]) >= tag_occurence_thres:
                features[[int(i) for i in items if int(i)<n_items], feature_index] = 1
                feature_index += 1
    end = time.time()
    print(end - start," for plotTags")
    start = time.time()

    return user_item_matrix, features, user_item_exp_neg_matrix



def split_data(user_item_matrix, split_ratio=(1, 1, 1), seed=6000):
    # set the seed to have deterministic results
    """
    Splits the Data for the tests
    :param user_item_matrix:
    :param split_ratio:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    train = dok_matrix(user_item_matrix.shape)
    validation = dok_matrix(user_item_matrix.shape)
    test = dok_matrix(user_item_matrix.shape)
    # convert it to lil format for fast row access
    user_item_matrix = lil_matrix(user_item_matrix)
    for user in tqdm(range(user_item_matrix.shape[0]), desc="Split data into train/valid/test"):
        items = list(user_item_matrix.rows[user])
        if len(items) >= 1:

            np.random.shuffle(items)

            train_count = int(len(items) * split_ratio[0] / sum(split_ratio))
            valid_count = int(len(items) * split_ratio[1] / sum(split_ratio))

            for i in items[0: train_count]:
                train[user, i] = 1
            for i in items[train_count: train_count + valid_count]:
                validation[user, i] = 1
            for i in items[train_count + valid_count:]:
                test[user, i] = 1
    print("{}/{}/{} train/valid/test samples".format(
        len(train.nonzero()[0]),
        len(validation.nonzero()[0]),
        len(test.nonzero()[0])))
    return train, validation, test
