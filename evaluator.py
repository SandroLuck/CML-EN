import tensorflow as tf
from scipy.sparse import lil_matrix


class RecallEvaluator(object):
    def __init__(self, model, train_user_item_matrix, test_user_item_matrix, train_user_item_exp_neg_matrix, test_user_item_exp_neg_matrix):
        """
        Create a evaluator for recall@K evaluation
        :param model: the model we are going to evaluate
        :param train_user_item_matrix: the user-item pairs used in the training set. These pairs will be ignored
               in the recall calculation
        :param test_user_item_matrix: the held-out user-item pairs we make prediction against
        """
        self.model = model
        self.train_user_item_matrix = lil_matrix(train_user_item_matrix)
        self.test_user_item_matrix = lil_matrix(test_user_item_matrix)
        n_users = train_user_item_matrix.shape[0]

        self.user_to_test_set = {u: set(self.test_user_item_matrix.rows[u]) for u in range(n_users) if self.test_user_item_matrix.rows[u]}

        if self.train_user_item_matrix is not None:
            self.user_to_train_set = {u: set(self.train_user_item_matrix.rows[u])
                                      for u in range(n_users) if self.train_user_item_matrix.rows[u]}
            self.max_train_count = max(len(row) for row in self.train_user_item_matrix.rows)
        else:
            self.max_train_count = 0

        # Add explicitly negative votes
        self.train_user_item_exp_neg_matrix = lil_matrix(train_user_item_exp_neg_matrix)
        self.test_user_item_exp_neg_matrix = lil_matrix(test_user_item_exp_neg_matrix)
        n_users_exp_neg = train_user_item_exp_neg_matrix.shape[0]

        self.user_to_test_exp_neg_set = {u: set(self.test_user_item_exp_neg_matrix.rows[u]) for u in range(n_users_exp_neg) if self.test_user_item_exp_neg_matrix.rows[u]}

        if self.train_user_item_exp_neg_matrix is not None:
            self.user_to_train_exp_neg_set = {u: set(self.train_user_item_exp_neg_matrix.rows[u])
                                      for u in range(n_users_exp_neg) if self.train_user_item_exp_neg_matrix.rows[u]}
            self.max_train_exp_neg_count = max(len(row) for row in self.train_user_item_exp_neg_matrix.rows)
        else:
            self.max_train_exp_neg_count = 0
    def NITK(self, k, tops, train, test):
        i = 0
        ignore_count = 0
        neg_in_top_k=0
        while (i <= k + ignore_count):
            if tops[i] in train:
                ignore_count += 1
            elif tops[i] in test:
                neg_in_top_k += 1
            i += 1
        return max(neg_in_top_k/float(k),neg_in_top_k/float(len(test)))

    def eval(self, sess, users, k=50, nI1=5,nI2=50):
        """
        Compute the Top-K recall for a particular user given the predicted scores to items
        And calculate the NegativeItemsInTopK
        :param users: the users to eval the recall
        :param k: compute the recall for the top K items
        :return: recall@K
        """
        # compute the top (K +  Max Number Of Training Items for any user) items for each user

        _, user_tops = sess.run(tf.nn.top_k(self.model.item_scores, k + self.max_train_count),
                                {self.model.score_user_ids: users})
        recalls = []
        exp_negative_items_in_top_k=[]
        exp_negative_items_in_top_5 = []
        exp_negative_items_in_top_10 = []

        precision_at_len_test=[]
        stat=0
        #print("Amount users: ",len(users))
        for user_id, tops in zip(users, user_tops):
            test_set = self.user_to_test_set.get(user_id, set())
            if len(test_set)>=5:
                train_set = self.user_to_train_set.get(user_id, set())
                #print("TOPS",tops)
                # get negative set
                train_exp_neg_set = self.user_to_train_exp_neg_set.get(user_id, set())
                test_exp_neg_set = self.user_to_test_exp_neg_set.get(user_id, set())
                length_of_interactions=len(test_set)+len(test_exp_neg_set)+len(train_exp_neg_set)+len(train_set)

                counter_items = 0
                hits = 0

                neg_in_tops = 0
                neg_in_top_5 = 0
                neg_in_top_10 = 0

                pos_items_in_len_test = 0
                exp_neg_items_in_len_test = 0
                i=0
                counter_items = 0

                for i in tops:
                    if i in train_set:
                        continue
                    if i in test_set:
                        pos_items_in_len_test += 1
                    counter_items += 1
                    if counter_items == k:
                        break
                counter_items = 0

                for i in tops:
                    if i in train_exp_neg_set:
                        continue
                    if i in test_exp_neg_set:
                        exp_neg_items_in_len_test += 1
                    counter_items += 1
                    if counter_items == k:
                        break
                if pos_items_in_len_test+exp_neg_items_in_len_test==0:
                    precision_at_len_test.append([float(0),length_of_interactions])
                else:
                    precision_at_len_test.append([pos_items_in_len_test/float(pos_items_in_len_test+exp_neg_items_in_len_test),length_of_interactions])
                top_n_items = 0

                for i in tops:
                    # ignore item in the training set
                    if i in train_set:
                        continue
                    elif i in test_set:
                        hits += 1
                    top_n_items += 1
                    if top_n_items == k:
                        break
                recalls.append([hits / float(len(test_set)),length_of_interactions])
                if len(test_exp_neg_set)>0:
                    exp_negative_items_in_top_5.append([self.NITK(nI1,tops,train_exp_neg_set,test_exp_neg_set),length_of_interactions])
                    exp_negative_items_in_top_10.append([self.NITK(nI2,tops,train_exp_neg_set,test_exp_neg_set),length_of_interactions])
                # For testing print lines for user
        #print(len( recalls), len(precision_at_len_test) ,len(exp_negative_items_in_top_5), len(exp_negative_items_in_top_10))
        return recalls, precision_at_len_test ,exp_negative_items_in_top_5, exp_negative_items_in_top_10