import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import functools
import numpy
import tensorflow as tf
import toolz
from tqdm import tqdm
from evaluator import RecallEvaluator
from sampler import WarpSampler
from utils import citeulike, split_data, movie100k, movie20m, books6m, ml1m
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime
from shutil import copyfile
import time


def lrelu(x, alpha=0.05):
    """
    Implemented to use leaky relu
    :param x:
    :param alpha:
    :return:
    """
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class CML(object):
    def __init__(self,
                 n_users,
                 n_items,
                 embed_dim=20,
                 features=None,
                 margin=1.5,
                 master_learning_rate=0.1,
                 clip_norm=1.0,
                 hidden_layer_dim=128,
                 dropout_rate=0.2,
                 feature_l2_reg=0.1,
                 feature_projection_scaling_factor=0.5,
                 use_rank_weight=True,
                 use_cov_loss=True,
                 cov_loss_weight=0.1
                 ):
        """

        :param n_users: number of users i.e. |U|
        :param n_items: number of items i.e. |V|
        :param embed_dim: embedding size i.e. K (default 20)
        :param features: (optional) the feature vectors of items, shape: (|V|, N_Features).
               Set it to None will disable feature loss(default: None)
        :param margin: hinge loss threshold i.e. z
        :param master_learning_rate: master learning rate for AdaGrad
        :param clip_norm: clip norm threshold (default 1.0)
        :param hidden_layer_dim: the size of feature projector's hidden layer (default: 128)
        :param dropout_rate: the dropout rate between the hidden layer to final feature projection layer
        :param feature_l2_reg: feature loss weight
        :param feature_projection_scaling_factor: scale the feature projection before compute l2 loss. Ideally,
               the scaled feature projection should be mostly within the clip_norm
        :param use_rank_weight: whether to use rank weight
        :param use_cov_loss: use covariance loss to discourage redundancy in the user/item embedding
        """

        # Initiate default vars
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        self.clip_norm = clip_norm
        self.margin = margin
        if features is not None:
            self.features = tf.constant(features, dtype=tf.float32)
        else:
            self.features = None

        self.master_learning_rate = master_learning_rate
        self.hidden_layer_dim = hidden_layer_dim
        self.dropout_rate = dropout_rate
        self.feature_l2_reg = feature_l2_reg
        self.feature_projection_scaling_factor = feature_projection_scaling_factor
        self.use_rank_weight = use_rank_weight
        self.use_cov_loss = use_cov_loss
        self.cov_loss_weight = cov_loss_weight

        #init tf placeholders
        self.user_positive_items_pairs = tf.placeholder(tf.int32, [None, 2])
        self.negative_samples = tf.placeholder(tf.int32, [None, None])
        self.score_user_ids = tf.placeholder(tf.int32, [None])
        #init exp neg placeholder
        self.user_exp_neg_items_pairs = tf.placeholder(tf.int32, [None, 2])
        # init pos neg pairs
        self.pos_neg_pairs = tf.placeholder(tf.int32, [None, 2])


        #define tensorboard
        self.merged_summary_op = None

        #call functions
        self.user_embeddings
        self.item_embeddings

        self.embedding_loss
        self.feature_loss
        self.loss
        self.optimize


    @define_scope
    def user_embeddings(self):
        return tf.Variable(tf.random_normal([self.n_users, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

    @define_scope
    def item_embeddings(self):
        return tf.Variable(tf.random_normal([self.n_items, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

    @define_scope
    def mlp_layer_1(self):
        return tf.layers.dense(inputs=self.features,
                               units=self.hidden_layer_dim,
                               activation=tf.nn.relu, name="mlp_layer_1")

    @define_scope
    def mlp_layer_2(self):
        dropout = tf.layers.dropout(inputs=self.mlp_layer_1, rate=self.dropout_rate)
        return tf.layers.dense(inputs=dropout, units=self.embed_dim, name="mlp_layer_2")

    @define_scope
    def feature_projection(self):
        """
        :return: the projection of the feature vectors to the user-item embedding
        """

        # feature loss
        if self.features is not None:
            # fully-connected layer
            output = self.mlp_layer_2 * self.feature_projection_scaling_factor

            # projection to the embedding
            return tf.clip_by_norm(output, self.clip_norm, axes=[1], name="feature_projection")

    @define_scope
    def feature_loss(self):
        """
        :return: the l2 loss of the distance between items' their embedding and their feature projection
        """
        loss = tf.constant(0, dtype=tf.float32)
        if self.feature_projection is not None:
            # the distance between feature projection and the item's actual location in the embedding
            feature_distance = tf.reduce_sum(tf.squared_difference(
                self.item_embeddings,
                self.feature_projection), 1)

            # apply regularization weight
            loss += tf.reduce_sum(feature_distance, name="feature_loss") * self.feature_l2_reg

        return loss

    @define_scope
    def covariance_loss(self):

        X = tf.concat((self.item_embeddings, self.user_embeddings), 0)
        n_rows = tf.cast(tf.shape(X)[0], tf.float32)
        X = X - (tf.reduce_mean(X, axis=0))
        cov = tf.matmul(X, X, transpose_a=True) / n_rows

        return tf.reduce_sum(tf.matrix_set_diag(cov, tf.zeros(self.embed_dim, tf.float32))) * self.cov_loss_weight

    @define_scope
    def exp_neg_loss(self):
        """
        Defines the loss between a positve and a negative item
        :return: the loss that should be added to the loss functions
        """
        users = tf.nn.embedding_lookup(self.user_embeddings,
                                       self.user_exp_neg_items_pairs[:, 0],
                                       name="users")


        exp_neg_items = tf.nn.embedding_lookup(self.item_embeddings, self.user_exp_neg_items_pairs[:, 1],
                                           name="exp_neg_items")
        exp_neg_distances = tf.reduce_sum(tf.squared_difference(users, exp_neg_items), 1, name="exp_neg_distances")

        distance_tuner=float(4.0)
        exp_neg_mean_distance=tf.reduce_mean(exp_neg_distances)
        exp_neg_min_distance=tf.reduce_min(exp_neg_distances)


        loss = tf.reduce_sum(tf.maximum(distance_tuner/exp_neg_distances, float(1)), name="loss_exp_neg")#tf.maximum(4/exp_neg_distances, 2)

        with tf.name_scope('exp_neg_loss'):
            tf.summary.scalar("exp_neg_mean_distance",exp_neg_mean_distance)
            tf.summary.scalar("exp_neg_minimal_dist",exp_neg_min_distance)
            tf.summary.scalar("n_items_divided_by_exp_neg_minimal_dist", float(self.n_items)/exp_neg_min_distance)
        return loss


    
    @define_scope
    def exp_pos_neg_pair_loss(self):
        """
        This function defines the loss,
        between a positve and a negative item
        :return: loss multiplied with positive-negative pair hyperparameter
        """
        pos_items = tf.nn.embedding_lookup(self.item_embeddings, self.pos_neg_pairs[:,0],name="pos_items")
        # negative item embedding (N, K, W)
        neg_items = tf.nn.embedding_lookup(self.item_embeddings, self.pos_neg_pairs[:,1],name="neg_items")
        pos_neg_distance = tf.reduce_sum(tf.squared_difference(pos_items, neg_items,name="pos_exp_neg_distance"))
        pos_neg_distance_max = tf.reduce_max(pos_neg_distance, name="pos_exp_neg_distance_max")
        pos_neg_distance_min = tf.reduce_min(pos_neg_distance, name="pos_exp_neg_distance_max")
        pos_neg_distance_mean = tf.reduce_mean(pos_neg_distance)
        distance_tuner=float(3.0)
        loss = tf.reduce_sum(tf.maximum(distance_tuner/pos_neg_distance, float(1)), name="loss_exp_neg")#tf.maximum(4/exp_neg_distances, 2)


        #sum_pos_neg = tf.reduce_sum(pos_neg_distance, name="pos_exp_neg_distance_loss")
        #loss = -sum_pos_neg
        with tf.name_scope('pos_exp_neg_distance'):
           tf.summary.scalar("pos_exp_neg_distance_loss", loss)
           tf.summary.scalar("pos_exp_neg_distance_mean", pos_neg_distance_mean)
           tf.summary.scalar("pos_exp_neg_distance_max", pos_neg_distance_max)
           tf.summary.scalar("pos_exp_neg_distance_min", pos_neg_distance_min)
        #The 0.005 in the following line, is the negative pair loss tuning parameter
        return loss*0.005


    @define_scope
    def embedding_loss(self):
        """
        :return: the distance metric loss
        """
        # Let
        # N = batch size,
        # K = embedding size,
        # W = number of negative samples per a user-positive-item pair

        # user embedding (N, K)
        users = tf.nn.embedding_lookup(self.user_embeddings,
                                       self.user_positive_items_pairs[:, 0],
                                       name="users")

        # positive item embedding (N, K)
        pos_items = tf.nn.embedding_lookup(self.item_embeddings, self.user_positive_items_pairs[:, 1],
                                           name="pos_items")
        # positive item to user distance (N)
        pos_distances = tf.reduce_sum(tf.squared_difference(users, pos_items), 1, name="pos_distances")

        # negative item embedding (N, K, W)
        neg_items = tf.transpose(tf.nn.embedding_lookup(self.item_embeddings, self.negative_samples),
                                 (0, 2, 1), name="neg_items")
        # distance to negative items (N x W)
        distance_to_neg_items = tf.reduce_sum(tf.squared_difference(tf.expand_dims(users, -1), neg_items), 1,
                                              name="distance_to_neg_items")








        # best negative item (among W negative samples) their distance to the user embedding (N)
        closest_negative_item_distances = tf.reduce_min(distance_to_neg_items, 1, name="closest_negative_distances")


        # compute hinge loss (N)
        loss_per_pair = tf.maximum(pos_distances - closest_negative_item_distances + self.margin, 0,
                                   name="pair_loss")

        # sandro compute loss pushing appart exp negative

        if self.use_rank_weight:
            # indicator matrix for impostors (N x W)
            impostors = (tf.expand_dims(pos_distances, -1) - distance_to_neg_items + self.margin) > 0
            # approximate the rank of positive item by (number of impostor / W per user-positive pair)
            rank = tf.reduce_mean(tf.cast(impostors, dtype=tf.float32), 1, name="rank_weight") * self.n_items
            # apply rank weight
            loss_per_pair *= tf.log(rank + 1)

        # the embedding loss
        loss = tf.reduce_sum(loss_per_pair, name="loss")
        return loss

    @define_scope
    def loss(self):
        """
        :return: the total loss = embedding loss + feature loss +exp_neg_loss
        """
        # the 1.25 in the following line is the, negative loss tuning parameter
        loss = self.embedding_loss + self.feature_loss + (self.exp_neg_loss + self.exp_pos_neg_pair_loss)*0.75
        if self.use_cov_loss:
            loss += self.covariance_loss
        with tf.name_scope('Loss'):
            tf.summary.scalar("loss:",loss)
            tf.summary.scalar("embedding_loss_influence_in_percentage",self.embedding_loss/loss)
            tf.summary.scalar("feature_loss_influence_in_percentage",self.feature_loss/loss)
            tf.summary.scalar("exp_neg_push_away_loss_influence_in_percentage",self.exp_neg_loss/loss)
            tf.summary.scalar("exp_pos_neg_pair_push_away_loss_influence_in_percentage",self.exp_pos_neg_pair_loss/loss)
            tf.summary.scalar("covariance_loss_influence_in_percentage",self.covariance_loss/loss)
        return loss

    @define_scope
    def clip_by_norm_op(self):
        return [tf.assign(self.user_embeddings, tf.clip_by_norm(self.user_embeddings, self.clip_norm, axes=[1])),
                tf.assign(self.item_embeddings, tf.clip_by_norm(self.item_embeddings, self.clip_norm, axes=[1]))]

    @define_scope
    def optimize(self):
        # have two separate learning rates. The first one for user/item embedding is un-normalized.
        # The second one for feature projector NN is normalized by the number of items.
        gds = []
        gds.append(tf.train
                   .AdamOptimizer(self.master_learning_rate)
                   .minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings]))
        if self.feature_projection is not None:
            gds.append(tf.train
                       .AdamOptimizer(self.master_learning_rate)
                       .minimize(self.feature_loss / self.n_items))

        with tf.control_dependencies(gds):
            return gds + [self.clip_by_norm_op]

    @define_scope
    def item_scores(self):
        # (N_USER_IDS, 1, K)
        user = tf.expand_dims(tf.nn.embedding_lookup(self.user_embeddings, self.score_user_ids), 1)
        # (1, N_ITEM, K)
        item = tf.expand_dims(self.item_embeddings, 0)
        # score = minus distance (N_USER, N_ITEM)
        return -tf.reduce_sum(tf.squared_difference(user, item), 2, name="scores")
def log_eval_stats_binned(valid_recalls,valid_precision_at_len_test,exp_neg_items_in_top_k,exp_neg_items_in_top_5,exp_neg_items_in_top_10):
    """
    Generate the recall for multiple classes
    :param valid_recalls:
    :param valid_precision_at_len_test:
    :param exp_neg_items_in_top_k:
    :param exp_neg_items_in_top_5:
    :param exp_neg_items_in_top_10:
    :return:
    """
    log_binned(valid_recalls,text='recall')
    log_binned(valid_precision_at_len_test,text='prec')
    #log_binned(exp_neg_items_in_top_k,text='NITK')
    log_binned(exp_neg_items_in_top_5,text='NIT5')
    log_binned(exp_neg_items_in_top_10,text='NIT10')
    print('\nGlobal avg:\n',50*'-')
    print("\nRecall on (sampled) validation set: {}".format(numpy.mean([i for i,j in valid_recalls])))

    print("\nExp Neg in Top 5 (sampled) validation set: {}".format(numpy.mean([i for i,j in exp_neg_items_in_top_5])))
    print("\nExp Neg in Top 10 (sampled) validation set: {}".format(numpy.mean([i for i,j in exp_neg_items_in_top_10])))
    print("\nPrecision at len(test) (sampled) validation set: {}".format(numpy.mean([i for i,j in valid_precision_at_len_test])))
    return numpy.mean([i for i,j in valid_recalls]), numpy.mean([i for i,j in valid_precision_at_len_test]),numpy.mean([i for i,j in exp_neg_items_in_top_5]), numpy.mean([i for i,j in exp_neg_items_in_top_10])

def log_binned(val_package,text='default: '):
    """
    Print the values for variouse different groups
    :param val_package:
    :param text:
    """
    i_30=[]
    i_40=[]
    i_50=[]
    i_60=[]
    i_70=[]
    i_80=[]
    i_90=[]
    i_100=[]
    for i,j in val_package:
        if j<=30:
            i_30.append(i)
        elif j<=40:
            i_40.append(i)
        elif j<=50:
            i_50.append(i)
        elif j<=60:
            i_60.append(i)
        elif j<=70:
            i_70.append(i)
        elif j<=80:
            i_80.append(i)
        elif j<=90:
            i_90.append(i)
        elif j<=10000:
            i_100.append(i)
    print("\ntil 30 {}: {}".format(text,numpy.mean(i_30)))
    print("til 40 {}: {}".format(text,numpy.mean(i_40)))
    print("til 50 {}: {}".format(text,numpy.mean(i_50)))
    print("til 60 {}: {}".format(text,numpy.mean(i_60)))
    print("til 70 {}: {}".format(text,numpy.mean(i_70)))
    print("til 80 {}: {}".format(text,numpy.mean(i_80)))
    print("til 90 {}: {}".format(text,numpy.mean(i_90)))
    print("til 100 and more {}: {}\n".format(text,numpy.mean(i_100)))

def printValForAll(valid_users,validation_recall,sess,k=50,nI1=5,nI2=10):
    """
    Handles the output for the Recall, Precision and NITK
    :param valid_users:
    :param validation_recall:
    :param sess:
    :param k:
    :param nI1:
    :param nI2:
    """
    valid_recalls = []
    valid_precision_at_len_test = []
    exp_neg_items_in_top_5 = []
    exp_neg_items_in_top_10 = []
    exp_neg_items_in_top_k = []
    # compute recall in chunks to utilize speedup provided by Tensorflow
    for user_chunk in toolz.partition_all(100, valid_users):
        val_recall, val_precision_at_len_test, exp_neg_in_top_5, exp_neg_in_top_10 = validation_recall.eval(sess,
                                                                                                            user_chunk,k=k,nI1=nI1,nI2=nI2)
        valid_recalls.extend([val_recall])
        exp_neg_items_in_top_5.extend([exp_neg_in_top_5])
        exp_neg_items_in_top_10.extend([exp_neg_in_top_10])
        valid_precision_at_len_test.extend([val_precision_at_len_test])
    flatten = lambda l: [item for sublist in l for item in sublist]
    log_eval_stats_binned(flatten(valid_recalls), flatten(valid_precision_at_len_test),
                          flatten(exp_neg_items_in_top_k), flatten(exp_neg_items_in_top_5),
                          flatten(exp_neg_items_in_top_10))

    print("\nRecall at {} validation set: {}".format(k,numpy.mean([i for i,j in flatten(valid_recalls)])))
    print("\nExp Neg in Top {} (sampled) validation set: {}".format(nI1,numpy.mean([i for i,j in flatten(exp_neg_items_in_top_5)])))
    print("\nExp Neg in Top {} (sampled) validation set: {}".format(nI2,numpy.mean([i for i,j in flatten(exp_neg_items_in_top_10)])))
    print("\nPrecision at {} (sampled) validation set: {}".format(k,numpy.mean([i for i, j in flatten(valid_precision_at_len_test)])))


def optimize(model, sampler, train, valid, test, train_exp_neg, valid_exp_neg, test_exp_neg, epochs=10):
    """
    Optimize the model. TODO: implement early-stopping
    :param model: model to optimize
    :param sampler: mini-batch sampler
    :param train: train user-item matrix
    :param valid: validation user-item matrix
    :param epochs: amount of epochs to run
    :return: None
    """
    merged_summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #if model.feature_projection is not None:
            # initialize item embedding with feature projection
        #    sess.run(tf.assign(model.item_embeddings, model.feature_projection))

        # sample some users to calculate recall validation
        test_users = numpy.random.choice(list(set(test.nonzero()[0])),size=1000,replace=False)

        #Variouse sections which would be needed for tensorboard are commented
        #The reason is that they don't work on the cluster


        # Initiate summary writer and give unique log dir to all
        #logs=str(os.getcwd())+'/train'
        users_name="_not_named"
        #users_name=raw_input("Enter a name for this runs log:")

        #log_dir=logs+"/iters_"+str(epochs*EVALUATION_EVERY_N_BATCHES)+"_time_"+str(datetime.datetime.now()).replace(" ","_")+"__"+str(users_name)
        #if not os.path.exists(log_dir):
        #    os.makedirs(log_dir)
        # copy the metadata
        #copyfile(logs+"/projector_config.pbtxt", log_dir+"/projector_config.pbtxt")
        #create statistics of the run
        #stat_file=open(log_dir+'/stat_file.dat', 'w+')



        #train_writer = tf.summary.FileWriter(log_dir,
        #                                     graph=tf.get_default_graph())
        #saver = tf.train.Saver()
        # init history to plot with matplot
        history=dict()
        history["Recall"] = []
        history["Prec"] = []
        history["NIT5"] = []
        history["NIT10"] = []


        for x in tqdm(xrange(epochs), desc='Epochs running...'):
            # create evaluator on validation set
            validation_recall = RecallEvaluator(model, train, test, train_exp_neg, test_exp_neg)
            # compute recall on validate set
            valid_recalls = []
            valid_precision_at_len_test= []
            exp_neg_items_in_top_k = []
            exp_neg_items_in_top_5 = []
            exp_neg_items_in_top_10 = []

            # compute recall in chunks to utilize speedup provided by Tensorflow
            for user_chunk in toolz.partition_all(100, test_users):
                val_recall, val_precision_at_len_test, exp_neg_in_top_5, exp_neg_in_top_10= validation_recall.eval(sess, user_chunk,k=50)
                valid_recalls.extend([val_recall])
                exp_neg_items_in_top_5.extend([exp_neg_in_top_5])
                exp_neg_items_in_top_10.extend([exp_neg_in_top_10])
                valid_precision_at_len_test.extend([val_precision_at_len_test])
            flatten = lambda l: [item for sublist in l for item in sublist]
            his_rec, hist_prec,his_nit5, hist_nit10=log_eval_stats_binned(flatten(valid_recalls),flatten(valid_precision_at_len_test),flatten(exp_neg_items_in_top_k),flatten(exp_neg_items_in_top_5),flatten(exp_neg_items_in_top_10))

            history["Recall"] += [his_rec]
            history["Prec"] += [hist_prec]
            history["NIT5"] += [his_nit5]
            history["NIT10"] += [hist_nit10]

            #NITK_summary=tf.summary.scalar("NITK", numpy.mean(exp_neg_items_in_top_k))
            # TODO: early stopping based on validation recall
            # train model
            losses = []
            model.merged_summary_op = tf.summary.merge_all()
            #train_writer.add_summary(model.merged_summary_op, x)
            # run n mini-batches
            for i in tqdm(range(EVALUATION_EVERY_N_BATCHES), desc="Optimizing..."):
                user_pos, neg, user_exp_neg, pos_neg_pairs = sampler.next_batch()
                _, loss, summary = sess.run((model.optimize, model.loss, model.merged_summary_op),
                                   {model.user_positive_items_pairs: user_pos,
                                    model.negative_samples: neg,
                                    model.user_exp_neg_items_pairs: user_exp_neg,
                                    model.pos_neg_pairs: pos_neg_pairs})
                #train_writer.add_summary(summary, i + (x * EVALUATION_EVERY_N_BATCHES))
                #saver.save(sess, os.path.join(log_dir, "model.ckpt"), i + (x * EVALUATION_EVERY_N_BATCHES))
                losses.append(loss)
            print "\nEpoch:"+str(x),
            print("\nTraining loss {}".format(numpy.mean(losses)))
        print(10*"\n")
        print("Training has ended!")
        print(10*"\n")

        #calculate recall on test set

        validation_recall = RecallEvaluator(model, train, valid, train_exp_neg, valid_exp_neg)

        valid_users = numpy.random.choice(list(set(valid.nonzero()[0])),size=int(len(list(set(valid.nonzero()[0])))),replace=False)
        printValForAll(valid_users,validation_recall,sess,k=10,nI1=1,nI2=5)
        printValForAll(valid_users,validation_recall,sess,k=20,nI1=10,nI2=20)
        printValForAll(valid_users,validation_recall,sess,k=50,nI1=30,nI2=40)
        printValForAll(valid_users,validation_recall,sess,k=75,nI1=50,nI2=60)
        printValForAll(valid_users,validation_recall,sess,k=100,nI1=70,nI2=80)


        #print 5*"\n"
        #stat_file.writelines("Val_recall at 50 on test set:{}".format(numpy.mean(val_recall))+"\n")
        #stat_file.writelines("NIT5 on test set:{}".format(numpy.mean(exp_neg_in_top_5))+"\n")
        #stat_file.writelines("NIT10 test set:{}".format(numpy.mean(exp_neg_in_top_10))+"\n")
        #stat_file.writelines("Precision at 50 test set: {}".format(numpy.mean(val_precision_at_len_test))+"\n")

        # print "Starting tsne"
        # print model.item_embeddings
        # tsne.tsne(model.item_embeddings, no_dims=2, initial_dims=100, perplexity=30.0)
    print "Starting summary:"
    # plot the current run
    #and ma a file
    pp = PdfPages('LastRunCMLEN_'+""+'.pdf')
    plt.figure(1)
    plt.xlabel('Epochs, each epoch is iterations: '+str(EVALUATION_EVERY_N_BATCHES))
    plt.title('Recall')
    plt.plot(history["Recall"])
    pp.savefig()

    plt.figure(2)
    plt.xlabel('Epochs, each epoch is iterations: '+str(EVALUATION_EVERY_N_BATCHES))

    plt.title('Precision')
    plt.plot(history["Prec"])
    pp.savefig()

    plt.figure(3)
    plt.xlabel('Epochs, each epoch is iterations: '+str(EVALUATION_EVERY_N_BATCHES))
    plt.title('NIT5')
    plt.plot(history["NIT5"])
    pp.savefig()

    plt.figure(4)
    plt.xlabel('Epochs, each epoch is iterations: '+str(EVALUATION_EVERY_N_BATCHES))
    plt.title('NIT50')
    plt.plot(history["NIT10"])
    pp.savefig()

    # End the session
    pp.close()
    sess.close()
    sampler.close()

    #try:
    #    os.system("tensorboard --logdir="+logs)
    #    "Started tensorboard"
    #except:
    #    print "Sth with your log dir is wrong"

    return

if __name__ == '__main__':
    try:
        BATCH_SIZE = int(sys.argv[1])
        N_NEGATIVE = int(sys.argv[2])
        EVALUATION_EVERY_N_BATCHES = int(sys.argv[3])
        EMBED_DIM = int(sys.argv[4])
    except:
        print "The input arguments are wrong, all need to be of type int"

        BATCH_SIZE = 100000
        N_NEGATIVE = 35
        EVALUATION_EVERY_N_BATCHES = 100
        EMBED_DIM = 100
    print "BATCH SIZE: "+str(BATCH_SIZE)+"\n", "Nr Negative: "+str(N_NEGATIVE)+"\n", "Evaluation every n batches: "+str(EVALUATION_EVERY_N_BATCHES)+"\n", "Embedding dimensions: "+str(EMBED_DIM)+"\n"
    # get user-item matrix for citeulike
    # user_item_matrix, features = citeulike(tag_occurence_thres=5)

    start = time.time()
    # Or get user-item matrix for movie100k, and the explicitly negative voted matrix
    #user_item_matrix, features, user_item_exp_neg_matrix = movie20m(tag_occurence_thres=2)
    #Here the decision, which dataset should be chosen is made
    if True:
        user_item_matrix, features, user_item_exp_neg_matrix = books6m(tag_occurence_thres=10)
    if False:
        user_item_matrix, features, user_item_exp_neg_matrix = ml1m(tag_occurence_thres=2)

    # negative and positive matrix have the same shape
    n_users, n_items = user_item_matrix.shape
    print(n_users, n_items )
    # make feature as dense matrix
    dense_features = features.toarray() + 1E-10


    # get train/valid/test user-item matrices
    train, valid, test = split_data(user_item_matrix)
    train_exp_neg, valid_exp_neg, test_exp_neg = split_data(user_item_exp_neg_matrix)

    # create warp sampler
    sampler = WarpSampler(train, train_exp_neg, batch_size=BATCH_SIZE, n_negative=N_NEGATIVE, check_negative=True)

    # WITHOUT features
    # Train a user-item joint embedding, where the items a user likes will be pulled closer to this users.
    # Once the embedding is trained, the recommendations are made by finding the k-Nearest-Neighbor to each user.

    #@sandro Uncommented due to being not needed
    #  model = CML(n_users,
    #             n_items,
    #             # set features to None to disable feature projection
    #             features=None,x
    #             # size of embedding
    #             embed_dim=EMBED_DIM,
    #             # the size of hinge loss margin.
    #             margin=1.9,
    #             # clip the embedding so that their norm <= clip_norm
    #             clip_norm=1,
    #             # learning rate for AdaGrad
    #             master_learning_rate=0.1,
    #
    #             # whether to enable rank weight. If True, the loss will be scaled by the estimated
    #             # log-rank of the positive items. If False, no weight will be applied.
    #
    #             # This is particularly useful to speed up the training for large item set.
    #
    #             # Weston, Jason, Samy Bengio, and Nicolas Usunier.
    #             # "Wsabie: Scaling up to large vocabulary image annotation." IJCAI. Vol. 11. 2011.
    #             use_rank_weight=True,
    #
    #             # whether to enable covariance regularization to encourage efficient use of the vector space.
    #             # More useful when the size of embedding is smaller (e.g. < 20 ).
    #             use_cov_loss=False,
    #
    #             # weight of the cov_loss
    #             cov_loss_weight=1
    #             )

    #optimize(model, sampler, train, valid)

    # WITH features
    # In this case, we additionally train a feature projector to project raw item features into the
    # embedding. The projection serves as "a prior" to inform the item's potential location in the embedding.
    # We use a two fully-connected layers NN as our feature projector. (This model is much more computation intensive.
    # A GPU machine is recommended)
    model = CML(n_users,
                n_items,
                # enable feature projection
                features=dense_features,
                embed_dim=EMBED_DIM,
                margin=2.0,
                clip_norm=1.1,
                master_learning_rate=0.05,
                # the size of the hidden layer in the feature projector NN
                hidden_layer_dim=512,
                # dropout rate between hidden layer and output layer in the feature projector NN
                dropout_rate=0.3,
                # scale the output of the NN so that the magnitude of the NN output is closer to the item embedding
                feature_projection_scaling_factor=1,
                # the penalty to the distance between projection and item's actual location in the embedding
                # tune this to adjust how much the embedding should be biased towards the item features.
                feature_l2_reg=0.2,

                # whether to enable rank weight. If True, the loss will be scaled by the estimated
                # log-rank of the positive items. If False, no weight will be applied.

                # This is particularly useful to speed up the training for large item set.

                # Weston, Jason, Samy Bengio, and Nicolas Usunier.
                # "Wsabie: Scaling up to large vocabulary image annotation." IJCAI. Vol. 11. 2011.
                use_rank_weight=True,

                # whether to enable covariance regularization to encourage efficient use of the vector space.
                # More useful when the size of embedding is smaller (e.g. < 20 ).
                use_cov_loss=False,

                # weight of the cov_loss
                cov_loss_weight=1
                )
    print("befor the optimize")
    optimize(model, sampler, train, valid, test, train_exp_neg, valid_exp_neg, test_exp_neg, epochs=15)
    print("Done optimizing")
    end = time.time()
    print(end - start," Totaltime for execution")
    os._exit
