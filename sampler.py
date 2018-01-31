import numpy
from multiprocessing import Process, Queue
from scipy.sparse import lil_matrix
from random import shuffle



def sample_function(user_item_matrix, user_item_exp_neg_matrix, batch_size, n_negative, result_queue, check_negative=True):
    """

    :param user_item_matrix: the user-item matrix for positive user-item pairs
    :param batch_size: number of samples to return
    :param n_negative: number of negative samples per user-positive-item pair
    :param result_queue: the output queue
    :return: None
    """
    user_item_matrix = lil_matrix(user_item_matrix)
    user_item_pairs = numpy.asarray(user_item_matrix.nonzero()).T


    user_to_positive_set = {u: set(row) for u, row in enumerate(user_item_matrix.rows)}

    #user exp negative parts
    user_item_exp_neg_matrix = lil_matrix(user_item_exp_neg_matrix)
    user_item_exp_neg_pairs = numpy.asarray(user_item_exp_neg_matrix.nonzero()).T
    user_to_exp_neg_set = {u: set(row) for u, row in enumerate(user_item_exp_neg_matrix.rows)}

    #TODO: the following might become a ram issue, if problems arise improve memory handling
    #here the negative-positive pairs are sampled
    exp_neg_and_pos_pairs=set()
    for u, row in enumerate(user_item_matrix.rows):
        neg_items=user_to_exp_neg_set.get(u)
        for item in row:
            for neg_item in neg_items:
                exp_neg_and_pos_pairs.add((item, neg_item))
        if len(exp_neg_and_pos_pairs)>100000000:
            break
    exp_neg_and_pos_pairs = numpy.asarray([x for x in exp_neg_and_pos_pairs])
    counter_batchs=0
    while True:
        numpy.random.shuffle(user_item_pairs)
        numpy.random.shuffle(user_item_exp_neg_pairs)
        numpy.random.shuffle(exp_neg_and_pos_pairs)
        for i in range(int(len(user_item_pairs) / batch_size)):

            user_positive_items_pairs = user_item_pairs[i * batch_size: (i + 1) * batch_size, :]
            user_exp_neg_items_pairs = user_item_exp_neg_pairs[i * batch_size: (i + 1) * batch_size, :]
            pos_neg_pairs = exp_neg_and_pos_pairs[i*batch_size: (i + 1) * batch_size]


            # sample negative samples
            negative_samples = numpy.random.randint(
                0,
                user_item_matrix.shape[1],
                size=(batch_size, n_negative))

            # Check if we sample any positive items as negative samples. OR explicitly negative samples as negative samples
            # Note: this step can be optional as the chance that we sample a positive item is fairly low given a
            # large item set.
            if check_negative:
                for user_positive, negatives, i in zip(user_positive_items_pairs,
                                                       negative_samples,
                                                       range(len(negative_samples))):
                    user = user_positive[0]
                    for j, neg in enumerate(negatives):
                        # Sandro added the logic that exp negative sample should not be in the normal negative set
                        while neg in user_to_positive_set[user]:
                            negative_samples[i, j] = neg = numpy.random.randint(0, user_item_matrix.shape[1])
            result_queue.put((user_positive_items_pairs, negative_samples, user_exp_neg_items_pairs, pos_neg_pairs))
            counter_batchs+=1
        #print('MAX amount batches possible:',counter_batchs)

class WarpSampler(object):
    """
    A generator that, in parallel, generates tuples: user-positive-item pairs, negative-items

    of the shapes (Batch Size, 2) and (Batch Size, N_Negative)
    """

    def __init__(self, user_item_matrix, user_item_exp_neg_matrix, batch_size=10000, n_negative=10, n_workers=5, check_negative=True):
        self.result_queue = Queue(maxsize=n_workers*2)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(user_item_matrix,
                                                      user_item_exp_neg_matrix,
                                                      batch_size,
                                                      n_negative,
                                                      self.result_queue,
                                                      check_negative)))
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:  # type: Process
            p.terminate()
            p.join()
