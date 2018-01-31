import csv

def calculateUserSimilarityMatrix():
    with open('userMlAboveAvg.dat', 'r') as u:
        with open('userSimilarityMatrix.dat', 'w+') as w:
            reader = csv.reader(u, delimiter=" ")
            tmp = list(reader)
            # Create and array or enough size for max(user_id) around 950
            out = [[] for j in range(1000)]
            out_matrix = [[-50 for i in range(1000)] for j in range(1000)]

            # count all likes for a movie
            for id, line in enumerate(tmp):
                for item in line:
                     #TODO BE REALY CAREFUL HERE SINCE ID+1 if used directly with tensorflow remember that id+1 only holds for movielens and
                     #TODO id is the right index for tensorflow
                    out[id+1].append(item)
            for id1, user1 in enumerate(out):
                for id2, user2 in enumerate(out):
                    out_matrix[id1][id2]=calculateUserSimilarityIndex(user1, user2)
            #TODO think how to use this for tensorflow
            print out_matrix[1]
            print out_matrix[2]
def calculateUserSimilarityIndex(user1Items, user2Items):
    try:
        hits = 0
        for item1 in user1Items:
            if item1 in user2Items:
                hits+=1
        return ((float(hits)/len(user1Items))+(float(hits)/len(user2Items)))/float(2)
    except:
        return -1

calculateUserSimilarityMatrix()
