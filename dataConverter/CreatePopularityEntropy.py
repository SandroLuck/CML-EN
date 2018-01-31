import csv
import matplotlib.pyplot as plt
from operator import itemgetter

def findMax(prob_per_movie_arr):
    max=[0,0,"error"]
    for id, prob,str in prob_per_movie_arr:
        if prob>max[1]:
            max=[id,prob,str]
    return max
def annotateAndPlot(prob_per_movie_arr,title):
    ax = plt.subplot()
    ax.scatter([0 for i, j, str in prob_per_movie_arr], [j for i, j, str in prob_per_movie_arr])
    for index,elem in enumerate(prob_per_movie_arr):
        ax.annotate(elem[2],(0,elem[1]))
    plt.title(title)
    plt.show(ax)
def onlyAboveThreshold(prob_per_movie_arr):
    toReturn=[[-1,0,""] for i in range(0,100)]
    for elem in prob_per_movie_arr:

        for toRet in toReturn:
            print(elem)
            print(toRet)
            if toRet[1]<elem[1]:
                toReturn[toRet[0]]=elem
    return toReturn

def CalculateEntropy():
    with open('userMlAboveAvg.dat', 'r') as p:
        with open('userMlBelowAvg.dat', 'r') as n:
            with open('u.item', 'r') as item:
                with open('statistics.popularity', 'w+') as w:

                    reader_p = csv.reader(p, delimiter=" ")
                    reader_n = csv.reader(n, delimiter=" ")
                    reader_item=csv.reader(item,delimiter="|")
                    id_to_string=[[i,"error"] for i in range(1800)]
                    for line in reader_item:
                        id_to_string[int(line[0])][1]=line[1]
                    sum_of_likes=0
                    sum_of_dislikes=0

                    like_sum_per_movie=[[i,0] for i in range(1800)]
                    dislike_sum_per_movie=[[i,0] for i in range(1800)]
                    interaction_sum_per_movie=[[i,0] for i in range(1800)]
                    for line in reader_p:
                        sum_of_likes+=len(line)-1
                        for i in range(1,len(line)):
                            like_sum_per_movie[int(line[i])][1]+=1
                            interaction_sum_per_movie[int(line[i])][1]+=1

                    for line in reader_n:
                        sum_of_dislikes+=len(line)-1
                        for i in range(1,len(line)):
                            dislike_sum_per_movie[int(line[i])][1]+=1
                            interaction_sum_per_movie[int(line[i])][1]+=1


                    # well define probability of a like as sum(likes per move)/sum(all likes) and same for dislikes
                    # this gives us negative popularity and positive popularity
                    # probability for a intercation is the sum(interactions)/sum(all interaction)
                    like_probability_per_movie=[[i,0,""] for i in range(1800)]
                    dislike_probability_per_movie=[[i,0,""] for i in range(1800)]
                    interaction_probability_per_movie=[[i,0,""] for i in range(1800)]

                    liked_dislikeratio=[[i,0,""] for i in range(1800)]

                    for id,sum in like_sum_per_movie:
                        like_probability_per_movie[id][1]=sum/float(sum_of_likes)
                        like_probability_per_movie[id][2]=id_to_string[id][1]

                        liked_dislikeratio[id][2]=id_to_string[id][1]

                        if dislike_sum_per_movie[id][1]!=0:
                            liked_dislikeratio[id][1]=like_sum_per_movie[id][1]/float(dislike_sum_per_movie[id][1])
                        else:
                            liked_dislikeratio[id][1]=like_sum_per_movie[id]
                    for id,sum in dislike_sum_per_movie:
                        dislike_probability_per_movie[id][1]=sum/float(sum_of_dislikes)
                        dislike_probability_per_movie[id][2] = id_to_string[id][1]



                    # calculat interaction probability
                    for id,sum in like_sum_per_movie:
                        interaction_probability_per_movie[id][1]=(like_sum_per_movie[id][1]+dislike_sum_per_movie[id][1])/float(sum_of_dislikes+sum_of_likes)
                        interaction_probability_per_movie[id][2]=id_to_string[id][1]

                    likes_minus_dislikeRatio=[[i,0,""] for i in range(1800)]
                    dislike_minus_likeRatio=[[i,0,""] for i in range(1800)]
                    for id,sum,string in interaction_probability_per_movie:
                        if like_sum_per_movie[id][1]+dislike_sum_per_movie[id][1]!=0:
                            likes_minus_dislikeRatio[id][1]=(like_sum_per_movie[id][1]-dislike_sum_per_movie[id][1])/float(like_sum_per_movie[id][1]+dislike_sum_per_movie[id][1])
                            likes_minus_dislikeRatio[id][2]=like_probability_per_movie[id][2]
                        likes_minus_dislikeRatio[id][0]=id
                    #Generate statistics
                    interaction_probability_per_movie=sorted(interaction_probability_per_movie, key=itemgetter(1))
                    dislike_probability_per_movie=sorted(dislike_probability_per_movie, key=itemgetter(1))
                    like_probability_per_movie=sorted(like_probability_per_movie, key=itemgetter(1))
                    likes_minus_dislikeRatio=sorted(likes_minus_dislikeRatio, key=itemgetter(1))
                    w.write("Top ten of category from least to highest \n\n")
                    for i in interaction_probability_per_movie[-10:]:
                        w.write("Interaction ratio top:"+str(i)+str(" like sum: "+str(like_sum_per_movie[i[0]][1])+" dislike sum:"+str(dislike_sum_per_movie[i[0]][1])+" interaction sum:"+str(interaction_sum_per_movie[i[0]][1]))+'\n')
                    w.write("\n")
                    for i in like_probability_per_movie[-10:]:
                        w.write("like ratio top:"+str(i)+str(" like sum: "+str(like_sum_per_movie[i[0]][1])+" dislike sum:"+str(dislike_sum_per_movie[i[0]][1])+" interaction sum:"+str(interaction_sum_per_movie[i[0]][1]))+'\n')
                    w.write("\n")
                    for i in dislike_probability_per_movie[-10:]:
                        w.write("dislik ratio top:"+str(i)+str(" like sum: "+str(like_sum_per_movie[i[0]][1])+" dislike sum:"+str(dislike_sum_per_movie[i[0]][1])+" interaction sum:"+str(interaction_sum_per_movie[i[0]][1]))+'\n')
                    w.write("\n")
                    for i in likes_minus_dislikeRatio[-10:]:
                        w.write("best like/dislike ratio:" + str(i) +str(" like sum: "+str(like_sum_per_movie[i[0]][1])+" dislike sum:"+str(dislike_sum_per_movie[i[0]][1])+" interaction sum:"+str(interaction_sum_per_movie[i[0]][1]))+ '\n')
                    w.write("\n")
                    for i in likes_minus_dislikeRatio[:10]:
                        w.write("worst, most dilikes compared to likes like/dislike ratio:" + str(i) + str(" like sum: "+str(like_sum_per_movie[i[0]][1])+" dislike sum:"+str(dislike_sum_per_movie[i[0]][1])+" interaction sum:"+str(interaction_sum_per_movie[i[0]][1]))+'\n')
                    w.write("\n")
                    for i in likes_minus_dislikeRatio[(int)(len(likes_minus_dislikeRatio)/2-5):(int)(len(likes_minus_dislikeRatio)/2+5)]:
                        w.write("contrversialy like/dislike ratio:" + str(i) +str(" like sum: "+str(like_sum_per_movie[i[0]][1])+" dislike sum:"+str(dislike_sum_per_movie[i[0]][1])+" interaction sum:"+str(interaction_sum_per_movie[i[0]][1]))+ '\n')
                    w.write("\n"+100*"-"+"\n")
                    for item in liked_dislikeratio:
                        if 0.8<item[1]<1.3:
                            w.write(" %s\n" % " ".join(str(x) for x in item))
                    w.write("\n")


CalculateEntropy()
