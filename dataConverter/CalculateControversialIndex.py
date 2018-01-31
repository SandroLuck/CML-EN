import csv
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

def calculateMovieControversialIndex():
    """
    Convert files to movieControversialIndex like/dislike foreach movie
    out a file movieControversialIndex.dat which is format,
    movieid likes/dislikes amountoflikesanddislikes
    also outputs movieRarity.dat with format
    movieid, zscore(-log(amountOfLikesMovie))
    also outputs userEccentricity.dat with format
    user_id, zscore(sum(positive_iteraction_rarity)/amount_positive_interaction)
    also outputs itemEccentricity.dat with format
    movieid, zscore(sum(user_who_liked_eccentricity)/amount_positive_interaction_for_movie)
    :type ratingForLike: specifiy above which rating is considered a like
    """
    with open('userMlAboveAvg.dat', 'r') as uP:
        with open('userMlBelowAvg.dat', 'r') as uN:
            with open('movieControversialIndex.dat', 'w+') as w:
                with open('movieRarity.dat', 'w+') as rarity:
                    with open('userEccentricity.dat.dat', 'w+') as user_eccentricity_out:
                        with open('itemEccentricity.dat.dat', 'w+') as item_eccentricity_out:
                            # read file into a csv reader
                            readerP = csv.reader(uP, delimiter=" ")
                            tmpP = list(readerP)
                            readerN = csv.reader(uN, delimiter=" ")
                            tmpN = list(readerN)

                            #out_tmp has format id, amountlikes, amountdislikes, amountInteraction
                            out_tmp = [[j,0,0,0] for j in range(2000)]
                            out = [[j,float(1), 0] for j in range(2000)]
                            # count all likes for a movie
                            for id,line in enumerate(tmpP):
                                for rat in line[1:]:
                                    out_tmp[int(rat)][0]=int(rat)
                                    out_tmp[int(rat)][1]=out_tmp[int(rat)][1]+1
                                    out_tmp[int(rat)][3]=out_tmp[int(rat)][3]+1
                            # count all dislikes
                            for id,line in enumerate(tmpN):
                                for rat in line[1:]:
                                    assert out_tmp[int(rat)][0]==int(rat)
                                    out_tmp[int(rat)][2]=out_tmp[int(rat)][2]+1
                                    out_tmp[int(rat)][3]=out_tmp[int(rat)][3]+1

                            for line in out_tmp:
                                out[int(line[0])][0]=int(line[0])
                                #if we have no dislikes just record likes
                                if int(line[2])!=0:
                                    out[int(line[0])][1]=float(float(line[1])/int(line[2]))
                                else:
                                    out[int(line[0])][1]=int(line[1])
                                out[int(line[0])][2]=int(line[3])


                            # Now also calculate Item Rarity
                            # Item rarity is -log(AmountOfInteractionsOnItem)
                            # TODO: improve since now some zero values are being considered
                            item_rarity=[[j,0] for j in range(1683)]
                            for item in item_rarity:
                                if out_tmp[item[0]][3]!=0:
                                    item[1]=-np.log(out_tmp[item[0]][3])
                            item_rarity_array=np.asarray(item_rarity)
                            item_rarity_array_fzscore=zscore(item_rarity_array[:,1])
                            item_rarity_array[:,1]=item_rarity_array_fzscore

                            #Now calculate user Eccentricity
                            user_eccentricity=[[j,0] for j in range(944)]
                            for id,line in enumerate(tmpP):
                                assert user_eccentricity[id+1][0]==id+1
                                count_pos_item=0
                                for rat in line[1:]:
                                    user_eccentricity[id+1][1]+=item_rarity_array[int(rat)][1]
                                    count_pos_item+=1
                                user_eccentricity[id+1][1] /= count_pos_item
                            user_eccentricity_array=np.asarray(user_eccentricity)
                            user_eccentricity_array_fzscore=zscore(user_eccentricity_array[:,1])
                            user_eccentricity_array[:,1]=user_eccentricity_array_fzscore

                            #Now finally calculate item_eccentricity
                            #first calculate sum of eccentric users who liked
                            item_eccentricity=[[j,0] for j in range(1683)]
                            for id,line in enumerate(tmpP):

                                for rat in line[1:]:
                                    item_eccentricity[int(rat)][1]+=user_eccentricity_array[id+1][1]
                            # divide by user amount who liked
                            for item in item_eccentricity:
                                assert item[0]==out_tmp[int(item[0])][0]
                                if out_tmp[int(item[0])][1]!=0:
                                    item[1]/=out_tmp[int(item[0])][1]
                                else:
                                    item[1]=0
                            item_eccentricity_array=np.asarray(item_eccentricity)
                            item_eccentricity_array_fzscore=zscore(item_eccentricity_array[:,1])
                            item_eccentricity_array[:,1]=item_eccentricity_array_fzscore
                            print item_eccentricity
                            for item in item_eccentricity:
                                item_eccentricity_out.write(str(int(item[0]))+" "+str(item[1])+"\n")

                            for item in user_eccentricity_array:
                                user_eccentricity_out.write(str(int(item[0]))+" "+str(item[1])+"\n")
                            #output rarity
                            for item in item_rarity_array:
                                rarity.write(str(int(item[0]))+" "+str(item[1])+"\n")
                            #output controversality
                            for item in out:
                                if item != [-1]:
                                    w.write("%s\n" % " ".join(str(x) for x in item))


def calculateAmountOfControversialMovieWatchedByUser(controversialLower=0.8, controversialUpper=1.3):
    """
    Convert files to userAmountContrversialsLiked foreach user
    out a file userAmountContrversialsLiked.dat which is format,
    userid, controversialLikedAmout, controversialDislikedAmount, likeAmount, dislikeAmount, controversialLike/likesAmount, controversialDislikes/dislikesAmount
    :type ratingForLike: specifiy above which rating is considered a like
    """
    with open('userMlAboveAvg.dat', 'r') as uP:
        with open('userMlBelowAvg.dat', 'r') as uN:
            with open('movieControversialIndex.dat', 'r') as cI:
                with open('userAmountContrversialsLiked.dat', 'w+') as w:
                    # read file into a csv reader
                    readerP = csv.reader(uP, delimiter=" ")
                    tmpP = list(readerP)
                    readerN = csv.reader(uN, delimiter=" ")
                    tmpN = list(readerN)
                    readerC = csv.reader(cI, delimiter=" ")
                    tmpC = list(readerC)
                    # Create and array or enough size for max(user_id) around 950
                    out_tmp = [[j,0,0,0,0,0,0] for j in range(944)]
                    controversial_movies_ids=[]
                    controversialCount=0
                    #calculate total count of controversial movies
                    for movie in tmpC:
                        if (float(movie[1]) > controversialLower) & (float(movie[1]) < controversialUpper):
                            controversialCount+=1

                    # Calculate AMount of controversial Movies liked
                    #id is plus one since
                    for id, user_dat in enumerate(tmpP):
                        out_tmp[id + 1][0] = id + 1
                        for movie_index in range(1, len(user_dat)):
                            #increase like count
                            out_tmp[id+1][3]+=1
                            if (float(tmpC[int(user_dat[movie_index])][1])>controversialLower) & (float(tmpC[int(user_dat[movie_index])][1])<controversialUpper):
                                out_tmp[id + 1][1] += 1
                                controversial_movies_ids.append(tmpC[int(user_dat[movie_index])][0])

                    for id, user_dat in enumerate(tmpN):
                        assert out_tmp[id+1][0]==id+1
                        #increase dislike count
                        for movie_index in range(1, len(user_dat)):
                            out_tmp[id+1][4]+=1
                            if (float(tmpC[int(user_dat[movie_index])][1])>controversialLower) & (float(tmpC[int(user_dat[movie_index])][1])<controversialUpper):
                                out_tmp[id + 1][2] += 1
                    #calculate how many controversial movies a user liked
                    for user_dat in out_tmp:
                        if user_dat[3]>0:
                            user_dat[5]=user_dat[1]/float(user_dat[3])

                    for i in range(0,len(out_tmp)):
                        try:
                            if out_tmp[i][4] != 0:
                                out_tmp[i][6]= out_tmp[i][2] / float(out_tmp[i][4])
                            else:
                                out_tmp[6]= out_tmp[i][2]
                        except:
                            out_tmp[i]=[i,0,0,0,0,0,0]
                            pass;

                    out_tmp=sorted(out_tmp, key=itemgetter(6))
                    for item in out_tmp:
                        w.write("%s\n" % " ".join(str(x) for x in item))
                    # plot
                    out_plot_controLike = []
                    out_plot_controDisLike = []
                    out_plot_y=[]
                    for index,item in enumerate(out_tmp):
                        out_plot_controLike.append(item[5])

                        out_plot_y.append(index)
                        out_plot_controDisLike.append(item[6])
                    plt.scatter(out_plot_y, out_plot_controLike, s=0.1)
                    plt.scatter(out_plot_y, out_plot_controDisLike,s=0.1)
                    plt.axhline(y=controversialCount/float(1682), color='r', linestyle='-')
                    plt.savefig('ControversialUserUsage.eps', format='eps', dpi=10000)
                    print controversial_movies_ids
                    plt.show()

calculateMovieControversialIndex()