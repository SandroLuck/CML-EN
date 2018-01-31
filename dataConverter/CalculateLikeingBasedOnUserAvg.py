import csv
def convertUserDataDependingOnUserAvg():
    """
    Convert file u.data to file usersML.dat
    Into a format which is understood by CML.py
    :type ratingForLike: specifiy above which rating is considered a like
    """
    with open('userAvgRating.dat', 'r') as r:
        with open('ratings.csv', 'r') as f:
            with open('uAboveAvgBooks.dat', 'w+') as wP:
                with open('uBelowAvgBooks.dat', 'w+') as wN:
                    # read file into a csv reader
                    reader = csv.reader(f, delimiter=",")
                    tmp = list(reader)
                    # Create and array or enough size for max(user_id) around 950
                    out_pos = [[-1 for i in range(1)] for j in range(60000)]
                    out_neg = [[-1 for i in range(1)] for j in range(60000)]

                    #read average rating per user
                    reader_rat=csv.reader(r, delimiter=' ')
                    rat_array=list(reader_rat)
                    print("Rating:",rat_array)
                    print("Rating at [2][1]",rat_array[2][1])
                    # create enough space for the rateing
                    rating = [-1 for j in range(60000)]
                    # creat array where avg rating of user is at pos userid
                    for line in rat_array:
                        rating[int(line[0])]=float(line[1])

                    # for each line
                    for line in tmp[1:]:
                        # if the rating is considered a like write into the line==user_id
                        # and add all movie_ids which the user liked to that line
                        # likeing means the rating is above the users avg rating
                        if (float(line[2]) > float(rating[int(line[0])])):
                            if out_pos[int(line[0])] == [-1]:
                                out_pos[int(line[0])] = [int(line[1])]
                            else:
                                out_pos[int(line[0])].append(int(line[1]))
                        # else it is below his preference
                        elif(float(line[2]) <= float(rating[int(line[0])])):
                            if out_neg[int(line[0])] == [-1]:
                                out_neg[int(line[0])] = [int(line[1])]
                            else:
                                out_neg[int(line[0])].append(int(line[1]))
                    # after all liked movies hae been writen to line nr==userid
                    # add length of all likes to first line and write the movie_ids behind
                    for item in out_pos:
                        if item != [-1]:
                            wP.write(str(len(item)) + " %s\n" % " ".join(str(x) for x in item))
                    for item in out_neg:
                        if item != [-1]:
                            wN.write(str(len(item)) + " %s\n" % " ".join(str(x) for x in item))

convertUserDataDependingOnUserAvg()