import csv
def convertUserData():
    """
    Convert file u.data to file userAvgRating.dat
    Into a format which in the first pos has the UserId and in the second pos has his average rating,
    This makes mainly sense for Movielense Rating system from 1-5
    """
    with open('ratings.csv', 'r') as f:
        with open('userAvgRating.dat', 'w+') as w:
            with open('ratingStat.dat', 'w+') as stat:
                # read file into a csv reader
                reader = csv.reader(f, delimiter=",")
                tmp = list(reader)
                # Create and array or enough size for max(user_id) around 950
                out = [[0 for i in range(3)] for j in range(60000)]
                # Create a rating statistic to know how much rating of each we have and what global average is
                statistic=[0,0,0,0,0]
                global_avg=[0,0]
                # for each line
                for line in tmp[1:]:
                    print(line[0])
                    # add user id at pos 0
                    # add rating sum at pos 1
                    # add rating amount at pos 2, should be the amount of ratings in total
                    out[int(line[0])][0]= int(line[0])
                    out[int(line[0])][1]+=int(line[2])
                    out[int(line[0])][2]+=1
                    #update statistic count ratings
                    statistic[int(line[2])-1]+=1
                    #update global avg sum pos 0, count of ratings pos 1
                    global_avg[0]+=int(line[2])
                    global_avg[1]+=1

                # after all liked movies hae been writen to line nr==userid
                # add length of all likes to first line and write the movie_ids behind
                for item in out:
                    #if we are not dividing by zero
                    # calculate avg user rating
                    if item[2]!=0:
                        w.write(str(item[0])+" "+str(item[1]/float(item[2]))+"\n")
                # write down rating statistics for datasets
                stat.write("Amounts of 1 ratings:"+str(statistic[0])+"\nAmounts of 2 ratings:"+str(statistic[1])+"\nAmounts of 3 ratings:"+str(statistic[2])+"\nAmounts of 4 ratings:"+str(statistic[3])+"\nAmounts of 5 ratings:"+str(statistic[4])+"\n\n")
                stat.write("Amount of ratings total:"+str(global_avg[1])+"\n")
                stat.write("Global avg rating:"+str(global_avg[0]/float(global_avg[1])))
convertUserData()