# This Converter converts the Movie 100k to CMl compatibility
#
import csv


def convertUserData(ratingForLike):
    """
    Convert file u.data to file usersML.dat
    Into a format which is understood by CML.py
    :type ratingForLike: specifiy above which rating is considered a like
    """
    with open('u.data', 'r') as f:
        with open('userMl.dat', 'w+') as w:
            # read file into a csv reader
            reader = csv.reader(f, delimiter="\t")
            tmp = list(reader)
            # Create and array or enough size for max(user_id) around 950
            out = [[-1 for i in range(1)] for j in range(950)]
            # for each line
            for line in tmp:
                # if the rating is considered a like write into the line==user_id
                # and add all movie_ids which the user liked to that line
                if (int(line[2]) >= int(ratingForLike)):
                    if out[int(line[0])] == [-1]:
                        out[int(line[0])] = [int(line[1])]
                    else:
                        out[int(line[0])].append(int(line[1]))
            # after all liked movies hae been writen to line nr==userid
            # add length of all likes to first line and write the movie_ids behind
            for item in out:
                if item != [-1]:
                    w.write(str(len(item)) + " %s\n" % " ".join(str(x) for x in item))


def convertGenreData():
    """
    Convert file u.data to file GenreItemML.dat
    Into a format which is understood by CML.py
    """
    with open('u.item', 'r') as f:
        with open('GenreItemMl.dat', 'w+') as w:
            # read file into a csv reader
            reader = csv.reader(f, delimiter="|")
            tmp = list(reader)
            # Create and array or enough size for max(genre) around 20
            out = [[-1 for i in range(1)] for j in range(500)]
            # for each line
            for line in tmp:
                # and add all movie_ids to the line they are a genre of
                # ignoring the tag 0 (if i start at 1 else it keeps it) which is unknown an probably unwanted
                for i in range(1, 19):
                    # The genre offset in the fiel u.item is 5 such that we can inncrement i simply
                    genreOffset = 5
                    # if the value is a -1 overwrite it since its only a placeholder
                    if (out[i] == [-1]) & (int(line[genreOffset + i]) == 1):
                        out[i] = [int(line[0])]
                    # if there is already data append the new movie id
                    elif int(line[genreOffset + i]) == 1:
                        out[i].append(int(line[0]))
            # after all movie ids have been writen to the genre line they are part of
            # add length of all likes to first line and write the movie_ids behind
            for item in out:
                if item != [-1]:
                    w.write(str(len(item)) + " %s\n" % " ".join(str(x) for x in item))


#convertGenreData()
convertUserData(4)
