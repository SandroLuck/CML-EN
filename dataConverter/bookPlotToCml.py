import csv

def bookTagToCml(file_in='book_tags.csv', tag_count_thres=5):
    """
    Convert file u.data to file book_tag_cml.dat
    Into a format which is understood by CML.py
    """
    with open(file_in, 'r') as f:
        with open('book_tag_cml.dat', 'w+') as w:
            gid_bid=getDictGoodBookIdToBookId()
            # read file into a csv reader
            reader = csv.reader(f, delimiter=",")
            tmp = list(reader)
            #Create a dict tag-list of movie ids with tag
            tag_dict=dict()
            #File is order 0=BookId,1=Tag_id,2=count of people that tagged so(i think)

            for line in tmp[1:]:
                if int(line[2])>=tag_count_thres:
                    if int(line[1]) in tag_dict:
                        tag_dict[int(line[1])].append(int(line[0]))
                    else:
                        tag_dict[int(line[1])]=[int(line[0])]
            # after all movie ids have been writen to the new plot tag line, which have now different ids but contain the movie ids which are in the movie lense ids
            # add length of all plot tags infront of the dataset
            for tag in tag_dict:
                w.write(str(len(tag_dict[tag]))+" %s\n" % " ".join(str(gid_bid[x]) for x in tag_dict[tag]))
def getDictGoodBookIdToBookId():
    """
    Necessary to convert the goodbooks dataset
    :return:
    """
    gid_bid=dict()
    with open('books.csv', 'r') as f:
        reader = csv.reader(f, delimiter=",")
        tmp = list(reader)
        for line in tmp[1:]:
            gid_bid[int(line[1])]=int(line[0])
    return gid_bid

bookTagToCml()
