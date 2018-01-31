import csv

def convertPlotTagsToCml():
    """
    Convert file u.data to file GenreItemML.dat
    Into a format which is understood by CML.py
    """
    with open('MovielensIdPlotTags.dat', 'r') as f:
        with open('PlotTagItemCMl.dat', 'w+') as w:
            # read file into a csv reader
            reader = csv.reader(f, delimiter=" ")
            tmp = list(reader)
            # Create and array or enough size for max(plottags) around 999'999 we are only considering tags up to 1mio since there are only 4 tags which have a bigger id
            # but they go up to billions in foreign languages
            out = [[-1 for i in range(1)] for j in range(999999)]
            # for each line
            for plot_tags in tmp:
                counter=0
                for item in plot_tags:
                    if counter!=0:
                        if out[int(item)]==[-1]:
                            out[int(item)]=[plot_tags[0]]
                        else:
                            out[int(item)].append(plot_tags[0])
                    counter=counter+1
            # after all movie ids have been writen to the new plot tag line, which have now different ids but contain the movie ids which are in the movie lense ids
            # add length of all plot tags infront of the dataset
            for item in out:
                if item!=[-1]:
                    print(item)
                    w.write(str(len(item))+" %s\n" % " ".join(str(x) for x in item))

#convertPlotTagsToCml()