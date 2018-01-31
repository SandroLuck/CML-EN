# example: https://api.themoviedb.org/3/search/movie?api_key=9b8eaa8ca6e8b8b7b72222de17a88c45&query=GoldenEye&primary&release_year=1995
# example get keywords https://api.themoviedb.org/3/movie/862/keywords?api_key=9b8eaa8ca6e8b8b7b72222de17a88c45
# This script is in python 3 not python 2.7 like the rest
print("starting:")
import time

import http.client
import json
from pprint import pprint
import csv
import codecs
import re
def getPlotTagsForMovies():
    #build connection
    """
    This functions find the plot tags for all the movies in the Movielens dataset
    and writes them to the file MovielensIdPlotTags.dat
    This functions runs for around 15 min
    1800 entries each 0.5 secs
    """
    conn = http.client.HTTPSConnection("api.themoviedb.org")
    payload = "{}"

    #open the files
    with codecs.open('MovieLensIdToMovieDbId.dat', 'r', "iso-8859-2") as f:
        with codecs.open('MovielensIdPlotTags.dat', 'w+', "iso-8859-2") as w:
            # read file into a csv reader
            reader = csv.reader(f, delimiter=" ")
            tmp = list(reader)
            # Create and array or enough size for max(movie_ids) around 1800
            movielens_id_to_imdb_id = dict()
            error_counter = 0
            # create space for the out array
            out = [[j for i in range(1)] for j in range(1700)]


            # create the dict
            for line in tmp:
                try:
                    #print line such that we know programm is running
                    print(line[0])
                    tmp_req="https://api.themoviedb.org/3/movie/"+line[1]+"/keywords?api_key=9b8eaa8ca6e8b8b7b72222de17a88c45"
                    conn.request("GET",
                                 tmp_req,
                                 payload)
                    res = conn.getresponse()
                    data = res.read()
                    # read the returned api json
                    data = json.loads(data.decode('iso-8859-2'))
                    data_keys=data['keywords']
                    for keyword in data_keys:
                        out[int(line[0])].append(int(keyword['id']))
                    time.sleep(0.35)
                except Exception as e:
                    # print exception if any
                    print("Exception:" + str(e))
                    print(data)
                    print(tmp_req)
                    error_counter = error_counter + 1
            for item in out:
                if item != [-1]:
                    w.write("%s\n" % " ".join(str(x) for x in item))
            print("We had that many errors:"+str(error_counter))


def convertMovieLenseIdToTheMovieDbId():
    """
    This functions find the IMDB ids for all the movies in the Movielens dataset
    and writes them to the file MovieLensIdToMovieDbId.dat
    This functions runs for around 15 min
    1800 entries each 0.5 secs
    """
    #build connection
    conn = http.client.HTTPSConnection("api.themoviedb.org")
    payload = "{}"

    #open the files
    with codecs.open('u.item', 'r', "iso-8859-2") as f:
        with codecs.open('MovieLensIdToMovieDbId.dat', 'w+', "iso-8859-2") as w:
            # read file into a csv reader
            reader = csv.reader(f, delimiter="|")
            tmp = list(reader)
            # Create and array or enough size for max(movie_ids) around 1800
            out = dict()
            error_counter=0
            for i in range(0,1700):
                out[i]=-1
            # for each line
            for line in tmp:
                try:
                    #print line such that we know the programm is running should go up to ~1700
                    print(line[0])
                    # strip the numbers from the file such that only movie title is left
                    tmp_movie_name = re.sub('\\(.*?\\)', '', line[1]).rstrip().replace(" ","+")
                    # find the movie release year such that right movie can be found in api
                    tmp_release_year = re.findall('\([0-9]*\)',line[1])[0].replace('(',"").replace(')',"")

                    # generate the api string, only works for ascii strings since api is slightly wierd with french symbols
                    tmp_req=("https://api.themoviedb.org/3/search/movie?api_key=9b8eaa8ca6e8b8b7b72222de17a88c45&query="+tmp_movie_name.encode('ascii','replace').decode('ascii','replace')+"&primary_release_year="+tmp_release_year)                    # call the api
                    conn.request("GET",
                                 tmp_req,
                                 payload)
                    res = conn.getresponse()
                    data = res.read()
                    # read the returned api json
                    data = json.loads(data.decode('iso-8859-2'))

                    # read the imdb id we are interested in
                    out[int(line[0])]=data['results'][0]['id']
                    # sleep a few moments since the api only allows 40 items per 10 secs
                    time.sleep(0.35)
                except Exception as e:
                    #retry movie without release year since somtimes the release years differ in the dat sets
                    try:
                        tmp_movie_name = re.sub('\\(.*?\\)', '', line[1]).rstrip().replace(" ","+")
                        # find the movie release year such that right movie can be found in api

                        # generate the api string, only works for ascii strings since api is slightly wierd with french symbols
                        tmp_req=("https://api.themoviedb.org/3/search/movie?api_key=9b8eaa8ca6e8b8b7b72222de17a88c45&query="+tmp_movie_name.encode('ascii','replace').decode('ascii','replace'))                    # call the api
                        conn.request("GET",
                                     tmp_req,
                                     payload)
                        res = conn.getresponse()
                        data = res.read()
                        # read the returned api json
                        data = json.loads(data.decode('iso-8859-2'))

                        # read the imdb id we are interested in
                        out[int(line[0])]=data['results'][0]['id']
                        # sleep a few moments since the api only allows 40 items per 10 secs
                        time.sleep(0.3)
                    except Exception as e:
                        # print exception if any
                        print("Exception:" + str(e))
                        print(data)
                        print(tmp_req)
                        error_counter=error_counter+1

            #after we found our imdb ids print them to the file
            list_key_value = [[k, v] for k, v in out.items()]
            for k, v in list_key_value:
                w.write(str(k) + " " + str(v) + "\n")
            print("We had that many errors:"+str(error_counter))


#getPlotTagsForMovies()
#convertMovieLenseIdToTheMovieDbId()
