# -*- coding: utf-8 -*-
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib import recommendation
from pyspark.mllib.recommendation import *

def parseArtistIdData(p):
	# read a row from artist_data.txt and convert to a tuple having id and artist name
    temp = p.rsplit('\t')
    if len(temp) != 2:
        return []
    else:
        try:
            return [(int(temp[0]), temp[1])]
        except:
            return []

def parseArtistAliasData(q):
	# read a row from artist_alias.txt and convert to a tuple having id of original name and id of alias
    temp = q.rsplit('\t')
    if len(temp) != 2:
        return []
    else:
        try:
            return [(int(temp[0]), int(temp[1]))]
        except:
            return []

def splitLine(line):
    return_value = line.split();
    return int(return_value[0]),int(return_value[1]),int(return_value[2])

conf = SparkConf().setAppName("AudioRecommenderSystem")
sc = SparkContext(conf=conf)

# DATA PATHS 
userArtistDataPath = 's3://aws-logs-021959201754-us-east-1/data/user_artist_data.txt'
artistDataPath = 's3://aws-logs-021959201754-us-east-1/data/artist_data.txt'
artistAliasDataPath = 's3://aws-logs-021959201754-us-east-1/data/artist_alias.txt'

userArtistDataRDD = sc.textFile(userArtistDataPath)
userArtistDataRDD.cache()
print "BEGIN...\n\n"
print "Found " + str(userArtistDataRDD.count()) + " rows from user_artist_data.txt\n"

ArtistDataRDD = sc.textFile(artistDataPath)
ID_artist = dict(ArtistDataRDD.flatMap(lambda x: parseArtistIdData(x)).collect())

ArtistAliasDataRDD = sc.textFile(artistAliasDataPath)
artistAlias = ArtistAliasDataRDD.flatMap(lambda x: parseArtistAliasData(x)).collectAsMap()

artistAliasBroadcast = sc.broadcast(artistAlias)

def mapAliasToOriginalArtistName(x):
	# check if the name is an alias and replace it with original id and name of artist
    if len(x.split())==3:
        userID, artistID, count = splitLine(x)
        finalArtistID = artistAliasBroadcast.value.get(artistID)
        if finalArtistID is None:
            finalArtistID = artistID
        return Rating(userID, finalArtistID, count)
    else:
        return Rating(None,None,None)

filteredUserArtistDataRDD = userArtistDataRDD.filter(lambda y: len(y.split())==3).map(lambda x: mapAliasToOriginalArtistName(x))
filteredUserArtistDataRDD.cache()

print "Model construction BEGIN..."
model = ALS.trainImplicit(filteredUserArtistDataRDD, 10, 5, 0.01)
print "Model construction END\n"

testID = 1000083

recommendations = map(lambda line: ID_artist.get(line.product), model.call("recommendProducts", testID, 10))
print '*********************************************'
print '****      TOP 10 RECOMMENDATIONS ARE     ****'
print '*********************************************'

for r in recommendations:
	print r
print '*********************************************'