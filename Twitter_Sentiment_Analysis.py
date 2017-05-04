# coding: utf-8

# In[ ]:

#import other libraries
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# import SQLContext and data types
from pyspark.sql import SQLContext
from pyspark.sql.types import *

#sc is a SparkContext.
sqlContext = SQLContext(sc)

# fetching the data of the twitter tweets from the parquet file
# the parquet file containis all the tweets from last 30 seconds
tweet_data = sqlContext.read.parquet("swift://notebooks.spark/tweetsFull.parquet")
tweet_data.registerTempTable("tweets");
sqlContext.cacheTable("tweets")
tweets = sqlContext.sql("SELECT * FROM tweets")
tweets.cache()

# In[ ]:

#create an array that will hold the count for each sentiment
sentimentDistribution=[0] * 9
#store the data in the array
for i, sentiment in enumerate(tweets.columns[-9:]):
    sentimentDistribution[i]=sqlContext.sql("SELECT count(*) as sentCount FROM tweets where " + sentiment + " > 60")
        .collect()[0].sentCount


# In[ ]:

get_ipython().magic('matplotlib inline')
ind=np.arange(9)
width = 0.35
bar = plt.bar(ind, sentimentDistribution, width, color='g', label = "distributions")

#Setting graph parameters
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*2.5, plSize[1]*2) )
plt.ylabel('Count of Tweets')
plt.xlabel('Sentiment Tone')
plt.title('Distribution of tweets by sentiments > 60%')
plt.xticks(ind+width, tweets.columns[-9:])
plt.legend()
plt.show()

# In[ ]:

#here write the topic that you want to search.
#for example, "#Bigdata" will fetch tweets related to big data
from operator import add
import re
rdd = tweets.flatMap( lambda t: re.split("s", t.text))
    .filter( lambda word: word.startswith("#") )
    .map( lambda word : (word, 1 ))
    .reduceByKey(add, 10).map(lambda (a,b): (b,a)).sortByKey(False).map(lambda (a,b):(b,a))
top10_tweets_with_tag = rdd.take(10)


# In[ ]:


#Setting graph parameters
get_ipython().magic('matplotlib inline')
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*2, plSize[1]*2) )
labels = [i[0] for i in top10_tweets_with_tag]
sizes = [int(i[1]) for i in top10_tweets_with_tag]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', "beige", "paleturquoise", "pink", "lightyellow", "coral"]
plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.show()

# In[ ]:

cols = tweets.columns[-9:]
def expand( t ):
    ret = []
    for s in [i[0] for i in top10_tweets_with_tag]:
        if ( s in t.text ):
            for tone in cols:
                ret += [s + u"-" + unicode(tone) + ":" + unicode(getattr(t, tone))]
    return ret
def makeList(l):
    return l if isinstance(l, list) else [l]

#creating the rdd
rdd = tweets.map(lambda t: t )
rdd = rdd.filter( lambda t: any(s in t.text for s in [i[0] for i in top10_tweets_with_tag] ) )

#creating  a flatMap
rdd = rdd.flatMap( expand )
rdd = rdd.map( lambda fullTag : (fullTag.split(":")[0], float( fullTag.split(":")[1]) ))

#formating the data
rdd = rdd.combineByKey((lambda x: (x,1)),
                  (lambda x, y: (x[0] + y, x[1] + 1)),
                  (lambda x, y: (x[0] + y[0], x[1] + y[1])))

#reindexing the map
rdd = rdd.map(lambda (key, ab): (key.split("-")[0], (key.split("-")[1], round(ab[0]/ab[1], 2))))
rdd = rdd.reduceByKey( lambda x, y : makeList(x) + makeList(y) )

#Sorting the tuples
rdd = rdd.mapValues( lambda x : sorted(x) )

#mapping the values
rdd = rdd.mapValues( lambda x : ([elt[0] for elt in x],[elt[1] for elt in x])  )

#sorting the tweet entries
def customCompare( key ):
    for (k,v) in top10_tweets_with_tag:
        if k == key:
            return v
    return 0
rdd = rdd.sortByKey(ascending=False, numPartitions=None, keyfunc = customCompare)

#taking the mean score
top10_tweets_with_tagMeanScores = rdd.take(10)

# In[ ]:

get_ipython().magic('matplotlib inline')

#Setting graph parameters
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*3, plSize[1]*2) )
top5tagsMeanScores = top10tagsMeanScores[:5]
width = 0
ind=np.arange(9)
(a,b) = top5tagsMeanScores[0]
labels=b[0]
colors = ["beige", "paleturquoise", "pink", "lightyellow", "coral", "lightgreen", "gainsboro", "aquamarine","c"]
idx=0
for key, value in top5tagsMeanScores:
    plt.bar(ind + width, value[1], 0.15, color=colors[idx], label=key)
    width += 0.15
    idx += 1
plt.xticks(ind+0.3, labels)
plt.ylabel('Average Score')
plt.xlabel('Tones')
plt.title('Breakdown of top hashtags by sentiment tones')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center',ncol=5, mode="expand", borderaxespad=0.)
plt.show()
