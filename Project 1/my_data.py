# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 20:37:59 2019

@author: John
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
import unicodedata
from sklearn.model_selection import train_test_split
import operator

def get_multiwine_data():
    winedata = pd.read_csv("./data/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
    #We found a lot of duplicates in the data set
    winedata = winedata.drop_duplicates('description')
    winedata = winedata[pd.notnull(winedata.price)]
    winedata = winedata[pd.notnull(winedata.country)]

    #There are a lot of varities of wine with identical genetic structure, but names that vary across borders.
    #We merge those
    winedata['variety'] = winedata['variety'].replace(['weissburgunder'], 'chardonnay')
    winedata['variety'] = winedata['variety'].replace(['spatburgunder'], 'pinot noir')
    winedata['variety'] = winedata['variety'].replace(['grauburgunder'], 'pinot gris')
    winedata['variety'] = winedata['variety'].replace(['garnacha'], 'grenache')
    winedata['variety'] = winedata['variety'].replace(['pinot nero'], 'pinot noir')
    winedata['variety'] = winedata['variety'].replace(['alvarinho'], 'albarino')
    
#        "Blend" is not a variety, Roses are blends
    winedata = winedata[winedata.variety.str.contains('blend') == False]
    winedata = winedata[winedata.variety.str.contains('Blend') == False]

    winedata = winedata[winedata.variety.str.contains('rose') == False]
    winedata = winedata[winedata.variety.str.contains('Rose') == False]
    
    #Finally, various tildas and umlauts confuse things, we move them
    def removeaccents(inputstr):
        nfkd_form = unicodedata.normalize('NFKD', inputstr)
        return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    
    winedata['variety'] = winedata['variety'].apply(removeaccents)
    winedata['description'] = winedata['description'].apply(removeaccents)
    
    winedata = winedata[winedata.variety.str.contains('blend') == False]
    winedata = winedata[winedata.variety.str.contains('Blend') == False]

    winedata = winedata[winedata.variety.str.contains('rose') == False]
    winedata = winedata[winedata.variety.str.contains('Rose') == False]
    
    winedata = winedata[winedata.variety.str.contains('Red') == False]
    winedata = winedata[winedata.variety.str.contains('red') == False]
    
    winedata = winedata[winedata.variety.str.contains('white') == False]
    winedata = winedata[winedata.variety.str.contains('White') == False]
    #Still, some varities we just don't have enough data for
    winedata = winedata.groupby('variety').filter(lambda x: len(x) > 1000)    
    return winedata

def get_singlewine_data():
    winedata = pd.read_csv("./data/singlewine-reviews/winequality-red.csv")
    return winedata

#Get the data ready to train by vectorizing, splitting into train/test, etc.
def ready_multiwine_data(data):
    #Now we turn those pesky words into numbers our fellow computers can understand
    Xd = data.drop(['designation','points','province','region_1','region_2','taster_name', 'taster_twitter_handle',
             'title', 'variety','winery'], axis = 1)
    yd = data.variety
    
    
    wine = data.variety.unique().tolist()
    wine.sort()
#    print(wine)
    output = set()
    for x in data.variety:
        x = x.lower()
        x = x.split()
        for y in x:
            output.add(y)
    variety_list = sorted(output)
#    print(variety_list)
    
    X_train, X_test, y_train, y_test = train_test_split(Xd, yd, random_state=1)
#    print("Data split")
    extras = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', 'cab',"%"]
    from nltk.corpus import stopwords
    stop = set(stopwords.words('english'))
    stop.update(variety_list)
    stop.update(extras)
    
    from scipy.sparse import hstack
    desc_vect = CountVectorizer(stop_words = stop, min_df = .05, max_df = 0.9, ngram_range = (1, 3))
    cnty_vect = CountVectorizer(stop_words = stop)

    trdesc = desc_vect.fit_transform(X_train.description)
#    sorted_x = sorted(desc_vect.vocabulary_.items(), key=operator.itemgetter(1))
#    print(sorted_x)
    #trcnty = X_train.country.values[:,None]
    trcnty = cnty_vect.fit_transform(X_train.country)
    trprice = X_train.price.values[:,None]
    X_train_dtm = hstack((trdesc, trcnty,  trprice))
#
#    print("Train data vectorized")
    tedesc = desc_vect.transform(X_test.description)
    #tecnty = X_test.country.values[:,None]
    tecnty = cnty_vect.transform(X_test.country)
    teprice = X_test.price.values[:,None]
    X_test_dtm = hstack((tedesc, tecnty, teprice))
    
#    print(X_train_dtm.shape)
    return X_train_dtm, X_test_dtm, y_train, y_test, wine

def ready_singlewine_data(data):
    Xd = data.drop(['quality'], axis = 1)
    yd = data.quality
    
    scores = data.quality.unique().tolist()
    scores.sort()
    
    X_train, X_test, y_train, y_test = train_test_split(Xd, yd, random_state=1)
    
    from scipy.sparse import hstack

#    trprice = X_train.price.values[:,None]
#    X_train_dtm = hstack([x[1].values[:,None] for x in X_train.iteritems()])
#    X_test_dtm = hstack(X_test)
    
#    print("Train data vectorized")
#    tedesc = desc_vect.transform(X_test.description)
#    #tecnty = X_test.country.values[:,None]
#    tecnty = cnty_vect.transform(X_test.country)
#    teprice = X_test.price.values[:,None]
#    X_test_dtm = hstack((tedesc, tecnty, teprice))
    
#    return X_train_dtm, X_test_dtm, y_train, y_test, scores
    return X_train, X_test, y_train, y_test, scores