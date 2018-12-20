# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.draw.dispersion import dispersion_plot

data = pd.read_csv("googleplaystore.csv")
data2 = pd.read_csv("googleplaystore_user_reviews.csv")

#%% cleaning google play store dataset
print("Google Play Store Dataset")
print(data.columns)

#%% cleaning columns
data.columns = data.columns.str.lower()
data.columns = data.columns.str.replace(" ","_")
    
#%% unwanted data
print(data[data.rating>5]) #unwanted data
data = data.drop(data[data.rating>5].index)        
wanttodrop = data[['last_updated', 'current_ver', 'android_ver']]
data = data.drop(wanttodrop,axis = 1)
data.drop_duplicates(subset='app', keep='first', inplace=True) #dublicated datas
    
#%% cleaning installs section
data.installs = data.installs.str.replace("+","")
data.installs = data.installs.str.replace(",","")
data.installs = data.installs.apply(lambda x : int(x))
    
#%% cleaning price section
data.price = data.price.str.replace("$","")
data.price = data.price.apply(lambda x : float(x))
    
#%%cleaning reviews section
data.reviews = data.reviews.apply(lambda x : int(x))
    
#%% rating mean of each category
rating = data["rating"]
category = data["category"]

#%% cleaning google play store review dataset
print("***********************************")
print("Google Play Store Review Dataset")
print(data2.columns)
print(data2.head())

data2 = data2.dropna(how = "any")   # dropping null data
data2 = data2.reset_index(drop = True)
data2.Translated_Review = data2.Translated_Review.str.lower() #making them lower so we can work on it easily
data2.Sentiment = data2.Sentiment.str.lower() 
freq = nltk.FreqDist(data2.Translated_Review) 

#%%
class Googleplaystore:
    #%% how mony app in each category
    def appincategory():
        plt.figure(figsize=(20,6))
        data.category.value_counts().plot(kind="pie",autopct = "%1.0f%%",pctdistance = 0.9,radius = 1.3,figsize = (12,12))
        plt.ylabel(" ")
        plt.show()    

    #%% avarege ratings of categories
    def ratingofcategory():
        print(data.groupby(category).rating.mean().head())
        data.groupby(category).rating.mean().sort_values(ascending = False).plot(kind = "bar",figsize = (12,12))
        plt.xlabel("categories")
        plt.ylabel("ratings")
        plt.title("Avarege ratings of categories")
        plt.show()
    
    #%% Total and average price of categories
    def priceofcategory():
        print(data.groupby(category)["price"].sum())
        data.groupby(category).price.mean().plot(kind="barh",figsize=(12,12))
        plt.xlabel("price",fontsize = 10)
        plt.title("Average price of categories")
        plt.show()
    
    #%% Total review of apps
    def totalreviewofapps():
        print(data.groupby("app").reviews.sum().sort_values(ascending = False).head())
    
    #%% Total review of each category
    def totalreviewofcat():
        data.groupby(category).reviews.sum().sort_values(ascending = False).plot(kind="bar",figsize = (10,10))
        plt.xlabel("category")
        plt.ylabel("reviews")  
        plt.title("Total review of each category")
        plt.show()
    
    #%% Rating Frequancy
    def ratingfrequency():
        print(data.rating.describe())
        data.rating.value_counts().plot(kind = "bar")
        plt.title("Rating Frequency")
        plt.xlabel("Ratings")
        plt.ylabel("Rating Counts")
        plt.show()
#%%                


class Googleplayreview:
    #%% most common list with plot 
    def mostcommonlist():
        print(freq.most_common(15))
        freq.plot(30,cumulative = False)
        
    #%% measure of a word’s homogeneity across the parts of a corpus
    def wordhomogenity():
        dispersion_plot(data2.Translated_Review,["good","awesome","usefull","love",
                                                 "brilliant","great","amazing","best"])

    #%% value of sentiments
    def valueofsentiments():
        print(data2.Sentiment.value_counts())
        data2.Sentiment.value_counts().plot(kind = "bar",figsize = (6,6))
        
    #%% sentiments of facebook apps
    def facebookapps():
        print(data2[data2["App"].str.contains("Facebook")])
        facebook_data = data2[data2["App"].str.contains("Facebook")]
        facebook_data.Sentiment.value_counts().plot(kind="pie",autopct = "%1.0f%%",pctdistance = 0.9,radius = 1.2,figsize = (10,10))
#%%
Googleplaystore.appincategory()
#Googleplaystore.ratingofcategory()
#Googleplaystore.priceofcategory()
#Googleplaystore.totalreviewofapps()
#Googleplaystore.totalreviewofcat()
#Googleplaystore.ratingfrequency()

#Googleplayreview.mostcommonlist()
#Googleplayreview.wordhomogenity()
#Googleplayreview.valueofsentiments()
#Googleplayreview.facebookapps()
