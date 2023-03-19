#!/usr/bin/env python
# coding: utf-8

# # Machine Learning For Food Lovers: Exploratory Data Analysis On Food Recipes And Reviews - Part 1
#
#
# #### Article 1:
# - Merging the recipes and recipe reviews datasets
# - Analyzing the recipes and reviews data
#
# ![image.png](attachment:image.png)
#
# Data Source & References: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

# In[1]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import ast
from numpy import nan

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer


pd.options.display.max_columns = 30
pd.options.display.float_format ='{:.2f}'.format


# In[2]:


wrecipes = "RAW_recipes.csv"

rawinteractions = "RAW_interactions.csv"


rr = pd.read_csv(rawrecipes, low_memory=False)
ri = pd.read_csv(rawinteractions, low_memory=False,parse_dates=['date'],delimiter=',', encoding="utf-8-sig")


# # Data Preparation
#
# Exploring, cleaning and preparing the raw data
#
# #### Recipe Data

# In[3]:


print(rr.shape)
rr.head(3)


# In[4]:


rr.describe()


# In[5]:


rr.describe(include='object')


# In[6]:


rr.hist(figsize=(12,8), bins=100)
plt.show()


# In[7]:


rr.n_ingredients.hist()


# In[8]:


# checking for duplicated recipe IDs
rr.id.value_counts(dropna=False)


# In[9]:


rr[rr.duplicated(keep=False)].sort_values(by="id")
# id column below shows NO duplicates


# In[10]:


rr.isna().sum()


# In[11]:


# top 3 - recipes with longest duration
rr.sort_values(by='minutes', ascending=False)[:3]


# In[12]:


# top 3 - recipes with longest steps
rr.sort_values(by='n_steps', ascending=False)[:3]


# In[13]:


# top 3 - recipes with most ingredients
rr.sort_values(by='n_ingredients', ascending=False)[:3]


# #### User-Reviews Data

# In[14]:


print(ri.shape)
ri.head(3)


# In[15]:


ri.describe()


# In[16]:


ri.describe(include='object')


# In[17]:


ri.hist(figsize=(12,8), bins=100)
plt.show()


# In[18]:


print(ri['review'].groupby(ri['user_id']).count().sort_values(ascending = False))


# In[19]:


reviewer_rating = ri.groupby('user_id').agg(Average_Rating = ('rating','mean'),Total_Reviews=('user_id','count') )

print(reviewer_rating.head())


# In[20]:


from datetime import datetime

timely = ri.groupby( [ "date"] ).size().to_frame(name = 'count').reset_index()
timely.rename(columns={'date': 'Date', 'count': 'No.of Submissions'}, inplace=True)
timely.head(3)


# In[21]:


timely = timely.set_index('Date')
#timely.rename(columns={'submitted': 'Date', 'count': 'No.of Submissions'}, inplace=True)
timely.head(3)
timely.plot(title='Number of recipes submitted over an 18-year period', figsize=(12,5))


# # Cleaning of Raw Data
#
#
# #### Cleaning: Recipes Data

# In[22]:


rr.info()


# In[23]:


# changing submitted to datetime formate

rr.submitted = pd.to_datetime(rr.submitted, errors='coerce')
rr.info()


# In[24]:


# checking description

rr.description.value_counts(dropna=False).head(10)


# In[25]:


rr['desc'] = rr['description'].copy()
rr['desc'] = rr['desc'].astype(str)

corpus = []
for i in range(0,len(rr)):
    review = re.sub('<.*?>','',rr['desc'][i])
    review = re.sub('[^a-zA-Z0-9]', ' ', review)
    review = review.lower()
    review =''.join(review)
    corpus.append(review)

rr['desc'] = corpus
rr['desc'] = rr['desc'].astype(str)
rr.desc.value_counts(dropna=False).head(5)


# In[26]:


rr['ingredients'] =  rr['ingredients'].apply(lambda x: x.replace('[','').replace(']',''))
rr['ingredients'] = rr['ingredients'].astype(str)
rr['steps'] =  rr['steps'].apply(lambda x: x.replace('[','').replace(']',''))
rr['steps'] = rr['steps'].astype(str)
rr['tags'] =  rr['tags'].apply(lambda x: x.replace('[','').replace(']',''))
rr['tags'] = rr['tags'].astype(str)
aaa0 = []
for i in range(0,len(rr)):
    desc0 = re.sub('[^a-zA-Z]', ' ', rr['desc'][i])
    desc0 = desc0.lower()
    desc0 = desc0.split()
    desc0 = ' '.join(desc0)
    aaa0.append(desc0)

rr['desc_cleaned'] = aaa0
rr['desc_cleaned'] = rr['desc_cleaned'].astype(str)
rr.desc_cleaned = rr.desc_cleaned.replace(r'',np.nan, regex=True)

print(rr.shape)
rr = rr.dropna()
rr.isna().sum()


# #### Cleaning: User Reviews Data
# - removing punctuation marks from text data
#
#
#
# _Note:_
# In this article, we limit cleaning of the text data to removal of punctuation. In the next article, we will consider further data preparation techniques such as stemming, removal of stopwords et.

# In[27]:


ri.info()


# In[28]:


ri.user_id = ri.user_id.astype(str)
ri.recipe_id = ri.recipe_id.astype(str)
ri.review = ri.review.fillna('no_review_comment')
lst_review = []
for i in range(0,len(ri)):
    rev = re.sub('[^a-zA-Z0-9]', ' ', ri['review'][i])
    #rev = ri['review'][i]
    rev = rev.lower()
    rev =''.join(rev)
    lst_review.append(rev)

ri['review_cleaned'] = lst_review

print(f"ri shape = {ri.shape}")
ri[['review', 'review_cleaned']][-5:]


#
# #### Merging
# - merging recipes data with user reviews data
#
# <strong>_Note:_</strong>
# - submitted = date recipe was submitted
# - date = review date (for the merged dataset, I choose the first date from the review submission dates)

# In[29]:


print(rr.shape, ri.shape)
ri.tail(3)


# In[30]:


# GOAL - merge
#ri_merged_review = ri.dropna()
ri_merged_review = ri.copy()

ri_merged_review = ri_merged_review.groupby('recipe_id').agg({'date':'first','rating':'mean', 'user_id':'count',
                                                             'review':', '.join, 'review_cleaned':', '.join}).reset_index()

print(f"ri shape = {ri.shape}; ri_merged_review = {ri_merged_review.shape}\n")
ri_merged_review.columns = ['id', 'date', 'rating', 'number_of_ratings' ,'review', 'review_cleaned']
ri_merged_review.id = ri_merged_review.id.astype(int)
ri_merged_review[:5]


# In[31]:


recipes_reviews = rr.merge(ri_merged_review, how='left', left_on='id',right_on='id')

print(f"Shape of merged df = {recipes_reviews.shape}\n")
recipes_reviews.head(1)


# In[32]:


recipes_reviews.to_csv("recipes_reviews_merged.csv")
recipes_reviews.info()


# In[33]:


recipes_reviews_file = "recipes_reviews_merged.csv"
df = pd.read_csv(recipes_reviews_file, low_memory=False, parse_dates=['submitted','date'])

df.info()


# In[34]:


print(f"min = {df.number_of_ratings.min()}, max={df.number_of_ratings.max()}")
df.number_of_ratings.hist()


# In[35]:


df.columns


# In[36]:


df = df[['name', 'id', 'minutes', 'submitted', 'desc_cleaned',
         'n_steps', 'n_ingredients', #'desc',
         'rating', 'number_of_ratings', 'review_cleaned']]


# In[37]:


# TOP 3 - highly rated, most number of ratings
df.loc[df.rating >= 4].sort_values(by='number_of_ratings', ascending=False)[:3]


# In[38]:


# TOP 3 - several ingredients, time-consuming recipes
df.loc[df.n_ingredients >= 10].sort_values(by='minutes', ascending=False)[:3]


# In[39]:


# TOP 3 - few ingredients, time-consuming recipes
df.loc[df.n_ingredients < 10].sort_values(by='minutes', ascending=False)[:3]


# In[40]:


# TOP 3 - most ingredents, time consuming or at least 100000 minute duration, at least 15 steps
# calculations
rrbw_df =  df[df['n_steps']!=0]

def gb(n, by, ascending=False, min_minutes=0,min_steps=0):

    rrbw = rrbw_df.loc[(rrbw_df.minutes >= min_minutes) & (rrbw_df.n_steps >= min_steps),
                 [by] ].sort_values(by = by, ascending=ascending).head(n).copy()
    return rrbw
gb(3,'n_ingredients', ascending=False, min_minutes=10,min_steps=15)


# In[41]:


rr_sweet = df.desc_cleaned.str.contains('sweet')
rr_sweet.value_counts()


# In[42]:


rr_sour = df.desc_cleaned.str.contains('sour')
rr_sour.value_counts()


# In[43]:


# recipes both sweet and sour, with longest duration
df.loc[rr_sweet & rr_sour, ['id','name','minutes']].sort_values(by='minutes', ascending=False)[:5]


# In[44]:


# TOP - sour dishes submitted before yr 2000; longeest duration in cooking
recipe_b4_yr2000 = df.submitted.between('1990-01-01','2000-01-01')
rr_sour = df.desc_cleaned.str.contains('sour')

sour_b4_yr2000 = df.loc[recipe_b4_yr2000 & rr_sour,
                       ['name','minutes','n_steps','n_ingredients']].sort_values(by='minutes',ascending=False)

sour_b4_yr2000[:5]


# In[45]:


# newest spicy dish, longest duration, several steps
spicy = df.desc_cleaned.str.contains('spicy')|df.desc_cleaned.str.contains('chili')|df.desc_cleaned.str.contains('chilli')
duration_min = df.minutes >= 50
steps_min = df.n_steps >= 10

longest_spicy = df.loc[spicy & duration_min & steps_min,
                       ['id','name','submitted','minutes','n_steps','n_ingredients',
                       'rating']].sort_values(by='submitted', ascending=False).set_index('id').head(5)

longest_spicy


# In[46]:


# newest spicy dish, shortest duration, few steps
highly_rated = df.rating >= 3
most_rated = df.number_of_ratings >= 1
spicy = df.desc_cleaned.str.contains('spicy')|df.desc_cleaned.str.contains('chili')|df.desc_cleaned.str.contains('chilli')
duration_min = df.minutes <= 1000
steps_min = df.n_steps <= 45

myoptions = df.loc[spicy & duration_min & steps_min & highly_rated & most_rated,
                       ['id','name','submitted','minutes','n_steps','n_ingredients',
                       'rating','number_of_ratings']].sort_values(by='number_of_ratings', ascending=False).set_index('id').head(5)
myoptions


# In[47]:


# TOP 50 - highly rated, most number of ratings #least time-consuming recipes
df_highly_rated = df.loc[df.rating >= 4].sort_values(by='number_of_ratings', ascending=False)[:50]
df_highly_rated


# In[48]:


# IDs of top 50 highly rated recipes with ratings higher or equal to 4 and with the most number of ratings
df_highly_rated.id.values


# In[49]:


df_highly_rated.plot(x="id", y=["number_of_ratings"], kind="bar", figsize=(16, 6),
                     title="Top 50 most highly rated recipes")
plt.show()
