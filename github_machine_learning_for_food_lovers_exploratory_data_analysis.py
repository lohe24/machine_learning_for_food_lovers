#!/usr/bin/env python
# coding: utf-8

# ![z%20pix%201%20natural%20stream%20grass.JPG](attachment:z%20pix%201%20natural%20stream%20grass.JPG)
# 
# 
# # Machine Learning For Food Lovers: Exploratory Data Analysis On Food Recipes And Recipe Reviews - Part 1
# 
# 
# Data Source & References: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions 

# In[1]:


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


rawrecipes = "RAW_recipes.csv" 
rawinteractions = "RAW_interactions.csv"  

rr = pd.read_csv(rawrecipes, low_memory=False)
ri = pd.read_csv(rawinteractions, low_memory=False,parse_dates=['date'])


# In[3]:


rr.shape, ri.shape


# # Exploratory Data Analysis (EDA): Raw Data
# 
# 
# #### EDA: Recipe Data

# In[4]:


print(rr.shape)
rr.head(3)


# In[5]:


rr.describe()


# In[6]:


rr.describe(include='object')


# In[7]:


rr.hist(figsize=(12,8), bins=100)
plt.show()


# In[8]:


rr.n_ingredients.hist()


# In[9]:


# top 3 - recipes with longest duration 
rr.sort_values(by='minutes', ascending=False)[:3]


# In[10]:


# top 3 - recipes with longest steps 
rr.sort_values(by='n_steps', ascending=False)[:3]


# In[11]:


# top 3 - recipes with most ingredients 
rr.sort_values(by='n_ingredients', ascending=False)[:3]


# #### EDA: User-Reviews Data

# In[12]:


print(ri.shape)
ri.head(3)


# In[13]:


ri.describe()


# In[14]:


ri.describe(include='object')


# In[15]:


ri.hist(figsize=(12,8), bins=100)
plt.show()


# In[16]:


# busiest reviewer 
print(ri.user_id.value_counts(dropna=False).head(5))

plt.figure(figsize=(12,2))
ri.user_id.value_counts().head(20).plot(kind='bar',fontsize=10)
plt.title('most active reviewers', fontsize=12)
plt.ylabel('number of reviews', fontsize=12)
plt.xlabel('user id', fontsize=12)
plt.show()


# In[17]:


reviewer_rating = ri.groupby('user_id').agg(Average_Rating = ('rating','mean'),Total_Reviews=('user_id','count') )

print(reviewer_rating.head())


# In[18]:


from datetime import datetime 

timely = ri.groupby( [ "date"] ).size().to_frame(name = 'count').reset_index()
timely.rename(columns={'date': 'Date', 'count': 'No.of Submissions'}, inplace=True)
timely.head(3)


# In[19]:


timely = timely.set_index('Date')
timely.head(3)
timely.plot(title='Number of recipes submitted over an 18-year period', figsize=(12,5))


# # Cleaning of Raw Data 
# 
# 
# #### Cleaning: Recipes Data

# In[20]:


rr.info()


# In[21]:


# checking id column 
rr.id.value_counts()


# In[22]:


# checking minutes 
rr.minutes.value_counts().tail(10)


# In[23]:


# checking submitted column 
rr.submitted.value_counts()


# In[24]:


# changing submitted to datetime formate 
rr.submitted = pd.to_datetime(rr.submitted, errors='coerce')
rr.info()


# In[25]:


# checking description 
rr.description.value_counts(dropna=False).head(10)


# In[26]:


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


# In[27]:


aaa0 = []
for i in range(0,len(rr)):
    #desc = rr.description.values[i]
    desc0 = re.sub('[^a-zA-Z]', ' ', rr['desc'][i])# 
    desc0 = desc0.lower()
    desc0 = desc0.split() # 
    desc0 = ' '.join(desc0) # place space between each word
    aaa0.append(desc0)

rr['desc_cleaned'] = aaa0
rr['desc_cleaned'] = rr['desc_cleaned'].astype(str)
rr.desc_cleaned = rr.desc_cleaned.replace(r'',np.nan, regex=True)
rr.desc_cleaned.value_counts(dropna=False).head(5)


# In[28]:


rr.name = rr.name.fillna('no_name_recipe') 
rr.desc_cleaned = rr.desc_cleaned.fillna('no_description')
rr.desc_cleaned.replace("nan", 'no_description', inplace=True) 
rr.desc_cleaned.value_counts(dropna=False).head(10)


# #### Cleaning: User Reviews Data

# In[29]:


ri.info()


# In[30]:


ri.user_id = ri.user_id.astype(str)
ri.user_id = ri.recipe_id.astype(str)

ri.review = ri.review.fillna('no_review_comment')

lst_review = []
for i in range(0,len(ri)):
    rev = re.sub('[^a-zA-Z0-9]', ' ', ri['review'][i])
    rev = rev.lower()
    rev =''.join(rev)
    lst_review.append(rev)
    
ri['review_cleaned'] = lst_review
ri.review_cleaned.value_counts(dropna=False).head(10)


# In[31]:


ri.isna().sum()


# # Merge & EDA: Post-Cleaning 
# 
# 
# - merging recipes data with user reviews data on the id of the recipes
# - for the user reviews dataset, some users gave more than one review on a recipe id, so merge all reviews given per recipe
# 
# <strong>NOTE:</strong>  
# - submitted = date recipe was submitted 
# - date = review date (first date from merged review submission dates) 
# 
# 
# #### Merge

# In[32]:


ri.loc[ri.recipe_id == 38]


# In[33]:


# GOAL - merge 
ri2 = ri.copy()

ri2 = ri2.groupby('recipe_id').agg({'date':'first','rating':'mean', 'user_id':'count', 'review':', '.join,
                                                              'review_cleaned':', '.join}).reset_index()

print(f"ri2 shape = {ri2.shape}; ri_merged_review2 = {ri2.shape}\n")
ri2.columns = ['id', 'date', 'rating', 'no_of_ratings' ,'review' , 'review_cleaned']
ri2[:5]


# In[34]:


ri2.no_of_ratings.value_counts().head(10)


# In[35]:


# GOAL - merge  
#ri_merged_review = ri.dropna()
ri_merged_review = ri.copy()

ri_merged_review = ri_merged_review.groupby('recipe_id').agg({'date':'first','rating':'mean', 'user_id':'count',
                                                             'review':', '.join, 'review_cleaned':', '.join}).reset_index()

print(f"ri shape = {ri.shape}; ri_merged_review = {ri_merged_review.shape}\n")
ri_merged_review.columns = ['id', 'date', 'rating', 'number_of_ratings' ,'review', 'review_cleaned']
ri_merged_review[:5]


# In[36]:


recipes_reviews = rr.merge(ri_merged_review, how='left', left_on='id',right_on='id')

print(f"Shape of merged df = {recipes_reviews.shape}\n")
recipes_reviews.head(1)


# In[37]:


recipes_reviews.to_csv("recipes_reviews_merged.csv")


recipes_reviews.info()


# In[38]:


recipes_reviews_file = "recipes_reviews_merged.csv"
df = pd.read_csv(recipes_reviews_file, low_memory=False, parse_dates=['submitted','date'])

df.info()


# In[39]:


df.id.value_counts().head(10)


# In[40]:


df.number_of_ratings.hist()


# In[41]:


# TOP 3 - highly rated, most number of ratings #least time-consuming recipes
df.loc[df.rating >= 4].sort_values(by='number_of_ratings', ascending=False)[:3]


# In[42]:


# TOP 3 - more than 10 ingredients which are the most time-consuming recipes
df.loc[df.n_ingredients >= 10].sort_values(by='minutes', ascending=False)[:3]


# In[43]:


# TOP 3 - few ingredients, time-consuming recipes
df.loc[df.n_ingredients < 10].sort_values(by='minutes', ascending=False)[:3]


# In[44]:


# calculations for non- missing values only
rrbw_df =  df.copy() 


# In[45]:


# TOP 3 - minutes, number of steps 

def gb(n, by, ascending=False, min_minutes=0,min_steps=0):
    
    rrbw = rrbw_df.loc[(rrbw_df.minutes >= min_minutes) & (rrbw_df.n_steps >= min_steps),
                 [by] ].sort_values(by = by, ascending=ascending).head(n).copy()
    
    return rrbw


gb(3,'n_ingredients', ascending=True, min_minutes=1000,min_steps=10)


# In[46]:


gb(3,'n_ingredients', ascending=True, min_minutes=1000,min_steps=15)


# In[47]:


rr_sweet = df.description.str.contains('sweet')
rr_sweet.value_counts()


# In[48]:


rr_sour = df.description.str.contains('sour')
rr_sour.value_counts()


# In[49]:


# recipes with both sweet and sour which are most time-consuming
df.loc[rr_sweet & rr_sour, ['id','name','minutes']].sort_values(by='minutes', ascending=False)[:5]


# In[50]:


# TOP - sour dishes with ratings submitted before yearr 2000; longeest duration in cooking 
recipe_b4_yr2000 = df.submitted.between('1990-01-01','2000-01-01')
rr_sour = df.description.str.contains('sour')
sour_b4_yr2000 = df.loc[recipe_b4_yr2000 & rr_sour,
                       ['name','minutes','n_steps','n_ingredients']].sort_values(by='minutes',ascending=False)
sour_b4_yr2000[:5]


# In[51]:


# newest spicy dish, longest duration, several steps  
spicy = df.description.str.contains('spicy')|df.description.str.contains('chili')|df.description.str.contains('chilli')
duration_min = df.minutes >= 50
steps_min = df.n_steps >= 10
longest_spicy = df.loc[spicy & duration_min & steps_min,
                       ['id','name','submitted','minutes','n_steps','n_ingredients',
                       'rating']].sort_values(by='submitted', ascending=False).set_index('id').head(5)
longest_spicy

