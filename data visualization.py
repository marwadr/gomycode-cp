#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("titanic-passengers.csv",delimiter = ";")
df


# In[2]:


df.info()


# In[55]:


df.columns.values


# In[57]:


total = titanic_passengers_df.isnull().sum().sort_values(ascending=False)
percent_1 = df.isnull().sum()/df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)


# In[79]:


df.isnull().sum()


# In[70]:


df['Age'].fillna(df['Age'].mean())


# In[4]:


df ['FamilySize'] = df ['SibSp'] + df ['Parch'] + 1


# In[3]:


df.drop('Cabin',axis = 1,inplace = True )


# In[8]:


def plot_correlation_map( df ):

    corr = df.corr()

    s , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    s = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

        )


# In[89]:


df


# In[85]:


df.groupby('Pclass').Survived.value_counts()


# In[90]:


g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[88]:


df.drop(('Name'),axis = 1,inplace = True)


# In[8]:


combine = [df]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab( df['Title'], df['Sex'])


# In[ ]:




