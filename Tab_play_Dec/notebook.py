#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


train_df = pd.read_csv("train.csv").sample(1000)


# In[3]:


test_df = pd.read_csv("test.csv").sample(1000)


# In[4]:


train_df.drop("Id", axis=1, inplace=True)
test_df.drop("Id", axis=1, inplace=True)


# In[5]:


#Data Preprocessing: is target imbalanced?


# In[6]:


y = train_df['Cover_Type']
#Values of y
print(f'In y there are {y.value_counts().sum()}')


# In[7]:


train_df['Cover_Type']


# In[8]:


#which is my target? It's called Cover_type column
print(f"Elements in target columns are divided as follow:\n{y.value_counts(sort=True)}")
print(f"Instead in percentile elements in target columns are divided as follow:\n{y.value_counts(normalize=True,sort=True)}")
#Target column is balanced
print(y.value_counts(normalize=True).plot.pie())


# As I see in the pie chart y values 3,4,5,6,7 are very small.
# On 4.000.000 values 3+4+5+6+7(267.077 values) are equal to 6,67%
# So I'd probably drop all the values  mentioned above

# In[9]:


#Redefine Y. How?
#One hot encoder; Then take  first column: all 0  values are probably equal to 2 in y --> target = pd.get_dummies(y)[1]
#I don't have this problem with DecisionTree
y.head()


# In[10]:


#Analysis of Dataframe


# In[11]:


#Missing values
train_df.isnull().sum().sum()


# In[12]:


train_df.columns


# In[13]:


features = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
    'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
    'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
    'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
    'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
    'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
    'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
    'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']


# In[14]:


soil_features = [x for x in features if x.startswith("Soil_Type")]
train_df["soil_type_count"] = train_df[soil_features].sum(axis=1)


wilderness_features = [x for x in features if x.startswith("Wilderness_Area")]
train_df["wilderness_area_count"] = train_df[wilderness_features].sum(axis=1)

train_df=  train_df.drop(['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
             'Wilderness_Area4','Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
             'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12','Soil_Type13', 'Soil_Type14','Soil_Type15',  'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
             'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
             'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
             'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
             'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40'], axis = 1)


# In[15]:


new_feature = train_df.columns.tolist()


# What does these features mean?
# #Ricordarsi che i valori sono generati da artificialmente a un'intelligenza artificiale

# In[16]:


train_df.head()


# In[17]:


train_df.describe().transpose()


# All variables are quantitative:
# Discrete one are: Cover_Type, Aspect, Slope, soil_type_count, wilderness_area_count
# Variables containing values not allowed:
# Aspect has values range [0??,360??]: so it doesn't allowed negative values and values greater than 360
# Distances can't have negatives values; I will solve this issue changing them in Manhattan distances
# Hillshade is measured within this interval [0??,255??]: so it doesn't allowed negative values and values greater than 255
# 
# Consider that dataset is generated by GANs, this kaggle discussion (https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/293612) has determined that Transforming data in such a way that it responds to real constraints increases prediction accuracy

# In[18]:


#Correlation
plt.figure(figsize=(12, 10))
data_corr = round(train_df[new_feature ].corr(),1)
sns.heatmap(data_corr, annot = True)


# As heatmap can show us there is no correlation between the variables

# In[19]:


#Check variance


# In[20]:


#Pairplot
sns.pairplot(train_df[new_feature])
#--> Considering that there isn't correlation between variables, Pairplot is useless


# In[21]:


fig, ax = plt.subplots(1, len(new_feature), figsize=(50, 15))
for i, col in enumerate(new_feature):
    sns.histplot(train_df[col], bins=50, ax=ax[i])


# In[22]:


plt.figure(figsize=(10,5))
sns.histplot(data =train_df)


# In[23]:


# BoxPlot for Aspect given target class
plt.figure(figsize=(12, 8))
sns.boxplot( data=train_df, orient= 'h',width=0.5);


# In[24]:


#Box plot Cover type and Elevation
plt.figure(figsize=(10,5))
sns.boxplot(data= train_df,x="Elevation",y="Cover_Type",orient = 'h',width=0.5)


# In[25]:


#Boxplot between Horizontal_distance_to_hydrology
plt.figure(figsize=(10,5))
sns.boxplot(data= train_df, x= "Horizontal_Distance_To_Hydrology", y="Cover_Type",orient="h")


# In[26]:


# BoxPlot between Aspect and Cover_type
plt.figure(figsize=(12, 5))
sns.boxplot(x="Aspect", y="Cover_Type", data=train_df,orient="h", width=0.5)


# Feature engineering: I've done it in lib.py file;
#  more info on this kaggle discussion: https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/293612
# 
