#!/usr/bin/env python
# coding: utf-8

# ## Load packages

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# ## Load data

# In[2]:


data = pd.read_csv('cars.csv')


# ## Drop bogus data features

# In[3]:


# Drop 'location_region' column because data is bogus
data = data.drop(['location_region'], axis=1)

# Drop "feature_n" columns because I'm unsure what data they convey
for i in range(10):
    data = data.drop(['feature_'+str(i)], axis=1)


# ## Seperate out features that need to be transformed to numeric

# In[4]:


bool_features = []
strg_features = []
features = data.columns
for feature in features:
    print('Feature = {}, Type = {}'.format(feature,type(data[feature][0])))
    if type(data[feature][0]) == np.bool_:
        bool_features.append(feature)
    if type(data[feature][0]) == str:
        strg_features.append(feature)


# ## Convert boolean data features to numeric 1/0
# *Boolean data can be converted to 1 for True or 0 for False to encode it numerically for downstream machine learning.*

# In[5]:


for bool_feature in bool_features:
    data.loc[data[bool_feature] == True, bool_feature] = 1
    data.loc[data[bool_feature] == False, bool_feature] = 0


# ## One-hot-encode string data features
# *Features that have string data (non-boolean) with more than 2 possible values need to be one-hot-encoded so that downstream machine learning algorithm can read the data properly. You can't just convert the different string values to integers (1, 2, 3, ... 81, ...) because machine learning algorithms will ascribe higher value to high integers when their should not be. For these data features, all values are equally valuable, they're just different and should be encoded as such.*

# In[6]:


for strg_feature in strg_features:
    # Print current feature's unique values
    print('Feature {}, Unique values = {}'.format(strg_feature, len(data[strg_feature].unique())))
    # One-hot-encode current categorical feature
    one_hot_encoder = OneHotEncoder(sparse=False)
    cur_vector = np.array(data[strg_feature].values).reshape(-1,1)
    cur_one_hot = one_hot_encoder.fit_transform(cur_vector)
    # Create column headings for one-hot-encoded categorical feature
    cols_to_add = []
    for i in range(cur_one_hot.shape[1]): cols_to_add.append(strg_feature+'_'+str(i))
    # Drop categorical feature and replace it with one-hot-encoded version
    data = data.drop(strg_feature, axis=1)
    data[cols_to_add] = cur_one_hot


# In[8]:


# Print out size of transformed data
print('Data size = {} x {}'.format(data.shape[0],data.shape[1]))


# In[10]:


# Print out sample of transformed data
print(data.head())


# In[ ]:




