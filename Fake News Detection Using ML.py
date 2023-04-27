#!/usr/bin/env python
# coding: utf-8

# ## import library ##

# In[52]:


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# ## inserting fake and real datase ##

# In[53]:


df = pd.read_csv("news.csv")


# In[54]:


df


# In[55]:


df.head(10)


# In[56]:


df.loc[(df['label']==0), ['label']]='FAKE'
df.loc[(df['label']==1), ['label']]='REAL'


# In[57]:


x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state = 42)


# In[58]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[59]:


vectorization = TfidfVectorizer(stop_words='english', max_df=0.7)


# In[60]:


xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# In[61]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[62]:


paclass=PassiveAggressiveClassifier(max_iter=50)


# In[64]:


paclass.fit(xv_train, y_train)


# In[66]:


pred=paclass.predict(xv_test)


# In[68]:


xy=accuracy_score(y_test, pred)
print(f'Accuracy: {round(xy*100,2)}%')


# In[ ]:




