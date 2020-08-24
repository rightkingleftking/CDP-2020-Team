#!/usr/bin/env python
# coding: utf-8

# # pandas
# ## DataFrame (matrix)

# In[142]:


import pandas as pd
import numpy as np


# In[ ]:





# In[143]:


dates = pd.date_range('20200828',periods=6)


# In[148]:


dates


# In[145]:


df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))


# In[151]:


df.tail(4)


# In[152]:


s = pd.Series(np.random.randint(0,7,size=10))


# In[153]:


s


# In[154]:


s.value_counts()


# In[155]:


df


# In[165]:


df.loc["2020-08-28"]


# In[3]:


l = list(range(10))
l


# In[10]:


a = list(filter(lambda a:a%2==0,l))


# In[11]:


a


# In[ ]:




