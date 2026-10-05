#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,Binarizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[73]:


economy=pd.read_csv("Indian_Economy_Sectorwise_Dataset.csv")


# In[74]:


economy


# In[75]:


economy.isnull().sum()


# In[76]:


economy.describe()


# In[77]:


economy.info()


# In[78]:


economy["Sector"].unique()


# In[79]:


economy["Sector"].value_counts()


# In[80]:


Y=economy.groupby("Year")["GDP_Lakh_Crore"]


# In[81]:


Y.plot(kind='bar')


# In[82]:


s=economy.groupby("Sector")["GDP_Lakh_Crore"]


# In[83]:


s.plot(kind='pie')


# In[84]:


scaler=MinMaxScaler()


# In[85]:


economy["Exports_Crore"]=scaler.fit_transform(economy[["Exports_Crore"]])


# In[86]:


economy["Imports_Crore"]=scaler.fit_transform(economy[["Imports_Crore"]])


# In[87]:


economy["FDI_Crore"]=scaler.fit_transform(economy[["FDI_Crore"]])


# In[88]:


economy.head()


# In[89]:


Binary=Binarizer(threshold=3.0)


# In[90]:


economy["Inflation_%"]=Binary.fit_transform(economy[["Inflation_%"]])


# In[91]:


economy.head()


# In[92]:


economy["Tax_Revenue_Crore"]=scaler.fit_transform(economy[["Tax_Revenue_Crore"]])


# In[131]:


x=economy[["GDP_Lakh_Crore"]]
y=economy[["Year"]]


# In[132]:


x_train,x_test,y_train,y_test=train_test_split(
x,
y,
test_size=0.20,
random_state=42)


# In[133]:


model=LinearRegression()


# In[134]:


model.fit(x_train,y_train)


# In[141]:


y_pred=model.predict(x)


# In[145]:


y_pred


# In[ ]:


scaler= StandardScaler()


# In[146]:


x_scaled=scaler.fit_transform(x)
y_scaled=scaler.fit_transform(y)


# In[148]:


import joblib

joblib.dump(
    model,
    "Indian_Economy_Yearly_GDP"
)

joblib.dump(
    scaler,
    "Indian_Economy_Yearly_GDP.pkl"
)


# In[ ]:




