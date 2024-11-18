#!/usr/bin/env python
# coding: utf-8

# # Modelling on the sklearn dataset
# 
# 
# 
# 

# In[5]:


from sklearn import datasets
diabetes = datasets.load_diabetes()


# In[6]:


#Dataset description and its components
diabetes


# In[7]:


print(diabetes.DESCR)


# In[8]:


#Featuring components names
print(diabetes.feature_names)


# In[9]:


#Creating X and Y data matrices

X = diabetes.data
Y = diabetes.target


# In[10]:


X.shape, Y.shape


# In[16]:


#Creating X and Y data matrices another way
# X, Y = datasets.load_diabetes(return_X_y=True)


# In[17]:


#Datasplit

from sklearn.model_selection import train_test_split


# In[18]:


#Perform 80/20 Data split
#80-train and 20-test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# # Building Linear Regression Model
# 

# In[27]:


#Importing Libraries
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[28]:


#Building linear regression
#First: Definition of the regression model

model = linear_model.LinearRegression()


# In[29]:


#Building the training model

model.fit(X_train, Y_train)


# In[33]:


#Aplication of the trained model to make a predition

Y_pred = model.predict(X_test)


# # Prediction results
# 
# 

# In[35]:


#Printing model performance

print('Coeffictients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error MSE: %.2f' % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f' % r2_score(Y_test, Y_pred))


# In[37]:


print(diabetes.feature_names)


# In[39]:


#30.93861856*(age) -204.03993943*(sex)...=152.18250865968872


# # String formatting

# In[41]:


#r2_score by default returns a floating number
r2_score(Y_test, Y_pred)


# In[42]:


r2_score(Y_test, Y_pred).dtype


# In[ ]:


#That's why previously was used a %.2f
#which is a rounding operator -modulo-
#originally %f


# # Scatter plot

# In[43]:


#Data

Y_test


# In[44]:


Y_pred


# In[46]:


#Making the scatter plot

import seaborn as sns

sns.scatterplot(Y_test, Y_pred)


# In[50]:


import matplotlib.pyplot as plt
sns.scatterplot(x=Y_test, y=Y_pred, marker='+')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Scatterplot of Predicted vs True Values')
plt.show()


# In[51]:


#Making the dots more transluscent 
sns.scatterplot(x=Y_test, y=Y_pred, alpha = 0.5)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Scatterplot of Predicted vs True Values')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




