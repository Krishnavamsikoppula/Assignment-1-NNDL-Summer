#!/usr/bin/env python
# coding: utf-8

# In[27]:


# importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# # Naïve Bayes

# In[28]:


# reading "Glass.csv" file
df = pd.read_csv("glass.csv")
df.head()


# In[29]:


# seperating x_data and y_data
y_data = df['Type']
x_data = df.drop('Type', axis=1)


# In[30]:


# x_data
x_data.head()


# In[31]:


# splitting the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=7)


# In[32]:


# train data shape
print(x_train.shape, y_train.shape)


# In[33]:


# test data shape
print(x_test.shape, y_test.shape)


# In[34]:


# training Naive Bayes Model
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)


# In[35]:


# predicting the x_test data using Naive Bayes Model
y_pred = nb_model.predict(x_test)
print(y_pred)


# In[36]:


# Naive Bayes Model score 
print(nb_model.score(x_test, y_test))


# In[37]:


# classification report of Naive Bayes Model
print(classification_report(y_test, y_pred))


# In[ ]:





# # Linear SVM

# In[38]:


# training Linear SVM Model
svm_model = LinearSVC(random_state=6)
svm_model.fit(x_train, y_train)


# In[39]:


# predicting the x_test data using Linear SVM Model
y_pred = svm_model.predict(x_test)
print(y_pred)


# In[40]:


# Linear SVM Model score 
print(svm_model.score(x_test, y_test))


# In[16]:


# classification report of Linear SVM Model
print(classification_report(y_test, y_pred))


# ### Linear SVM has better accuracy than Naive Bayes Model because SVM can perform well in classifying multi-dimentional data and since Naive Bayes is based upon the frequency of occurance it was not able to classify data.

# In[ ]:





# # Linear Regression

# In[41]:


# reading "Salary Data.csv" file
salary_df = pd.read_csv("Salary_Data.csv")
salary_df.head()


# In[42]:


# seperating x_data and y_data
y_data = salary_df['Salary']
x_data = salary_df.drop('Salary', axis=1)


# In[43]:


# x_data
print(x_data.head())


# In[44]:


# splitting the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=(1/3), random_state=7)


# In[45]:


# training Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)


# In[46]:


# predicting the x_test data using Linear Regression Model
y_pred = linear_model.predict(x_test)
print(y_pred)


# In[49]:


# calculating mean square error
mean_squared_error(y_test, y_pred)


# In[48]:


# visualizing x_train data
plt.scatter(x_train, y_train)
plt.xlabel("Years Of Experience")
plt.ylabel("Salary");
plt.title("Experience vs Salary - Train Data");


# In[28]:


# visualizing x_test data
plt.scatter(x_test, y_test)
plt.xlabel("Years Of Experience")
plt.ylabel("Salary");
plt.title("Experience vs Salary - Test Data");


# In[ ]:




