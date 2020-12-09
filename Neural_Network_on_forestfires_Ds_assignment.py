#!/usr/bin/env python
# coding: utf-8

# # Importing the required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


# # Loading the dataset

# In[2]:


forestfires = pd.read_csv(r"C:\Users\Binita Mandal\Desktop\finity\Neural networks\forestfires.csv")


# In[3]:


# First rows of the dataset
forestfires.head()


# In[4]:


# Last rows of the dataset
forestfires.tail()


# ### So we have 516 rows to deal with. Lets get the columns and remove those which does not play any part in data processing

# In[5]:


forestfires.columns


# In[6]:


forestfires.info()


# In[7]:


forestfires.size


# In[8]:


forestfires.shape


# In[9]:


forestfires.dtypes


# In[10]:


# Dropping the month and day columns 
forestfires = forestfires.drop(['month'], axis=1)
forestfires = forestfires.drop(['day'], axis=1)
forestfires.head()


# In[11]:


# Dataset Categorical variables encoding
forestfires['size_category'] = forestfires['size_category'].map({'small':0, 'large':1})


# # Exploratory Data Analysis

# In[12]:


# correlation_matrix

sns.heatmap(forestfires.corr()>0.6, cmap='Greens')


# In[13]:


forestfires.isnull().sum()


# In[14]:


forestfires.describe()


# In[15]:


# Boxplot
sns.boxplot(data = forestfires, orient = "h")


# In[16]:


sns.countplot(forestfires['size_category'])
plt.show()


# In[17]:


# Splitting dataset
X = forestfires.drop(['size_category'], axis=1)
y = forestfires['size_category']


# In[18]:


# Transformation MinMaxScalar
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X = scaler.fit_transform(X)


# In[19]:


# Splitting data into train & test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify= y, test_size=0.2, random_state=42 )


# In[20]:


((X_train.shape, y_train.shape),(X_test.shape, y_test.shape))


# # Neural Network Model

# In[21]:


# generating the data set
from sklearn.datasets import make_classification
X, y = make_classification(n_features =2, n_redundant =0, n_informative=2, random_state=3)


# In[22]:


# Visualization
plt.scatter(X[y==0][:,0], X[y==0][:,1], s=100, edgecolors='k')
plt.scatter(X[y==1][:,0], X[y==1][:,1], s=100, edgecolors='k', marker='^')
plt.show()


# In[23]:


# MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification


# ## Getting mlp score

# In[24]:


from sklearn.neural_network import MLPClassifier
X, y = make_classification(n_features =2, n_redundant=0, n_informative=2, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
mlp = MLPClassifier(max_iter=1000)
mlp.fit(X_train, y_train)
print("accuracy:", mlp.score(X_test, y_test))


# In[25]:


mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,50))


# In[26]:


from sklearn.neural_network import MLPClassifier

#increasing the hidden layers gives more accuracy
clf = MLPClassifier(activation ='relu',solver='lbfgs', alpha=0.0001,hidden_layer_sizes=(3), random_state=1)
clf.fit(X,y)

pred_values = clf.predict(X)
print(pred_values)


# In[27]:


import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

confusion_matrix = confusion_matrix(y,pred_values)
confusion_matrix


# In[28]:


classification_report = classification_report(y,pred_values)
print(classification_report)


# In[29]:


print("Accuracy:",metrics.accuracy_score(y,pred_values))


# ### so from this model we have accuracy of 87. 

# In[ ]:




