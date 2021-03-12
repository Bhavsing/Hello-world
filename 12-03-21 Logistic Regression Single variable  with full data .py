#!/usr/bin/env python
# coding: utf-8

# ### Simple and Univariate Logistic Regression using Age , Purchase The End goll is to find the best fit cow line 

# # What is the Lagistic Regression is a Clasification target 
# # Target Variable is a Purchase 
# # Gender, Age, EstimatedSalary 
# # This is the Classification problems
# 
# ## Let's Compaer with Linear & Logistic Regression
# Logstic Regression is some kind of motification of the Linear Regression
# Linear Regression simple continue data 
# 
# Logistic Regression range is 0 to 1 
# 
# ### the Equation of the Linear Regression
# Linear Regression y = b0 + b1 * X 
# Error formula c=y-y^ c=(y-y^)2 Sum(y-y^)2 
# Sigmoid Function  p=1/1 +e -y
# 
# ### Logistic Regression Equation 
# log(p/1-p) = b0 +b1 * x
# 
# ### for example look like outliers but not outliers wrong prediction
# let's see
# 
# ## Logistic Regression Error formula
# Binary cross entropy or log loss
# -(y(log(p))+(1-y)*log(1-p))
# 
# ### what is Error is nothing but making wrong prediction or deviation 
# Probabulity range is 0 to 1 
# 
# Y == is Actual value 
# p == is a pred value
# 
# ## calculate the Error now 
# actual value is  y = 1 
# predict value is y^ = 0.2 
# 
# -(1 * log(0.2)+(1-1)*log(1-0.2))
# =- (log(0.2)) =- (-0.698) = 0.698 # - * - + that's why +ve value
# 
# ## Which ever sigmoid cow have a list Error that cow should be is a Best Logistic Regression Line 
# 
# ## If Actual value is 0 What is the Equation 
# if Y =0 The equation should be -((1-y)*log(1-p))
# 

# In[141]:


# For Remebring Logistic Equation
# Logistic will work 

# (y = b0 + b1 *x
#p=1/(1+e^-y)
#p=1/(1+e^-(b0 + b1 *x))
#(1+e^-(b0 + b1 *x)) =1/p
#e^-(b0 + b1 *x)=1/p-1
#e^-(b0 + b1 *x)=1-p/p
#-(b0 + b1 *x)=log(1-p/p)
#-(-(b0 + b1 *x))=-(log(1-p/p))
#b0 + b1 *x=log(1-p/p)^-1
#b0 + b1 *x=log(p/1-p)
#log(p/1-p) = b0 + b1 *x -- Logistic Equation)


# In[142]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[143]:


df = pd.read_csv('Social_Network_Ads.csv')


# In[144]:


df.info()


# In[145]:


df.head() # top 5 rows
# Target Variable is a Purchase 
# Gender, Age, EstimatedSalary 
# This is the Classification problems 


# In[146]:


df.tail()# Bottom 5 rows


# In[147]:


df


# In[148]:


df.shape


# #### Let's Do the Preprocessing 

# #### Handile The Missing value

# In[149]:


df.isnull().any()
# don't have any null values in this data


# In[150]:


df.boxplot() # there is no outliers


# In[151]:


sns.boxplot(df["EstimatedSalary"])


# In[152]:


sns.scatterplot(df["Age"],df["EstimatedSalary"])
# i want to superat ethre will purchase or not use hue


# In[153]:


# use hue
sns.scatterplot(df["Age"],df["EstimatedSalary"],hue=df["Purchased"])


# ### Done the Preprocessing 
# 1. there is No any null & outliers 
# 

# ### for practical going first Age with Purchase

# # Split Dependent and Independent Variable

# In[154]:


x=df.iloc[:,2:3] # let we go with Age independet variable
x
# 2 3 we need the 2 dimensional data that's why will go for 2,3 


# In[155]:


y=df.iloc[:,4]
y


# #### Split the Train And Test 
# Devoid the data into 4 parts 

# In[156]:


from sklearn.model_selection import train_test_split


# In[157]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)


# In[158]:


x_train.shape


# In[159]:


y_test.shape


# In[160]:


plt.scatter(x,y)# x is a Age y is a purchase


# #### Next Step Scaling
# due to clasification problem

# #### The Algorithms will be requaire the scaling part
# 1. Linear Regression Gradient Descent Approach due to de/dm count
# 2. Logistic Regression
# 3. SVM
# 4. KNN
# 5. Gradient Boosting
# 6. Neural Networks all 

# ### Scaling
# Min Max Scaler formula -- xscaled = x-xmin/(xmax - xmain
# 0 to 1 min max range, Min Max Scaler formula
# 
# 
# 
# ####  What Scaling ? 
# Nothing but Getting the data into one single range,
# 
# what is the range of minimax scaler 0 to 1
# 
# ####  Why Scaling  ?
# Applying Scaling means some of the data pitns it may have far a way So some of grediant algorithm my effected to models
# 

# In[161]:


# Min Max Scaler formula -- xscaled = x-xmin/(xmax - xmain
# 0 to 1 min max range, Min Max Scaler formula
from sklearn.preprocessing import MinMaxScaler


# In[162]:


sc=MinMaxScaler()


# In[163]:


x_train=sc.fit_transform(x_train)


# In[164]:


x_test=sc.fit_transform(x_test)


# # Next step Building the Model
# logistic regression is a linear regression base model

# In[165]:


from sklearn.linear_model import LogisticRegression


# In[166]:


lr_model=LogisticRegression()


# In[167]:


lr_model.fit(x_train, y_train) # fit & train the model


# ### Next step is Prediction
# 
# Now will go Sigmoid functions also
# Sigmoid range is a 0 to 1

# #### Predict & Probabilty is only for Test data 

# In[168]:


lr_model.predict(x_test)
# pred fuction directly give 0 to 1 
# is not give probabilty here actualy 


# In[169]:


# use the prob
lr_model.predict_proba(x_test)
# two probabilty 0 and 1 
# i want to go with probabilty 1 


# In[170]:


# i want to go with probabilty 1 like who will purchase 
y_pred_prob=lr_model.predict_proba(x_test)[:,1]

# Compare with the 0.5 thresold value


# In[171]:


y_pred_prab


# #### If i want my won thresold point Like 0.7 let's we see but no need it just for seeing

# In[172]:


# if i want my won thresold point
y_pred_prab>0.7 # but go in medical cases
# True means is a >0.7 who will purchase the product
# False means is a <0.7 who will not purchase the product


# In[173]:


y_test # this is my actual value # Bottom one is a Pred prob value


# In[174]:


# Comper 
y_pred_prab


# In[175]:


plt.scatter(x,y)# x is a Age y is a purchase
plt.xlabel("Age")
plt.ylabel("Purchase")
plt.show()


# In[176]:


plt.scatter(x_test,y_test)# x is a Age y is a purchase
plt.scatter(x_test,y_pred_prob) # compare with actual value with pred proba value
plt.xlabel("Age")
plt.ylabel("Purchase")
plt.show()


# #### Top plot will work only a test data  Same also can work train data but we should do the pred & proba

# #### Make it Train data also as a pred and proba

# In[177]:


y_pred_train=lr_model.predict_proba(x_train)[:,1]

# a chanses of people who will purchase the product


# In[178]:


y_pred_train


# In[179]:


plt.scatter(x_train,y_train)# x is a Age y is a purchase
plt.scatter(x_train,y_pred_train) # compare with actual value with pred proba value
plt.xlabel("Age")
plt.ylabel("Purchase")
plt.show()


# #### So this is a best fit sigmoid cow`

# #### For Understanding purpose only take 1 Variable Is not posible to show multi dimensional graph in python 

# In[ ]:




