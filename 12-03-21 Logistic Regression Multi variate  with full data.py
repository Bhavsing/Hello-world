#!/usr/bin/env python
# coding: utf-8

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
# if Y =1 The equation should be -y(logp )== -log(p)

# In[1]:


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


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('Social_Network_Ads.csv')


# In[6]:


df.info()


# In[7]:


df.head() # top 5 rows
# Target Variable is a Purchase 
# Gender, Age, EstimatedSalary 
# This is the Classification problems 


# In[8]:


df.tail()# Bottom 5 rows


# In[91]:


df


# In[9]:


df.shape


# #### Do the Preprocessing 

# #### Handile The Missing values & Outliers

# In[13]:


df.isnull().any()
# don't have any null values in this data


# In[14]:


df.boxplot() # there is no outliers


# In[15]:


sns.boxplot(df["EstimatedSalary"])


# In[101]:


sns.scatterplot(df["Age"],df["EstimatedSalary"])
# i want to superat ethre will purchase or not use hue


# In[104]:


# use hue
sns.scatterplot(df["Age"],df["EstimatedSalary"],hue=df["Purchased"])


# #### There is no null & no Outliers is very clear data 

# In[16]:


df.head(3)


# ### Let's Do the Categorical Encoding 
# Gender Nominal or ordinal ?
# 
# Gender Is a Nominal data 
# 
# apply the One Hot Encoding to do 

# ### Encoding Categorical Data

# In[17]:


df=pd.get_dummies(df,columns=["Gender"])


# In[18]:


df


# In[19]:


df.shape


# ### Done the Preprocessing 
# 1. checked there No any null & outliers 
# 2. Gender is a Nominal data 
# 3. Converted To Numaricl use by One Hot Encoding Categorical data 
# 

# # Split the data Independent & Dependend

# In[21]:


x=df.iloc[:,[1,2,4,5]]
x


# In[23]:


y=df.iloc[:,3]
y


# ### Splited the data as a X , Y
# 1. X Independent Variable is a Age, EstimatedSalary, Gender_Female,	Gender_Male
# 2. y Dependent is a Purchase

# #### Split the Train And Test  

# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)


# In[26]:


x_train.shape


# In[30]:


y_test.shape


# ### Splited  data as a x_train, x_test, y_train, y_test
# 1. is Nothing but making into two deferent dataset 
# part for better understanding algorithm as 80,20 
# 

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

# In[32]:


# Min Max Scaler formula -- xscaled = x-xmin/(xmax - xmain
# 0 to 1 min max range, Min Max Scaler formula
from sklearn.preprocessing import MinMaxScaler


# In[33]:


sc=MinMaxScaler()


# In[34]:


x_train=sc.fit_transform(x_train)


# In[45]:


x_train # converted to 0 to 1 range


# In[46]:


x_test=sc.transform(x_test)


# # Next step Building the Model
# logistic regression is a linear regression base model

# In[47]:


from sklearn.linear_model import LogisticRegression


# In[48]:


lr_model=LogisticRegression()


# In[49]:


lr_model.fit(x_train, y_train) # fit & train the model


# ### Next step is Prediction
# 
# Now will go Sigmoid functions also
# Sigmoid range is a 0 to 1

# #### Predict & Probabilty is only for Test data 

# In[75]:


y_pred=lr_model.predict(x_test)
# pred fuction directly give 0 to 1 
# is not give probabilty here actualy 


# In[76]:


y_pred


# In[77]:


# use the prob
y_pred_prob=lr_model.predict_proba(x_test)
# two probabilty 0 and 1 
# i want to go with probabilty 1 


# In[78]:


y_pred_prob


# In[79]:


# i want to go with probabilty 1 like who will purchase 
y_pred_prob=lr_model.predict_proba(x_test)[:,1]

# Compare with the 0.5 thresold value


# In[80]:


y_pred_prob


# #### If i want my won thresold point Like 0.7 let's we see but no need it just for seeing

# In[67]:


# if i want my won thresold point
y_pred_prob>0.7 # but go in medical cases
# True means is a >0.7 who will purchase the product
# False means is a <0.7 who will not purchase the product


# In[68]:


y_test # this is my actual value # Bottom one is a Pred prob value


# In[96]:


# Comper 
y_pred_prob


# ### Still Now Done the Prediction

# ### Next will go with Evaluation
# How will do to find the Accuracy

# ### Accuracy 
# 1. accuracy is only work for Classification type of problem, 
# 2. Will not work for Regression problem
# 
# what is accuracy score 0 1
# 1. formula percentage of corect prediction
# 

# In[109]:


y_pred


# In[110]:


from sklearn.metrics import accuracy_score


# In[111]:


accuracy_score(y_test, y_pred)
#accuracy_score(y_test, y_pred)*100


# #### what is tresold is indicate best is a 0.5

# ### Next step confusion_matrix
# 
# #### what is the next metrics to use for 
# Confution metrics

# In[114]:


from sklearn.metrics import confusion_matrix


# In[115]:


confusion_matrix(y_test,y_pred)
[52,  0] # 51 true -ve,  0 false +ve,
[28,  0] # 28 false -ve,  0 True +ve,


# # Multiple Logistic Regression
# followed by 
# 
# 1. Handling Missing values & Outliers use linear graph to find linear or non linear
# 2. Gender Nominal Categorical Converted into Numaric apply the One Hot Encoding to do
# 3. Splited the data as a X Independent Variable, y is a Dependent Variable
# 4. Splited data as a x_train, x_test, y_train, y_test making into two deferent dataset
# 5. Building the logistic regression model is a linear regression base model
# 6. Predicted Sigmoid functions also Sigmoid range is a 0 to 1
# 7. Evaluation for finding of the Accuracy,What is accuracy score 0 1 formula percentage of corect prediction
# 8. confusion_matrix

# #### Let's do Some realtime predictio ether person will purchase or not the product

# In[116]:


age=25
salary=80000
gender="Male"

# creating some real time data


# In[117]:


df=np.array([[age,salary,0,1]]) # convering to array


# In[118]:


df


# In[119]:


df=sc.transform(df) # applying scaling


# In[121]:


lr_model.predict_proba(df)[:,1] # check the prob is the chance have or not to purchase


# In[ ]:




