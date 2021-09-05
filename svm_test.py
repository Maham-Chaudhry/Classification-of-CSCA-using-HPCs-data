
# coding: utf-8

# In[1]:


#########################Import Libraries################
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import datasets
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from numpy.linalg import inv
from math import log1p
from math import exp
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from operator import itemgetter


# In[18]:


####################Read Data Files##################
Data = pd.read_csv('Train_Combined_SVM.csv')
Data1=np.array(Data.drop(['Label'], 1))

data_val = pd.read_csv('Test_FL.csv')
Data2=np.array(data_val.drop(['Label'], 1))


# In[19]:


#################Split data into test ,train#################
X_train=Data1
Y_train=Data.Label
X_val=Data2
Y_val=data_val.Label


# In[4]:


##################Train Model and finding its parameters####################
logreg = LinearSVC(random_state=1)
model=logreg.fit(X_train, Y_train)
w=model.coef_
b=model.intercept_
print w
print b


# In[20]:


##############Cross Validation accuracy####################
scores = cross_val_score(logreg, X_train, Y_train, cv=5)
print scores                                              
print accuracy_score(Y_val,model.predict(X_val))*100


# In[6]:


################Testing parameters of SVM ######################
a = np.empty(len(Data.Label));
for i in range (0,len(Data.Label)):
    x=np.asmatrix(Data1[i,:])
    xt=np.transpose(x)
    z=(np.matmul(w, xt)) + b
    if (z > 0):
        a[i]='1'
    else: 
        a[i]='0'


# In[7]:


#####################finding the accuracy for extracted parameters ############
count=0
b1=[]
per=0
for i in range (0,len(Data.Label)):
    if a[i] == Data.Label[i]:
        count=count+1
    else: 
        b1.append(i)
per= (count*100/i)
per

