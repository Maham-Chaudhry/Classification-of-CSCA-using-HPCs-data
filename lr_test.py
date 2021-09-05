
# coding: utf-8

# In[1]:


#########################Import Libraries################
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LR
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
import matplotlib.pyplot as plt
from operator import itemgetter


# In[18]:


####################Read Data Files##################
Data = pd.read_csv('Train_Combined.csv')
Data1=np.array(Data.drop(['Label'], 1))

data_val = pd.read_csv('Test_NL.csv')
Data2=np.array(data_val.drop(['Label'], 1))


# In[19]:


#################Split data into test ,train#################
X_train=Data1
Y_train=Data.Label
X_val=Data2
Y_val=data_val.Label


# In[20]:


##################Train LR Model and find its parameters####################
logreg = LR()
model=logreg.fit(X_train, Y_train)
w=model.coef_
b=model.intercept_


# In[5]:


##############Cross Validation accuracy####################
scores = cross_val_score(model, X_train, Y_train, cv=5)
print scores                                              
print accuracy_score(Y_val,model.predict(X_val))*100


# In[6]:


################Testing parameters of LR ######################
a = np.empty(len(Data.Label));
for i in range (0,len(Data.Label)):
    x=np.asmatrix(Data1[i,:])
    xt=np.transpose(x)
    z=np.matmul(w, xt) + b
    y=1/(1+(exp(-z)))
    if (y > 0.5):
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


# In[21]:


#################Plotting ROC for LR ###################################
logit_roc_auc = roc_auc_score(Y_val, model.predict(X_val))
fpr, tpr, thresholds = roc_curve(Y_val, model.predict_proba(X_val)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='LR-NL (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('LR_ROC_NL')
plt.show()

