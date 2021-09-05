
# coding: utf-8

# In[43]:


#########################Import Libraries################
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
import matplotlib.pyplot as plt
from operator import itemgetter
import matplotlib.patches as mpatches


# In[44]:


####################Read Data Files##################
Data = pd.read_csv('Train.csv')
Data1=np.array(Data.drop(['Label'], 1))

data_val = pd.read_csv('Test_NL.csv')
Data2=np.array(data_val.drop(['Label'], 1))


# In[45]:


#################Split data into test ,train#################
X_train=Data1
Y_train=Data.Label
X_val=Data2
Y_val=data_val.Label


# In[46]:


##################Train Model####################
logreg = LDA(store_covariance=True)
model=logreg.fit(X_train, Y_train)


# In[47]:


##############Cross Validation accuracy####################
scores = cross_val_score(logreg, X_train, Y_train, cv=5)
print scores                                              
print accuracy_score(Y_val,model.predict(X_val))*100


# In[48]:


######################################finding parameters of LDA################
c= inv(np.asmatrix(model.covariance_))
u= model.means_
u0= np.asmatrix(u[0,:])
u1= np.asmatrix(u[1,:])
p=model.priors_
p0=log1p(p[0]-1)
p1=log1p(p[1]-1)
a0=np.matmul(u0, c)
a1=np.matmul(u1, c)
u0t= np.transpose(u0)
u1t= np.transpose(u1)
b0=np.divide(u0t,2)
b1=np.divide(u1t,2)
c0=np.matmul(a0, b0)
c1=np.matmul(a1, b1)
d0 = np.subtract(p0,c0)
d1 = np.subtract(p1,c1)


# In[50]:


################Testing parameters of LDA ######################
a = np.empty(len(Data.Label));
for i in range (0,len(Data.Label)):
    x=np.asmatrix(Data1[i,:])
    xt=np.transpose(x)
    y1=np.matmul(a1, xt)
    y0=np.matmul(a0, xt)
    f1=np.add(y1,d1)
    f0=np.add(y0,d0)
    if (f1 > f0):
        a[i]='1'
    else: 
        a[i]='0'


# In[51]:


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


# In[22]:


#################Plotting ROC for LDA ###################################
logit_roc_auc = roc_auc_score(Y_val, logreg.predict(X_val))
fpr, tpr, thresholds = roc_curve(Y_val, logreg.predict_proba(X_val)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='LDA-NL (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('LDA_ROC_NL')
plt.show()

