
# coding: utf-8

# In[26]:


#########################Import Libraries################
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
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
from math import log
import matplotlib.pyplot as plt
from operator import itemgetter
import matplotlib.patches as mpatches


# In[57]:


####################Read Data Files##################
Data = pd.read_csv('Train_Combined.csv')
Data1=np.array(Data.drop(['Label'], 1))

data_val = pd.read_csv('STATS_QDA_Det_Total.csv')
Data2=np.array(data_val.drop(['Label'], 1))


# In[58]:


#################Split data into test ,train#################
X_train=Data1
Y_train=Data.Label
X_val=Data2
Y_val=data_val.Label


# In[59]:


##################Train Model####################
logreg = QDA(store_covariance=True)
model=logreg.fit(X_train, Y_train)


# In[60]:


##############Cross Validation accuracy####################
scores = cross_val_score(logreg, X_train, Y_train, cv=5)
print scores                                              
print accuracy_score(Y_val,model.predict(X_val))*100


# In[50]:


##################Train LR Model and find its parameters####################
c= model.covariances_
c0=inv(np.asmatrix(c[0]))
c1=inv(np.asmatrix(c[1]))
u= model.means_
u0= np.asmatrix(u[0])
u1= np.asmatrix(u[1])
p=model.priors_
p0=log1p(p[0]-1)
p1=log1p(p[1]-1)
v0=-0.5*log(np.linalg.det(c[0]))
v1=-0.5*log(np.linalg.det(c[1]))
w0=-0.5*u0*c0*np.transpose(u0)
w1=-0.5*u1*c1*np.transpose(u1)


# In[45]:


################Testing parameters of LR ######################
a = np.empty(len(Data.Label));
for i in range (0,len(Data.Label)):
    x=np.asmatrix(Data1[i,:])
    xt=np.transpose(x)
    y1=(-0.5*x*c1*xt)+(x*c1*np.transpose(u1))+(w1+v1+p1)
    y0=(-0.5*x*c0*xt)+(x*c0*np.transpose(u0))+w0+v0+p0
    if (y1 > y0):
        a[i]='1'
    else: 
        a[i]='0'


# In[34]:


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


# In[41]:


#################Plotting ROC for QDA ###################################
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

