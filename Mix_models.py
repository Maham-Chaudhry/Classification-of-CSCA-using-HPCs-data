
# coding: utf-8

# In[1]:


#################################import Libraries##################################
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
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
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model


# In[2]:


#####################Read Data Files############################
Data = pd.read_csv('Train.csv')

Data_val_NL = pd.read_csv('Test_NL.csv')
Data_val_AL = pd.read_csv('Test_AL.csv')
Data_val_FL = pd.read_csv('Test_FL.csv')


# In[3]:


########################### Data seperation into output and input form####################
X_train=np.array(Data.drop(['Label'], 1))
Y_train=Data.Label

X_val_NL = Data_val_NL.drop(['Label'],1)
Y_val_NL = Data_val_NL.Label

X_val_AL = Data_val_AL.drop(['Label'],1)
Y_val_AL = Data_val_AL.Label
                            
X_val_FL = Data_val_FL.drop(['Label'],1)
Y_val_FL = Data_val_FL.Label


# In[4]:


#########LDA Model Training and testing#############
LdA = LDA(store_covariance=True)
lDa=LdA.fit(X_train, Y_train)
print cross_val_score(lDa, X_train, Y_train, cv=5)
print accuracy_score(Y_val_NL,lDa.predict(X_val_NL))*100
print accuracy_score(Y_val_AL,lDa.predict(X_val_AL))*100
print accuracy_score(Y_val_FL,lDa.predict(X_val_FL))*100


# In[5]:


#########LR Model Training and testing#############
lR = LR()
Lr=lR.fit(X_train, Y_train)
print cross_val_score(Lr, X_train, Y_train, cv=5)
print accuracy_score(Y_val_NL,Lr.predict(X_val_NL))*100
print accuracy_score(Y_val_AL,Lr.predict(X_val_AL))*100
print accuracy_score(Y_val_FL,Lr.predict(X_val_FL))*100


# In[6]:


#########QDA Model Training and testing#############
QD=QuadraticDiscriminantAnalysis()
QDM = QD.fit(X_train, Y_train)
print cross_val_score(QDM, X_train, Y_train, cv=5)
print accuracy_score(Y_val_NL,QDM.predict(X_val_NL))*100
print accuracy_score(Y_val_AL,QDM.predict(X_val_AL))*100
print accuracy_score(Y_val_FL,QDM.predict(X_val_FL))*100


# In[7]:


#########SVC Model Training and testing#############
SvC = LinearSVC(random_state=1)
sVc= SvC.fit(X_train, Y_train)
print cross_val_score(sVc, X_train, Y_train, cv=5)
print accuracy_score(Y_val_NL,sVc.predict(X_val_NL))*100
print accuracy_score(Y_val_AL,sVc.predict(X_val_AL))*100
print accuracy_score(Y_val_FL,sVc.predict(X_val_FL))*100 


# In[8]:


#########Nearest centroid Model Training and testing#############
NC = NearestCentroid()
NCM=NC.fit(X_train, Y_train)
print cross_val_score(NCM, X_train, Y_train, cv=5)
print accuracy_score(Y_val_NL,NCM.predict(X_val_NL))*100
print accuracy_score(Y_val_AL,NCM.predict(X_val_AL))*100
print accuracy_score(Y_val_FL,NCM.predict(X_val_FL))*100                                              


# In[9]:


#########guassian Model Training and testing#############
gnb = GaussianNB()
GNBM = gnb.fit(X_train, Y_train)
print cross_val_score(GNBM, X_train, Y_train, cv=5)
print accuracy_score(Y_val_NL,GNBM.predict(X_val_NL))*100
print accuracy_score(Y_val_AL,GNBM.predict(X_val_AL))*100
print accuracy_score(Y_val_FL,GNBM.predict(X_val_FL))*100 


# In[10]:


#########KNN Model Training and testing#############
KNN = KNeighborsClassifier(n_neighbors=4)
KNNM = KNN.fit(X_train, Y_train)
print cross_val_score(KNNM, X_train, Y_train, cv=5)
print accuracy_score(Y_val_NL,KNNM.predict(X_val_NL))*100
print accuracy_score(Y_val_AL,KNNM.predict(X_val_AL))*100
print accuracy_score(Y_val_FL,KNNM.predict(X_val_FL))*100 


# In[11]:


#########Dummy Classifier Model Training and testing#############
DC=DummyClassifier()
DCM = DC.fit(X_train, Y_train)
print cross_val_score(DCM, X_train, Y_train, cv=5)
print accuracy_score(Y_val_NL,DCM.predict(X_val_NL))*100
print accuracy_score(Y_val_AL,DCM.predict(X_val_AL))*100
print accuracy_score(Y_val_FL,DCM.predict(X_val_FL))*100 


# In[12]:


#########Perceptron Model Training and testing#############
PE=Perceptron()
PEM = PE.fit(X_train, Y_train)
print cross_val_score(PEM, X_train, Y_train, cv=5)
print accuracy_score(Y_val_NL,PEM.predict(X_val_NL))*100
print accuracy_score(Y_val_AL,PEM.predict(X_val_AL))*100
print accuracy_score(Y_val_FL,PEM.predict(X_val_FL))*100 


# In[13]:


#########Decision trees Model Training and testing#############
DT = tree.DecisionTreeClassifier(max_depth=4,random_state=0)
DTM = DT.fit(X_train, Y_train)
print cross_val_score(DTM, X_train, Y_train, cv=5)
print accuracy_score(Y_val_NL,DTM.predict(X_val_NL))*100
print accuracy_score(Y_val_AL,DTM.predict(X_val_AL))*100
print accuracy_score(Y_val_FL,DTM.predict(X_val_FL))*100 


# In[14]:


#########Random Forest Model Training and testing#############
RF = RandomForestClassifier(n_jobs=2,random_state=0)
RFM = RF.fit(X_train, Y_train)
print cross_val_score(RFM, X_train, Y_train, cv=5)
print accuracy_score(Y_val_NL,RFM.predict(X_val_NL))*100
print accuracy_score(Y_val_AL,RFM.predict(X_val_AL))*100
print accuracy_score(Y_val_FL,RFM.predict(X_val_FL))*100 

