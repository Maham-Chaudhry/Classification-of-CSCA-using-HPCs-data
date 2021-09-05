
# coding: utf-8

# In[19]:


#######################import Libraries################
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import datasets
import statsmodels.api as sm
from scipy import stats
from operator import itemgetter
import matplotlib.patches as mpatches


# In[20]:


#################read Data Files#####################
Data = pd.read_csv('Train.csv')

Data_val_NL = pd.read_csv('Test_NL.csv')
Data_val_AL = pd.read_csv('Test_AL.csv')
Data_val_FL = pd.read_csv('Test_FL.csv')


# In[21]:


###########################Split data into attack/ No attack ####################
group=Data.groupby('Label')
Attack= group.get_group(1).values
Noattack= group.get_group(0).values
a=list(Data)

group_val_NL=Data_val_NL.groupby('Label')
Attack_NL= group_val_NL.get_group(1).values
Noattack_NL= group_val_NL.get_group(0).values
a_val_NL=list(Data_val_NL)

group_val_AL=Data_val_AL.groupby('Label')
Attack_AL= group_val_AL.get_group(1).values
Noattack_AL= group_val_AL.get_group(0).values
a_val_AL=list(Data_val_AL)

group_val_FL=Data_val_FL.groupby('Label')
Attack_FL= group_val_FL.get_group(1).values
Noattack_FL= group_val_FL.get_group(0).values
a_val_FL=list(Data_val_FL)


# In[22]:


#############################Plot data#################33
fig1 = plt.figure()
plt.hist(Attack[:,0], bins=50, alpha=0.9, facecolor='r')
plt.hist(Noattack[:,0], bins=50, alpha=0.9, facecolor='g')
plt.xlim(0,1500)
plt.xlabel(a[0])
plt.ylabel('Frequency')
plt.show()
fig1.savefig('R_Train1.png')

fig2 = plt.figure()
plt.hist(Attack[:,1], bins=50, alpha=0.9, facecolor='r')
plt.hist(Noattack[:,1], bins=50, alpha=0.9, facecolor='g')
plt.xlim(-10,600)
plt.xlabel(a[1])
plt.ylabel('Frequency')
plt.show()
fig2.savefig('R_Train2.png')

fig3 = plt.figure()
plt.hist(Attack[:,2], bins=50, alpha=0.9, facecolor='r')
plt.hist(Noattack[:,2], bins=50, alpha=0.9, facecolor='g')
plt.xlim(-10,500)
plt.xlabel(a[2])
plt.ylabel('Frequency')
plt.show()
fig3.savefig('R_Train3.png')

fig4 = plt.figure()
plt.hist(Attack[:,3], bins=50, alpha=0.7, facecolor='r')
plt.hist(Noattack[:,3], bins=50, alpha=0.7, facecolor='g')
plt.xlim(0,750000)
plt.xlabel(a[3])
plt.ylabel('Frequency')
plt.show()
fig4.savefig('R_Train4.png')



# In[23]:


#########################Plot Data#####################3
fig5 = plt.figure()
plt.hist(Attack_NL[:,0], bins=50, alpha=0.9, facecolor='r')
plt.hist(Noattack_NL[:,0], bins=50, alpha=0.9, facecolor='g')
plt.xlim(-10,500)
plt.xlabel(a_val_NL[0])
plt.ylabel('Frequency')
plt.show()
fig5.savefig('R_TstNL1.png')

fig6 = plt.figure()
plt.hist(Attack_NL[:,1], bins=50, alpha=0.9, facecolor='r')
plt.hist(Noattack_NL[:,1], bins=50, alpha=0.9, facecolor='g')
plt.xlim(-10,300)
plt.xlabel(a_val_NL[1])
plt.ylabel('Frequency')
plt.show()
fig6.savefig('R_TstNL2.png')


fig7 = plt.figure()
plt.hist(Attack_NL[:,2], bins=50, alpha=0.9, facecolor='r')
plt.hist(Noattack_NL[:,2], bins=50, alpha=0.9, facecolor='g')
plt.xlim(-10,400)
plt.xlabel(a_val_NL[2])
plt.ylabel('Frequency')
plt.show()
fig7.savefig('R_TstNL3.png')


fig8 = plt.figure()
plt.hist(Attack_NL[:,3], bins=50, alpha=0.9, facecolor='r')
plt.hist(Noattack_NL[:,3], bins=50, alpha=0.9, facecolor='g')
plt.xlim(-10,600000)
plt.xlabel(a_val_NL[3])
plt.ylabel('Frequency')
plt.show()
fig8.savefig('R_TstNL4.png')



# In[24]:


##################Plot Data###############################
fig9 = plt.figure()
plt.hist(Attack_AL[:,0], bins=50, alpha=0.9, facecolor='r')
plt.hist(Noattack_AL[:,0], bins=50, alpha=0.9, facecolor='g')
plt.xlim(-10,1500)
plt.xlabel(a_val_AL[0])
plt.ylabel('Frequency')
plt.show()
fig9.savefig('R_TstAL1.png')

fig10 = plt.figure()
plt.hist(Attack_AL[:,1], bins=50, alpha=0.9, facecolor='r')
plt.hist(Noattack_AL[:,1], bins=50, alpha=0.9, facecolor='g')
plt.xlim(-10,600)
plt.xlabel(a_val_AL[1])
plt.ylabel('Frequency')
plt.show()
fig10.savefig('R_TstAL2.png')

fig11 = plt.figure()
plt.hist(Attack_AL[:,2], bins=50, alpha=0.9, facecolor='r')
plt.hist(Noattack_AL[:,2], bins=50, alpha=0.9, facecolor='g')
plt.xlim(-10,600)
plt.xlabel(a_val_AL[2])
plt.show()
fig11.savefig('R_TstAL3.png')

fig12 = plt.figure()
plt.hist(Attack_AL[:,3], bins=50, alpha=0.9, facecolor='r')
plt.hist(Noattack_AL[:,3], bins=50, alpha=0.9, facecolor='g')
plt.xlim(-10,750000)
plt.xlabel(a_val_AL[3])
plt.ylabel('Frequency')
plt.show()
fig12.savefig('R_TstAL4.png')


# In[25]:


##############################Plot Data################################
fig13 = plt.figure()
plt.hist(Attack_FL[:,0], bins=50, alpha=0.9, facecolor='r')
plt.hist(Noattack_FL[:,0], bins=50, alpha=0.9, facecolor='g')
plt.xlim(-10,1100)
plt.xlabel(a_val_FL[0])
plt.ylabel('Frequency')
plt.show()
fig13.savefig('R_TstFL1.png')

fig14 = plt.figure()
plt.hist(Attack_FL[:,1], bins=50, alpha=0.9, facecolor='r')
plt.hist(Noattack_FL[:,1], bins=50, alpha=0.9, facecolor='g')
plt.xlim(-10,600)
plt.xlabel(a_val_FL[1])
plt.ylabel('Frequency')
plt.show()
fig14.savefig('R_TstFL2.png')


fig15 = plt.figure()
plt.hist(Attack_FL[:,2], bins=50, alpha=0.9, facecolor='r')
plt.hist(Noattack_FL[:,2], bins=50, alpha=0.9, facecolor='g')
plt.xlim(-10,600)
plt.xlabel(a_val_FL[2])
plt.ylabel('Frequency')
plt.show()
fig15.savefig('R_TstFL3.png')

fig16 = plt.figure()
plt.hist(Attack_FL[:,3], bins=50, alpha=0.8, facecolor='r')
plt.hist(Noattack_FL[:,3], bins=50, alpha=0.8, facecolor='g')
plt.xlim(-10,750000)
plt.xlabel(a_val_FL[3])
plt.ylabel('Frequency')
plt.show()
fig16.savefig('R_TstFL4.png')

