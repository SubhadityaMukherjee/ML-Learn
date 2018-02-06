# %% PREPROCESSING

import numpy as np
from sklearn import preprocessing

ind = np.array([[1.0,2.0,3.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
print(ind,'\n')

#N0rmalize
dp = preprocessing.normalize(ind,norm='l1') #l1 for least abs dev and l2 for least squares
print(dp,'\n')

#binarization
db = preprocessing.Binarizer(threshold = 2).transform(ind) #all data above 2 becomes 0
print(db,'\n')

# mean removal
print("Mean: ",ind.mean(axis=0))
print("Std: ",ind.std(axis = 0))
ds = preprocessing.scale(ind)
print("New Mean: ",ds.mean(axis=0))
print("New Std: ",ds.std(axis = 0))
#Min max scaling
dm = preprocessing.MinMaxScaler(feature_range = (0,1)).fit_transform(ind) #scales all between 0 and 1
print(dm,'\n')

# %% LABLE ENCODING
import numpy as np
from sklearn import preprocessing

#sample input labels
il = ['red','black','blue','green','red','red','purple']
le = preprocessing.LabelEncoder().fit(il)
for i,item in enumerate(le.classes_):
    print(item, i)

# %% plots (matplotlib)

import numpy as np
import matplotlib.pyplot as plt

#normal plot
p = np.array([1,2,3])
plt.plot(p)
plt.show()

#scatter plot
p = np.array([1,2,3])
q= np.array([10,50,30])
plt.scatter(p,q)
plt.show()

# %% pandas
import numpy as np
import pandas as pd

#series
p = np.array([1,2,3])
rownames =['a','b','c']
ser = pd.Series(p,index =rownames)
print(ser)
print(ser[0])
print(ser['c'],'\n')

#Data frame
p = np.array([[1,2,3],[10,50,30]])
col = ['a','b','c']
row = ['ra','rb']
df = pd.DataFrame(p,index =row,columns = col)
print(df,'\n')

# %% csv with numpy
from numpy import loadtxt
import os
os.chdir(os.getcwd())
f = open('numb.csv','r')
data = loadtxt(f,delimiter = ",")
print(data.shape)

# %% csv with pandas
from pandas import read_csv as rc
f = 'text.csv' #or url
names = ['head','tail','row']
d = rc(f,names=names)
print(data.shape)

# %% view data/aspects
from pandas import read_csv as rc
import numpy as np
f = 'numb.csv' #or url
names = ['head','tail','row']
d = rc(f,names=names)

#display part
print(d.head(5),'\n')

#dimensions
print(data.shape,'\n')

#datatypes
print(d.dtypes,'\n')

#overall stats
ds = d.describe()
print(ds,'\n')

#class distribution
cc = d.groupby('head').size()
print(cc,'\n')

#pearson correlation
from pandas import set_option as so
so('display.width',100)
so('precision',3)
cor = d.corr(method = 'pearson')
print(cor,'\n')

#skew
print(d.skew())

# %% PLOTS

from matplotlib import pyplot as pp
from pandas import read_csv as rc

f = 'numb.csv'
names = ['head','tail','row']
d = rc(f,names=names)

#histogram
d.hist()
pp.show()

#density plots
d.plot(kind ='density',subplots = True,layout =(3,3),sharex=False,sharey = False)
pp.show()

#box plots
d.plot(kind ='box',subplots = True,layout =(3,3),sharex=False,sharey = False)
pp.show()

#correlation matrix plot
'''
Used to plot correlation. If two variables change int he same direc then positive correlation.
some models like linear and logistic reg. have poor performnace if variables are highly correlated
'''
corre = d.corr()

fig = pp.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corre,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,3,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pp.show()
