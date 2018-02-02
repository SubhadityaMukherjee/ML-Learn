# %% PREPROCESSING

import numpy as np
from sklearn import preprocessing

ind = np.array([[1.0,2.0,3.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
print(ind)

#N0rmalize
dp = preprocessing.normalize(ind,norm='l1') #l1 for least abs dev and l2 for least squares
print(dp)

#binarization
db = preprocessing.Binarizer(threshold = 2).transform(ind) #all data above 2 becomes 0
print(db)

# mean removal
print("Mean: ",ind.mean(axis=0))
print("Std: ",ind.std(axis = 0))
ds = preprocessing.scale(ind)
print("New Mean: ",ds.mean(axis=0))
print("New Std: ",ds.std(axis = 0))
#Min max scaling
dm = preprocessing.MinMaxScaler(feature_range = (0,1)).fit_transform(ind) #scales all between 0 and 1
print(dm)

# %% LABLE ENCODING
import numpy as np
from sklearn import preprocessing

#sample input labels
il = ['red','black','blue','green','red','red','purple']
le = preprocessing.LabelEncoder().fit(il)
for i,item in enumerate(le.classes_):
    print(item, i)
