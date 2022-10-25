import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
import pickle

path0='./data/'

df_dragon = pd.read_csv(path0 + 'dragon_all.csv', index_col=0)
# 575 by 4869
df_percept = pd.read_csv(path0 + 'percept_single.csv', index_col=0)
# 338 by 21

id_all = []
for the_id in df_percept.index.tolist():
    if the_id in df_dragon.index:
        id_all += [the_id]

id_all = np.array(id_all)
len(id_all)
#338

## feature & gt
df_feature = df_dragon.loc[id_all.tolist(),:]
df_gt = df_percept.loc[id_all.tolist(),:]

### train-test partition
#np.random.seed(0)
#np.random.shuffle(id_all)
#ratio=[0.9,0.1]
#
#num1 = int(len(id_all)*ratio[0])
#id_train = id_all[:num1]
#id_test = id_all[num1:]
#
#id_train.sort()
#id_test.sort()

## train lasso for each percept
path1 = './model_single/'
os.system('mkdir -p ' + path1)
model_all = []
for i in range(df_gt.shape[1]):
    print(df_gt.columns[i])
    X = df_feature.to_numpy()
    X = np.nan_to_num(X)
    y = df_gt.to_numpy()[:,i]
    #the_model = LassoCV(cv=10, random_state=0).fit(X, y)
    the_model = ElasticNetCV(cv=10, random_state=0, l1_ratio=[0.001, 0.1, 0.5, 0.95, 1], n_alphas=20).fit(X, y)
    name_model = path1 + str(i) + '_' + df_gt.columns[i]
    pickle.dump(the_model, open(name_model, 'wb'))
    #the_model=pickle.load(open(name_model, 'rb'))

## elastic net - lasso/ridge
#a * ||w||_1 + 0.5 * b * ||w||_2^2
#where:
#alpha = a + b and l1_ratio = a / (a + b)

# matlab version
#if i==1
#    [Coeff,Info]=lasso(Training_set,K(:,i),'CV',10,'DFmax',5000,'Alpha',0.00001);
#elseif i==2
#    [Coeff,Info]=lasso(Training_set,K(:,i),'CV',10,'DFmax',500,'Alpha',1);
#else
#    [Coeff,Info]=lasso(Training_set,K(:,i),'CV',10,'DFmax',500,'Alpha',0.95);



