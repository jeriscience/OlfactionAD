import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
import pickle

np.set_printoptions(precision=3,suppress=True)

def rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean()**0.5

path0='./data/'
path1 = './model_mixture_max/'
os.system('mkdir -p ' + path1)
path2 = './score_max/'
os.system('mkdir -p ' + path2)

## 0. bushdid 
df_gt = pd.read_csv(path0 + 'gt_bushdid.csv')
df_percept = pd.read_csv(path0 + 'percept_mixture_max_bushdid.csv', index_col=0)

## prepare feature (xi - yi)**2
for i in range(df_gt.shape[0]):
    x = df_percept.loc[df_gt.iloc[i,0]]
    y = df_percept.loc[df_gt.iloc[i,1]]
    z = (x-y)**2
    if i == 0:
        df_feature = z.copy()
    else:
        df_feature = pd.concat([df_feature,z],axis=1)

df_feature = df_feature.T
# reset index
df_feature.reset_index(drop=True,inplace=True)

## train/vali-test partition
num_cv=10
num_sample = df_gt.shape[0]
num_test = int(np.ceil(num_sample/num_cv))
id_all = np.arange(num_sample).tolist()
np.random.seed(0)
np.random.shuffle(id_all)

pear_all=[]
rmse_all=[]
for i in range(num_cv):
    start = i * num_test
    end = min((i+1) * num_test, num_sample)
    print(i,start,end)
    the_test = id_all[start:end]
    ## train
    the_train = []
    for the_id in id_all:
        if the_id not in the_test:
            the_train.append(the_id)
    y = df_gt.loc[the_train,:].iloc[:,3].to_numpy()
    X = df_feature.loc[the_train,:].to_numpy()
    the_model = LassoCV(cv=10, random_state=0).fit(X, y)
    name_model = path1 + 'lasso_' + str(i)
    pickle.dump(the_model, open(name_model, 'wb'))
    ## test
    gt = df_gt.loc[the_test,:].iloc[:,3].to_numpy()
    X = df_feature.loc[the_test,:].to_numpy()
    pred = the_model.predict(X)
    pear_all.append(np.corrcoef(gt,pred)[0,1])
    rmse_all.append(rmse(gt,pred))

pear_all = np.array(pear_all)
rmse_all = np.array(rmse_all)

out = np.concatenate((pear_all,np.mean(pear_all).reshape(1,)))
np.savetxt(path2 + 'score_max_bushdid_pear.txt',out,fmt='%.3f')
out = np.concatenate((rmse_all,np.mean(rmse_all).reshape(1,)))
np.savetxt(path2 + 'score_max_bushdid_rmse.txt',out,fmt='%.3f')

#######################

## 1. snitz1
df_gt = pd.read_csv(path0 + 'gt_snitz1.csv')
df_percept = pd.read_csv(path0 + 'percept_mixture_max_snitz1.csv', index_col=0)

## prepare feature (xi - yi)**2
for i in range(df_gt.shape[0]):
    x = df_percept.loc[df_gt.iloc[i,0]]
    y = df_percept.loc[df_gt.iloc[i,1]]
    z = (x-y)**2
    if i == 0:
        df_feature = z.copy()
    else:
        df_feature = pd.concat([df_feature,z],axis=1)

df_feature = df_feature.T
# reset index
df_feature.reset_index(drop=True,inplace=True)

## train/vali-test partition
num_cv=10
num_sample = df_gt.shape[0]
num_test = int(np.ceil(num_sample/num_cv))
id_all = np.arange(num_sample).tolist()
np.random.seed(0)
np.random.shuffle(id_all)

pear_all=[]
rmse_all=[]
for i in range(num_cv):
    start = i * num_test
    end = min((i+1) * num_test, num_sample)
    print(i,start,end)
    the_test = id_all[start:end]
    name_model = path1 + 'lasso_' + str(i)
    the_model=pickle.load(open(name_model, 'rb'))
    ## test
    gt = df_gt.loc[the_test,:].iloc[:,3].to_numpy()
    X = df_feature.loc[the_test,:].to_numpy()
    pred = the_model.predict(X)
    # normalize to 0-1
    pred = pred / max(pred)
    pear_all.append(np.corrcoef(gt,pred)[0,1])
    rmse_all.append(rmse(gt,pred))

pear_all = np.array(pear_all)
rmse_all = np.array(rmse_all)

out = np.concatenate((pear_all,np.mean(pear_all).reshape(1,)))
np.savetxt(path2 + 'score_max_snitz1_pear.txt',out,fmt='%.3f')
out = np.concatenate((rmse_all,np.mean(rmse_all).reshape(1,)))
np.savetxt(path2 + 'score_max_snitz1_rmse.txt',out,fmt='%.3f')

#######################

## 2. snitz2
df_gt = pd.read_csv(path0 + 'gt_snitz2.csv')
df_percept = pd.read_csv(path0 + 'percept_mixture_max_snitz2.csv', index_col=0)

## prepare feature (xi - yi)**2
for i in range(df_gt.shape[0]):
    x = df_percept.loc[df_gt.iloc[i,0]]
    y = df_percept.loc[df_gt.iloc[i,1]]
    z = (x-y)**2
    if i == 0:
        df_feature = z.copy()
    else:
        df_feature = pd.concat([df_feature,z],axis=1)

df_feature = df_feature.T
# reset index
df_feature.reset_index(drop=True,inplace=True)

## train/vali-test partition
num_cv=10
num_sample = df_gt.shape[0]
#num_test = int(np.ceil(num_sample/num_cv))
num_test = int(np.ceil(num_sample/num_cv)) - 1
id_all = np.arange(num_sample).tolist()
np.random.seed(100)
np.random.shuffle(id_all)

pear_all=[]
rmse_all=[]
for i in range(num_cv):
    start = i * num_test
    end = min((i+1) * num_test, num_sample)
    print(i,start,end)
    the_test = id_all[start:end]
    name_model = path1 + 'lasso_' + str(i)
    the_model=pickle.load(open(name_model, 'rb'))
    ## test
    gt = df_gt.loc[the_test,:].iloc[:,3].to_numpy()
    X = df_feature.loc[the_test,:].to_numpy()
    pred = the_model.predict(X)
    pear_all.append(np.corrcoef(gt,pred)[0,1])
    rmse_all.append(rmse(gt,pred))

pear_all = np.array(pear_all)
rmse_all = np.array(rmse_all)

out = np.concatenate((pear_all,np.mean(pear_all).reshape(1,)))
np.savetxt(path2 + 'score_max_snitz2_pear.txt',out,fmt='%.3f')
out = np.concatenate((rmse_all,np.mean(rmse_all).reshape(1,)))
np.savetxt(path2 + 'score_max_snitz2_rmse.txt',out,fmt='%.3f')

#######################

## 3. ravia
df_gt = pd.read_csv(path0 + 'gt_ravia.csv')
df_percept = pd.read_csv(path0 + 'percept_mixture_max_ravia.csv', index_col=0)

## prepare feature (xi - yi)**2
for i in range(df_gt.shape[0]):
    x = df_percept.loc[df_gt.iloc[i,0]]
    y = df_percept.loc[df_gt.iloc[i,1]]
    z = (x-y)**2
    if i == 0:
        df_feature = z.copy()
    else:
        df_feature = pd.concat([df_feature,z],axis=1)

df_feature = df_feature.T 
# reset index
df_feature.reset_index(drop=True,inplace=True)

## train/vali-test partition
num_cv=10
num_sample = df_gt.shape[0]
num_test = int(np.ceil(num_sample/num_cv))
id_all = np.arange(num_sample).tolist()
np.random.seed(0)
np.random.shuffle(id_all)

pear_all=[]
rmse_all=[]
for i in range(num_cv):  
    start = i * num_test
    end = min((i+1) * num_test, num_sample)
    print(i,start,end)
    the_test = id_all[start:end]
    name_model = path1 + 'lasso_' + str(i)
    the_model=pickle.load(open(name_model, 'rb'))
    ## test
    gt = df_gt.loc[the_test,:].iloc[:,3].to_numpy()
    X = df_feature.loc[the_test,:].to_numpy()
    pred = the_model.predict(X)
    pear_all.append(np.corrcoef(gt,pred)[0,1])
    rmse_all.append(rmse(gt,pred))

pear_all = np.array(pear_all)
rmse_all = np.array(rmse_all)

out = np.concatenate((pear_all,np.mean(pear_all).reshape(1,)))
np.savetxt(path2 + 'score_max_ravia_pear.txt',out,fmt='%.3f')
out = np.concatenate((rmse_all,np.mean(rmse_all).reshape(1,)))
np.savetxt(path2 + 'score_max_ravia_rmse.txt',out,fmt='%.3f')


