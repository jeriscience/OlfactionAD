import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
import pickle

path0='./data/'
path1='./model_single/'

# 575 by 4869
df_dragon = pd.read_csv(path0 + 'dragon_all.csv', index_col=0)
# 338 by 21
df_percept = pd.read_csv(path0 + 'percept_single.csv', index_col=0)

df0 = pd.read_csv(path0 + 'mixture_snitz1.csv', index_col=0)
df1 = pd.read_csv(path0 + 'mixture_snitz2.csv', index_col=0)
df2 = pd.read_csv(path0 + 'mixture_bushdid.csv', index_col=0)
df3 = pd.read_csv(path0 + 'mixture_ravia.csv', index_col=0)

# all single molecules
id_all = np.concatenate((np.unique(df0.to_numpy()),\
    np.unique(df1.to_numpy()),\
    np.unique(df2.to_numpy()),\
    np.unique(df3.to_numpy())))

id_all = np.unique(id_all)
id_all = id_all[~np.isnan(id_all)]
id_all = id_all[id_all>0]
id_all = id_all.astype('int')
id_all = id_all.tolist()
# 165

id_exclude = []
for the_id in id_all:
    if the_id not in df_dragon.index:
        print(the_id)
        id_exclude += [the_id]
#7284
#11173
#84682
#5284503
#5318042
#11002307

for the_id in id_exclude:
    id_all.remove(the_id)

len(id_all)
#159

X = df_dragon.loc[id_all,:].to_numpy()
X = np.nan_to_num(X)
pred_all = []
for i in range(df_percept.shape[1]):
    print(i)
    name_model = path1 + str(i) + '_' + df_percept.columns[i]
    the_model=pickle.load(open(name_model, 'rb'))
    pred = the_model.predict(X)
    pred_all += [pred]

pred_all = np.array(pred_all)
pred_all = pred_all.T

df_pred = pd.DataFrame(data=pred_all)
df_pred.columns = df_percept.columns
df_pred.index = id_all
#df_pred['CID'] = id_all
#df_pred = df_pred[['CID'] + df_percept.columns.tolist()]

df_pred.to_csv(path0 + 'pred_percept_single.csv')

#############################
## extract features for mixture

df0 = pd.read_csv(path0 + 'mixture_snitz1.csv', index_col=0)
df1 = pd.read_csv(path0 + 'mixture_snitz2.csv', index_col=0)
df2 = pd.read_csv(path0 + 'mixture_bushdid.csv', index_col=0)
df3 = pd.read_csv(path0 + 'mixture_ravia.csv', index_col=0)

the_df = df0
mixture_all=[]
for i in range(the_df.shape[0]):
    feature_all=[]
    for j in range(the_df.shape[1]):
        if the_df.iloc[i,j] > 0: #also exclude nan
            the_id = int(the_df.iloc[i,j])
            if the_id in df_pred.index:
                feature_all += [df_pred.loc[the_id,:].tolist()]
    feature_all=np.array(feature_all)
    mixture_all += [(np.mean(feature_all,axis=0)).tolist()]

mixture_all=np.array(mixture_all)
df_mixture = pd.DataFrame(data=mixture_all)
df_mixture.columns = df_percept.columns
df_mixture.index = the_df.index
df_mixture.to_csv(path0 + 'percept_mixture_snitz1.csv')

the_df = df1
mixture_all=[]
for i in range(the_df.shape[0]):
    feature_all=[]
    for j in range(the_df.shape[1]):
        if the_df.iloc[i,j] > 0: #also exclude nan
            the_id = int(the_df.iloc[i,j])
            if the_id in df_pred.index:
                feature_all += [df_pred.loc[the_id,:].tolist()]
    feature_all=np.array(feature_all)
    mixture_all += [(np.mean(feature_all,axis=0)).tolist()]

mixture_all=np.array(mixture_all)
df_mixture = pd.DataFrame(data=mixture_all)
df_mixture.columns = df_percept.columns
df_mixture.index = the_df.index
df_mixture.to_csv(path0 + 'percept_mixture_snitz2.csv')

the_df = df2
mixture_all=[]
for i in range(the_df.shape[0]):
    feature_all=[]
    for j in range(the_df.shape[1]):
        if the_df.iloc[i,j] > 0: #also exclude nan
            the_id = int(the_df.iloc[i,j])
            if the_id in df_pred.index:
                feature_all += [df_pred.loc[the_id,:].tolist()]
    feature_all=np.array(feature_all)
    mixture_all += [(np.mean(feature_all,axis=0)).tolist()]

mixture_all=np.array(mixture_all)
df_mixture = pd.DataFrame(data=mixture_all)
df_mixture.columns = df_percept.columns
df_mixture.index = the_df.index
df_mixture.to_csv(path0 + 'percept_mixture_bushdid.csv')

the_df = df3
mixture_all=[]
for i in range(the_df.shape[0]):
    feature_all=[]
    for j in range(the_df.shape[1]):
        if the_df.iloc[i,j] > 0: #also exclude nan
            the_id = int(the_df.iloc[i,j])
            if the_id in df_pred.index:
                feature_all += [df_pred.loc[the_id,:].tolist()]
    feature_all=np.array(feature_all)
    mixture_all += [(np.mean(feature_all,axis=0)).tolist()]

mixture_all=np.array(mixture_all)
df_mixture = pd.DataFrame(data=mixture_all)
df_mixture.columns = df_percept.columns
df_mixture.index = the_df.index
df_mixture.to_csv(path0 + 'percept_mixture_ravia.csv')

