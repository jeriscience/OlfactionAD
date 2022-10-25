import os
import sys
import numpy as np
import pandas as pd

path0='./data/old/'
path1='./data/'

df0 = pd.read_csv(path0 + 'dragon_dream.txt',delimiter='\t',header=None)

df1 = pd.read_csv(path0 + 'dragons.csv')
#add 'complexity from pubmed' using 0..
df1 = df1.iloc[:,1:]
the_name = df1.columns
df1['complexity from pubmed']=0
tmp = ['CID','complexity from pubmed'] + the_name[1:].tolist()
df1 = df1.loc[:,tmp]

df2 = pd.read_csv(path0 + 'dragondravnieksand0857.txt', delimiter='\t')

df3 = pd.read_csv(path0 + '3dragons.txt', delimiter='\t')
#add 'complexity from pubmed' using 0..
the_name = df3.columns
df3['complexity from pubmed']=0
tmp = ['CID','complexity from pubmed'] + the_name[1:].tolist()
df3 = df3.loc[:,tmp]

## rename and concatenate; sort and remove duplicates 606 -> 575 molecules
df0.columns = df1.columns
df2.columns = df1.columns
df3.columns = df1.columns
df_dragon = pd.concat([df0,df1,df2,df3])
df_dragon = df_dragon.sort_values('CID')
df_dragon = df_dragon.drop_duplicates(subset='CID', ignore_index=True)

# fill nan with 0
df_dragon.fillna(0,inplace=True)
# transform y=log(x+100)
df_dragon.iloc[:,1:] = np.log(df_dragon.iloc[:,1:] + 100)

df_dragon['CID'] = df_dragon['CID'].astype('int')
df_dragon.to_csv(path1 + 'dragon_all.csv', index=False)
## save 575 by (4869 + 1)

#df_percept = pd.read_csv(path0 + 'dravnieks.csv')
df_percept = pd.read_csv(path1 + 'percept_single.csv')
## 338 by (22+1); 
for the_id in df_percept['CID']:
    if the_id not in df_dragon['CID'].tolist():
        print(the_id)


