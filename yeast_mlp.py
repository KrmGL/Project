# %% import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("yeast.csv")

print(df.info())

df.drop(["SWISS-PROT"],axis=1,inplace = True)  #işimize yaramayan sütunu kaldırdık. 
#%%
#MIT:0    NUC:1  CYT:2  ME1:3  ME2:4
#MEC3:5   EXC:6  VAC:7  ERL:8  POX:9
#  local:localization(bulunma konumu)
#  CYT (cytosolic or cytoskeletal)                    
#  NUC (nuclear) (nükleer)                                     
#  MIT (mitochondrial)(mitokondriyal)                                
#  ME3 (membrane protein, no N-terminal signal)(membran proteini, N-terminal sinyali yok)      
#  ME2 (membrane protein, uncleaved signal)(membran proteini, temizlenmemiş sinyal)          
#  ME1 (membrane protein, cleaved signal)(membran proteini, bölünmüş sinyal)              
#  EXC (extracellular)(hücre dışı)                                 
#  VAC (vacuolar)(vakuolar)                                      
#  POX (peroxisomal) (peroksizomal)                                  
#  ERL (endoplasmic reticulum lumen)(endoplazmik retikulum lümeni)                    

df["local"]=[ 0 if each=="MIT" else each for each in df.local]
df["local"]=[ 1 if each=="NUC" else each for each in df.local]
df["local"]=[ 2 if each=="CYT" else each for each in df.local]
df["local"]=[ 3 if each=="ME1" else each for each in df.local]
df["local"]=[ 4 if each=="ME2" else each for each in df.local]
df["local"]=[ 5 if each=="ME3" else each for each in df.local]
df["local"]=[ 6 if each=="EXC" else each for each in df.local]
df["local"]=[ 7 if each=="VAC" else each for each in df.local]
df["local"]=[ 8 if each=="ERL" else each for each in df.local]
df["local"]=[ 9 if each=="POX" else each for each in df.local]


#%%

y=df.local.values
x_data=df.drop(["local"],axis=1)
# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x_data,y,test_size=0.3,random_state=1)

#%%
from sklearn.neural_network import MLPClassifier

mlpc_model=MLPClassifier()
mlpc_model.fit(x_train,y_train)

# %%
print("score:",mlpc_model.score(x_test,y_test))
# %% model tuning

from sklearn.model_selection import GridSearchCV

mlpc_params={"alpha":[1,0.1,0.01,0.005],
             "hidden_layer_sizes":[(10,10),(100,100,100),(3,5)]}
# mlpc=MLPClassifier() #ilk çalışan
mlpc=MLPClassifier(activation="logistic",solver="lbfgs") #2. çalışan
mlpc_cv_model=GridSearchCV(mlpc,mlpc_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)

# %%
print(mlpc_cv_model.best_params_)
# %%

mlpc_tuned=MLPClassifier(alpha=0.1,hidden_layer_sizes=(3,5),activation="logistic",solver="lbfgs").fit(x_train,y_train)
# %%
print("score:",mlpc_tuned.score(x_test,y_test))




