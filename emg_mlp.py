# %% import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("emg_fingers1.csv")

#veri setini el üzerine yerleştirilen 8 kanallı emg ölçümlerinden sırasıyla;
#baş-işaret-orta-yüzük-küçük parmaklardan alınan veriler ile oluşturuldu.
#baş:0-işaret:1-orta:2-yüzük:3-küçük:4
#her parmak için 500 veri vardır.

print(df.info()) 
# %%
y=df.finger.values
x_data=df.drop(["finger"],axis=1)

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


