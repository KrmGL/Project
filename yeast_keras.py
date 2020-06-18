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
# %%
# keras

from keras.models import Sequential
from keras.layers import Dense,Activation

#%%

model=Sequential()
model.add(Dense(16,input_dim=8))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(optimizer="adam",loss="binary_crossentropy",metris=["accuracy"])

egitim=model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))


# %%
plt.plot(egitim.history['loss'])
plt.plot(egitim.history['val_loss'])
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('loss values')
plt.legend(['loss','val_loss'],loc='lower right')
plt.show()

# %% cm
import sklearn.metrics as metrics
y_pred=model.predict_classes(x_test)
# %%
print("acc:",metrics.accuracy_score(y_test,y_pred))
# %%
print("cm:",metrics.confusion_matrix(y_test,y_pred))

# %%
print("f1:",metrics.f1_score(y_test,y_pred))

# %%
print(metrics.classification_report(y_test,y_pred))
# %% roc ve auc
probs=model.predict_proba(x_test)
fpr,tpr,threshold=metrics.roc_curve(y_test,probs)
roc_auc=metrics.auc(fpr,tpr)

plt.title("ROC")
plt.plot(fpr,tpr,label="AUC=%0.2f" %roc_auc)
plt.legend(loc="lower right")
plt.ylabel("TPR")
plt.xlabel("FPR")
plt.show()






