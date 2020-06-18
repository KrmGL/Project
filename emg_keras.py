import pandas as pd
import numpy as np
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






 