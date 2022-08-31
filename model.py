import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


dataset=pd.read_csv("US COVID.csv")
dataset.drop(["date","state"],axis=1,inplace=True)
X=dataset.iloc[:,0:-1].values
y=dataset.iloc[:,-1].values
y=y.reshape(len(y),1)

from sklearn.impute import SimpleImputer
impute=SimpleImputer(missing_values=np.nan,strategy="mean")
impute.fit(X[:,1:2])
X[:,1:2]=impute.transform(X[:,1:2])


impute.fit(y)
y=impute.transform(y)

from sklearn.preprocessing import OrdinalEncoder
encoder=OrdinalEncoder()
X[:,0:1]=encoder.fit_transform(X[:,0:1])


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=25,random_state=0)
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
np.set_printoptions(precision=3)
print(np.concatenate((y_test.reshape(len(y_test),1),y_pred.reshape(len(y_pred),1)),1))


# saving model to the disk
pickle.dump(regressor,open("model.pkl","wb"))


# loading model to compare the results
model=pickle.load(open("model.pkl","rb"))