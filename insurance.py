import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

file_path = os.path.join(os.path.dirname(__file__), 'datasets', 'ins.csv')

print("Resolved path:", file_path)
print("Exists:", os.path.exists(file_path))

df = pd.read_csv(file_path)


X=df.drop('charges',axis=1)
y=df['charges']
X=pd.get_dummies(X,drop_first=True)

X_train,X_test,y_train,y_test= train_test_split(X,y,
                                                test_size=0.2,random_state=42)

model=LinearRegression()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("R2=",r2_score(y_test,y_pred))
print("mean squared error",mean_squared_error(y_test,y_pred))




