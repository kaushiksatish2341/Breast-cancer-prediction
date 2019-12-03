import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge as linreg
from sklearn.model_selection import train_test_split

a=pd.read_csv('breast-cancer.csv')
a=a.fillna(method='ffill')

#print(a)

X=a[['radius_mean','texture_mean','perimeter_mean','area_mean']]
y=a['fractal_dimension_worst']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=8)

lr=linreg().fit(X_train,y_train)

print('Coefficient: ',lr.coef_)
print('Intercept: ',lr.intercept_)

print('R-squared score(training):{:.3f}'.format(lr.score(X_train,y_train)))
print('R-squared score(test):{:.3f}'.format(lr.score(X_test,y_test)))

