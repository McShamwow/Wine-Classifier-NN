import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

X = pd.read_csv('frx.csv')
y = pd.read_csv('fry.csv')

X = X.iloc[1:, 1:]
y = y.iloc[1:, 1:]

model = ExtraTreesClassifier()
model.fit(X, y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()