import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier

matplotlib.use('TkAgg')

presos = pd.read_csv('presos.csv')

# sns.pairplot(presos)

presos['SEXO'].fillna(30, inplace=True)
presos['IDADE'].fillna(0, inplace=True)
presos['REGIME'].fillna(1650, inplace=True)
presos['FACCAO'].fillna(0, inplace=True)
presos['CARGO_FACCAO'].fillna(0, inplace=True)
presos.loc[presos['IDADE'] > 60, 'IDADE'] = 0
presos.loc[presos['IDADE'] < 18, 'IDADE'] = presos['IDADE'].mean()
presos['IDADE'] = presos['IDADE'].apply(lambda x: int(x))
presos['SEXO'] = presos['SEXO'].apply(lambda x: int(x))
presos['REGIME'] = presos['REGIME'].apply(lambda x: int(x))
presos['FACCAO'] = presos['FACCAO'].apply(lambda x: int(x))
presos['CARGO_FACCAO'] = presos['CARGO_FACCAO'].apply(lambda x: int(x))
presos['UNIDADE'] = presos['UNIDADE'].apply(lambda x: int(x))

X = presos.iloc[:, :5]
y = presos.iloc[:, -1]

# print(y)
clf = RandomForestClassifier(max_depth=10, random_state=0)

clf.fit(X, y)
print(clf.feature_importances_)

pickle.dump(clf, open('uppredictionmodel.pkl', 'wb'))
model = pickle.load(open('uppredictionmodel.pkl', 'rb'))
#
print(model.predict([[31, 22, 2410, 0, 0]]))
