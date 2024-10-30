import pandas as pd

import numpy as np

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# NaN treatment and remove useless columns (?)

test = test.loc[:, test.columns != 'Cabin']
test = test.loc[:, test.columns != 'Name']
test = test.loc[:, test.columns != 'Ticket']

test['Embarked'] = test['Embarked'].fillna('S')
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

dictEmbarked = {'C': 1, 'Q': 2, 'S': 3}
test['Embarked'] = test['Embarked'].map(dictEmbarked)
dictSex = {"male": 1, "female": 2}
test['Sex'] = test['Sex'].map(dictSex)

dfWithoutCabin = df.loc[:, df.columns != 'Cabin']
dfWithoutCabin = dfWithoutCabin.loc[:, dfWithoutCabin.columns != 'Name']
dfWithoutCabin = dfWithoutCabin.loc[:, dfWithoutCabin.columns != 'Ticket']

dfWithoutCabin['Embarked'] = dfWithoutCabin['Embarked'].fillna('S')
dfWithoutCabin['Age'] = dfWithoutCabin['Age'].fillna(dfWithoutCabin['Age'].mean())

dfWithoutCabin['Embarked'] = dfWithoutCabin['Embarked'].map(dictEmbarked)
dfWithoutCabin['Sex'] = dfWithoutCabin['Sex'].map(dictSex)

# Build model

feature_cols = ['PassengerId', 'Pclass', 'Sex', 'Age','SibSp','Parch','Fare', 'Embarked']

Y = dfWithoutCabin['Survived']
X = dfWithoutCabin[feature_cols]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=16)

model = LogisticRegression(random_state=16)
model.fit(X_train, Y_train)

Y_pred = model.predict(test)
print(Y_pred)

# Make submission like Kaggle requires

submission = pd.DataFrame(Y_pred)
submission['PassengerId'] = test['PassengerId']
submission.set_index('PassengerId')
submission.rename(columns = {0:'Survived'}, inplace = True)
columns_titles = ["PassengerId","Survived"]
submission=submission.reindex(columns=columns_titles)

submission.to_csv('sublogreg.csv', encoding='utf-8', index=False)




