import pandas as pd

import numpy as np

from sklearn import linear_model
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

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

# View and describe dataframe, trying to find NaN

#print(df)
#print(df.describe())
#print(df.isnull().sum())

# NaN treatment

dfWithoutCabin = df.loc[:, df.columns != 'Cabin']
dfWithoutCabin = dfWithoutCabin.loc[:, dfWithoutCabin.columns != 'Name']
dfWithoutCabin = dfWithoutCabin.loc[:, dfWithoutCabin.columns != 'Ticket']

dfWithoutCabin['Embarked'] = dfWithoutCabin['Embarked'].fillna('S')
dfWithoutCabin['Age'] = dfWithoutCabin['Age'].fillna(dfWithoutCabin['Age'].mean())

dfWithoutCabin['Embarked'] = dfWithoutCabin['Embarked'].map(dictEmbarked)
dfWithoutCabin['Sex'] = dfWithoutCabin['Sex'].map(dictSex)

Y = dfWithoutCabin['Survived']
X = dfWithoutCabin.loc[:, dfWithoutCabin.columns != 'Survived']

model = LinearRegression()
model.fit(X, Y)

predictions = model.predict(test)
predOut = np.where(predictions > 0.5, 1, 0)

submission = pd.DataFrame(predOut)
submission['PassengerId'] = test['PassengerId']
submission.set_index('PassengerId')
submission.rename(columns = {0:'Survived'}, inplace = True)
columns_titles = ["PassengerId","Survived"]
submission=submission.reindex(columns=columns_titles)

submission.to_csv('subm.csv', encoding='utf-8', index=False)



