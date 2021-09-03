# Building a Model with kfold

from sklearn.model_selection import KFold
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")

df['male'] = df['Sex'] == 'male'

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

model = LogisticRegression()


# creating a model with Kfold instead of train_test_split function

kf = KFold(n_splits=5, shuffle=True)

#for train, test in Kf.split(X):
        #train, test
splits = list(kf.split(X))
first_split = splits[0]
train_indices, test_indices = first_split


X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]


model.fit(X_train, y_train)
print(model.score(X_test, y_test))

