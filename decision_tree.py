# creating a model with decision tree class

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import KFold

df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")




df['male'] = df['Sex'] == 'male'

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

model = DecisionTreeClassifier()

kf = KFold(n_splits=5, shuffle=True)



for criterion in ['gini','entropy']:
    print(f"Decision tree - {criterion}")
    recall = []
    accuracy = []
    precision = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dt = DecisionTreeClassifier(criterion = criterion)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
    
    accuracy.append(accuracy_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    

print('accuracy: ', np.mean(accuracy))
print('recall: ', np.mean(recall))
print('precision: ', np.mean(precision))
    
    

#model.fit(X_train, y_train)


#print('Accuracy: ', model.score(X_test, y_test))

#y_pred = model.predict(X_test)

#print('recall_score :', recall_score(y_test, y_pred))
#print('precision_score :',  precision_score(y_test, y_pred))
#print('F1_score :', f1_score(y_test, y_pred))