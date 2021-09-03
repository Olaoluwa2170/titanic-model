# from sklearn.linear_model import LogisticRegression 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, precision_recall_fscore_support, roc_curve



#sensitivity_score = recall_score
def specificity_score(y_true, y_pred):
    p, r, s, f = precision_recall_fscore_support(y_true, y_pred)
    return r[0]



df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")




df['male'] = df['Sex'] == 'male'

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
model = LogisticRegression(random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
sensitivity_score = recall_score



print(sum(y_pred))
print(model.score(X_test, y_test))
print('accuracy_score:', accuracy_score(y_test, y_pred))


print('precision_score:', precision_score(y_test, y_pred))
print('f1_score:', f1_score(y_test, y_pred))

print('confusion matrix:', confusion_matrix(y_test, y_pred))



print('sensitivity_score: ', sensitivity_score(y_test, y_pred))
print('specificity: ', specificity_score(y_test, y_pred))
print('predict proba:')
model.predict_proba(X_test)


y_pred = model.predict_proba(X_test)[:, 1] > 0.75




print('precision_score:', precision_score(y_test, y_pred))
print('recall: ', recall_score(y_test, y_pred))


