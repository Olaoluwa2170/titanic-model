# Model Comparison Building the Models with Scikit-learn


from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'


kf = KFold(n_splits=5, shuffle=True)


# create three different feature matrices X1, X2 and X3. All will have the same target y
X1 = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
X2 = df[['Pclass', 'male', 'Age']].values
X3 = df[['Fare', 'Age']].values
y = df['Survived'].values




def score_model(X, y, kf):
    accuracy_scores = [] 
    precision_scores = [] 
    recall_scores = [] 
    f1_scores = [] 
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index] 
        model = LogisticRegression(random_state = 0) 
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test) 
        accuracy_scores.append(accuracy_score(y_test, y_pred)) 
        precision_scores.append(precision_score(y_test, y_pred)) 
        recall_scores.append(recall_score(y_test, y_pred)) 
        f1_scores.append(f1_score(y_test, y_pred))
    
    print("accuracy:", np.mean(accuracy_scores)) 
    print("precision:", np.mean(precision_scores)) 
    print("recall:", np.mean(recall_scores)) 
    print("f1 score:", np.mean(f1_scores))
    
print('Logistic Regression with all the features')
score_model(X1, y, kf)
print()

print('Logistic Regression with Pclass, male, age')
score_model(X2, y, kf)
print()

print('Logistic Regression with Fare and age ')
score_model(X3, y, kf)

"""if we compare the first two models, they have almost identical scores. The third model has lower scores for all four metrics. The first two are thus much better options than the third. This matches intuition since the third model doesn’t have access to the sex of the passenger. Our expectation is that women are more likely to survive, so having the sex would be a very valuable predictor.
 Since the first two models have equivalent results, it makes sense to choose the simpler model, the one that uses the Pclass, Sex & Age features.
 Now that we’ve made a choice of a best model, we build a single final model using all of the data."""




