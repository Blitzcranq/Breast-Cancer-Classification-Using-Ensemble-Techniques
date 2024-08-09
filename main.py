import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer Wisconsin (Diagnostic) dataset
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Data imputation
imp = SimpleImputer(strategy='most_frequent')
X_train_imputed = imp.fit_transform(X_train)
X_test_imputed = imp.transform(X_test)

# Decision Tree Classifier
clf_decision_tree = DecisionTreeClassifier(random_state=0)
clf_decision_tree.fit(X_train_imputed, y_train)
y_pred_decision_tree = clf_decision_tree.predict(X_test_imputed)

# Gaussian Naive Bayes Classifier
clf_naive_bayes = GaussianNB()  # Use Gaussian Naive Bayes
clf_naive_bayes.fit(X_train_imputed, y_train)
y_pred_naive_bayes = clf_naive_bayes.predict(X_test_imputed)

# Create new classifiers for ensemble techniques for Decision Tree
bagging_classifier_dt = BaggingClassifier(estimator=clf_decision_tree, n_estimators=50, random_state=0)
adaboost_classifier_dt = AdaBoostClassifier(estimator=clf_decision_tree, n_estimators=50, random_state=0, algorithm='SAMME')

# Stacking classifier with Decision Tree as base and Bagging and AdaBoost as estimators
estimators_dt = [('bagging', bagging_classifier_dt), ('adaboost', adaboost_classifier_dt)]
stacking_classifier_dt = StackingClassifier(estimators=estimators_dt, final_estimator=clf_decision_tree, cv=3)

# Create new classifiers for ensemble techniques for Naive Bayes
bagging_classifier_nb = BaggingClassifier(estimator=clf_naive_bayes, n_estimators=50, random_state=0)
adaboost_classifier_nb = AdaBoostClassifier(estimator=clf_naive_bayes, n_estimators=50, random_state=0, algorithm='SAMME')

# Stacking classifier with Naive Bayes as base and Bagging and AdaBoost as estimators
estimators_nb = [('bagging', bagging_classifier_nb), ('adaboost', adaboost_classifier_nb)]
stacking_classifier_nb = StackingClassifier(estimators=estimators_nb, final_estimator=clf_naive_bayes, cv=3)

# Train the bagging classifier for Decision Tree
bagging_classifier_dt.fit(X_train_imputed, y_train)

# Train the AdaBoost classifier for Decision Tree
adaboost_classifier_dt.fit(X_train_imputed, y_train)

# Train the stacking classifier for Decision Tree
stacking_classifier_dt.fit(X_train_imputed, y_train)

# Train the bagging classifier for Naive Bayes
bagging_classifier_nb.fit(X_train_imputed, y_train)

# Train the AdaBoost classifier for Naive Bayes
adaboost_classifier_nb.fit(X_train_imputed, y_train)

# Train the stacking classifier for Naive Bayes
stacking_classifier_nb.fit(X_train_imputed, y_train)

# Make predictions with the ensemble classifiers for Decision Tree
y_pred_bagging_dt = bagging_classifier_dt.predict(X_test_imputed)
y_pred_adaboost_dt = adaboost_classifier_dt.predict(X_test_imputed)
y_pred_stacking_dt = stacking_classifier_dt.predict(X_test_imputed)

# Make predictions with the ensemble classifiers for Naive Bayes
y_pred_bagging_nb = bagging_classifier_nb.predict(X_test_imputed)
y_pred_adaboost_nb = adaboost_classifier_nb.predict(X_test_imputed)
y_pred_stacking_nb = stacking_classifier_nb.predict(X_test_imputed)

# Evaluate Decision Tree Classifier
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)

# Evaluate Naive Bayes Classifier
accuracy_naive_bayes = accuracy_score(y_test, y_pred_naive_bayes)

# Evaluate Bagging Classifier for Decision Tree
accuracy_bagging_dt = accuracy_score(y_test, y_pred_bagging_dt)

# Evaluate AdaBoost Classifier for Decision Tree
accuracy_adaboost_dt = accuracy_score(y_test, y_pred_adaboost_dt)

# Evaluate Stacking Classifier for Decision Tree
accuracy_stacking_dt = accuracy_score(y_test, y_pred_stacking_dt)

# Evaluate Bagging Classifier for Naive Bayes
accuracy_bagging_nb = accuracy_score(y_test, y_pred_bagging_nb)

# Evaluate AdaBoost Classifier for Naive Bayes
accuracy_adaboost_nb = accuracy_score(y_test, y_pred_adaboost_nb)

# Evaluate Stacking Classifier for Naive Bayes
accuracy_stacking_nb = accuracy_score(y_test, y_pred_stacking_nb)

# Output the accuracies
print("Decision Tree Classifier:")
print(f"Accuracy: {accuracy_decision_tree:.2f}")

print("Naive Bayes Classifier:")
print(f"Accuracy: {accuracy_naive_bayes:.2f}")

print("Bagging Classifier for Decision Tree:")
print(f"Accuracy: {accuracy_bagging_dt:.2f}")

print("AdaBoost Classifier for Decision Tree:")
print(f"Accuracy: {accuracy_adaboost_dt:.2f}")

print("Stacking Classifier for Decision Tree:")
print(f"Accuracy: {accuracy_stacking_dt:.2f}")

print("Bagging Classifier for Naive Bayes:")
print(f"Accuracy: {accuracy_bagging_nb:.2f}")

print("AdaBoost Classifier for Naive Bayes:")
print(f"Accuracy: {accuracy_adaboost_nb:.2f}")

print("Stacking Classifier for Naive Bayes:")
print(f"Accuracy: {accuracy_stacking_nb:.2f}")
