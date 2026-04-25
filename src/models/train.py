from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def train_baselines(X_train, y_train, random_state: int):
    print("Training Baseline Models...")
    
    # 1. Gaussian Naive Bayes
    print(" -> Fitting Gaussian Naive Bayes...")
    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)
    
    # 2. Decision Tree
    print(" -> Fitting Decision Tree...")
    dt_clf = DecisionTreeClassifier(random_state=random_state)
    dt_clf.fit(X_train, y_train)
    
    print("Baseline training complete.")
    return nb_clf, dt_clf