from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


def run_model_training(X_train,X_test,y_train,y_test):
    log = LogisticRegression()
    log.fit(X_train,y_train)
    log.score(X_test,y_test)
    y_pred = log.predict(X_test)
    print(classification_report(y_test,y_pred))
    return log
