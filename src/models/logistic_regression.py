from sklearn.linear_model import LogisticRegression

def train_logistic_regression(x_train, y_train):
    lg = LogisticRegression()
    lg.fit(x_train, y_train)
    return lg
