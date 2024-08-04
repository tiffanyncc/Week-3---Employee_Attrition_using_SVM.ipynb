from sklearn.svm import SVC

def train_svm(x_train, y_train, kernel='linear'):
    svm = SVC(kernel=kernel)
    model = svm.fit(x_train, y_train)
    return model
