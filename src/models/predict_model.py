from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def predict_model(model, x):
    return model.predict(x)

def evaluate_model(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    return cm, acc
