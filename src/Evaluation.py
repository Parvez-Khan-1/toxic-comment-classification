from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def get_model_accuracy(actual_result, predicted_result):
    return accuracy_score(actual_result, predicted_result, normalize=True)


def show_confusion_matrix(actual_result, predicted_result):
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(actual_result, predicted_result).ravel()
    return true_negative, false_positive, false_negative, true_positive


def show_classification_report(actual_result, predicted_result, target_names):
    return classification_report(actual_result, predicted_result, target_names)
