import numpy as np
import pandas as pd
from utilities import labels
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score


def fit_naive_bayes(x: np.array, y: np.array, print_results=True):

    # split data using stratify sampling
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

    # train the model
    model = GaussianNB()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    # print results
    if print_results:
        cm = confusion_matrix(y_test, predictions)
        cr = classification_report(y_test, predictions, target_names=list(labels.values()))

        print("Outcome:\nNaive Bayes correctly predicted the classification for %s of %s signs using %s attributes\n"
              % (sum(predictions == y_test), len(y_test), x.shape[1]))
        print("Classification report:\n%s\n" % cr)
        print("Confusion matrix:\n%s\n" % pd.DataFrame(data=cm, index=labels, columns=labels))

    return balanced_accuracy_score(y_test, predictions)
