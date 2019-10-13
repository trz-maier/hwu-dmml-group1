import numpy as np
import pandas as pd
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report


def fit_naive_bayes(input_data: np.array, output_data: np.array):

    population_size = len(input_data)
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # set up training data
    training_size = int(population_size*0.7)
    training_indexes = random.sample(range(population_size), training_size)
    training_input = input_data[training_indexes]
    training_output = output_data[training_indexes]

    # train the model
    model = GaussianNB()
    model.fit(training_input, training_output)

    # set up validation data
    validation_indexes = [x for x in range(population_size) if x not in training_indexes]
    validation_input = input_data[validation_indexes]
    validation_output = output_data[validation_indexes]

    # predict outputs of validation data
    predictions = model.predict(validation_input)
    cm = pd.DataFrame(data=confusion_matrix(validation_output, predictions), index=labels, columns=labels)
    cr = classification_report(validation_output, predictions, target_names=labels)

    # print results
    print("""
Outcome:
Naive Bayes correctly predicted the classification for %s of %s signs

Classification report:
%s

Confusion matrix:
%s
    """
          % (sum(predictions == validation_output), len(validation_output), cr, cm))
