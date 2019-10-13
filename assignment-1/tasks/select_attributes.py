import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from utilities import get_image_from_array
from tasks import fit_naive_bayes


def find_best_attribute_number(x, y, step: int):
    best = [0, 0]
    x_array = []
    y_array = []
    for v in range(step, x.shape[1]-step, step):
        score = fit_naive_bayes(get_new_feature_set(x, y, v, False), y, False)
        x_array.append(v)
        y_array.append(score)
        if score > best[1]:
            best = [v, score]
    plt.plot(x_array, y_array)
    plt.show()
    return best[0]


def get_new_feature_set(x, y, k: int, show_image=True):
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(x, y)
    scores = selector.scores_
    scores *= 255.0/scores.max()
    if show_image:
        get_image_from_array(array=scores).show()
    return selector.fit_transform(X=x, y=y)
