from sklearn.feature_selection import SelectKBest, chi2
from utilities import get_image_from_array


def get_attribute_importance(x, y, k: int):
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(x, y)
    scores = selector.scores_
    scores *= 255.0/scores.max()
    get_image_from_array(array=scores).show()
    return selector.fit_transform(X=x, y=y)
