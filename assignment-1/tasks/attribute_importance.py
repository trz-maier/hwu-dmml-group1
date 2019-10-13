from sklearn.ensemble import ExtraTreesClassifier
from utilities import get_image_from_array


def get_attribute_importance(x, y):
    model = ExtraTreesClassifier()
    model.fit(x, y)
    imp = model.feature_importances_
    imp *= 255.0/imp.max()
    get_image_from_array(array=imp).show()
