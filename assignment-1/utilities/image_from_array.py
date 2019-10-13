import numpy as np
from PIL import Image


def get_image_from_array(array: np.array, x=48, y=48):
    image = Image.fromarray(array.astype(int).reshape(x, y))
    return image
