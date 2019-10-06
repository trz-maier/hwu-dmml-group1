import numpy as np
from PIL import Image


def get_image_from_array(array: np.array, x=48, y=48, brightness=1.0, enhance=True):
    if enhance:
        array = (array / (array.max() / 255.)).astype(int)
    array = array*brightness
    image = Image.fromarray(array.astype(int).reshape(x, y))
    return image
