import sys
import numpy as np
import pandas as pd
from utilities import *

# opening the csv file (place assignment files in data folder but do not add them to github as they are too large)
all_data = pd.read_csv("%s/data/x_train_gr_smpl.csv" % sys.path[0]).astype(int)

# add training samples as numpy arrays to a list based on labels, randomize order
training_samples = []
for x in range(10):
    index = pd.read_csv("%s/data/y_train_smpl_%s.csv" % (sys.path[0], x), squeeze=True).values.astype(bool)
    sample = all_data[[not x for x in index]].to_numpy()
    np.random.shuffle(sample)
    training_samples.append(sample)

# show specific images, different each time you execute above code
get_image_from_array(array=training_samples[0][299], brightness=1.2, enhance=True).show()  # random speed limit 60 sign
get_image_from_array(array=training_samples[5][299], brightness=1.2, enhance=True).show()  # random give way sign
get_image_from_array(array=training_samples[6][102], brightness=1.2, enhance=True).show()  # random stop sign
