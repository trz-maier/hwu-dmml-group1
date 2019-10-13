import sys
import random
import pandas as pd
import numpy as np
from tasks import *
from utilities import *

# define location of your data folder
loc = sys.path[0]

# open csv file
df = pd.read_csv("%s/data/x_train_gr_smpl.csv" % loc).astype(int)

# apply image enhancements
df = df.apply(func=enhance_image)

# label data-frame based on sample data
for x in range(10):
    index = ~pd.read_csv("%s/data/y_train_smpl_%s.csv" % (loc, x), squeeze=True).astype(bool)  # reversed flags
    df.loc[index, 'label'] = str(x)

# randomize data (same result each time)
np.random.seed(42)
permutation = np.random.permutation(df.shape[0])

# create randomized input/output arrays
input_data = df.iloc[:, 0:2304].to_numpy()[permutation]
output_data = df.iloc[:, 2304].to_numpy()[permutation]

# show a random image
get_image_from_array(array=random.choice(input_data)).show()

# fit naive bayes and print results
fit_naive_bayes(x=input_data, y=output_data)

# find best number of attributes and use naive bayes again
# best = find_best_attribute_number(x=input_data, y=output_data, step=100, max=2000)
best = 700
input_data_trimmed = get_new_feature_set(x=input_data, y=output_data, k=best)
fit_naive_bayes(x=input_data_trimmed, y=output_data)

