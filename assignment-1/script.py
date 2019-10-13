import sys
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from utilities import *


# open csv file
df = pd.read_csv("%s/data/x_train_gr_smpl.csv" % sys.path[0]).astype(int)

# apply image enhancements
df = df.apply(func=enhance_image)

# add training samples as numpy arrays to a list based on labels, randomize order
for x in range(10):
    index = ~pd.read_csv("%s/data/y_train_smpl_%s.csv" % (sys.path[0], x), squeeze=True).astype(bool)
    df.loc[index, 'label'] = str(x)

# shuffle data-frame rows
df_shuffled = df.sample(frac=1)

# create input/output arrays
input_data = df_shuffled.iloc[:, 0:2304].to_numpy()
output_data = df_shuffled['label'].to_numpy()

# show specific images, different each time you execute above code
get_image_from_array(array=input_data[3]).show()
get_image_from_array(array=input_data[199]).show()
get_image_from_array(array=input_data[8562]).show()



