import sys
import numpy as np
import pandas as pd
from tasks import *
from utilities import *


labels = {
    '0': 'Speed limit 60',
    '1': 'Speed limit 80',
    '2': 'Speed limit 80 lifter',
    '3': 'Right of way at crossing',
    '4': 'Right of way in general',
    '5': 'Give way',
    '6': 'Stop',
    '7': 'No speed limit general',
    '8': 'Turn right down',
    '9': 'Turn left down'
}

# open csv file
df = pd.read_csv("%s/data/x_train_gr_smpl.csv" % sys.path[0]).astype(int)

# apply image enhancements
df = df.apply(func=enhance_image)

# label data-frame based on sample data
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


# fit naive bayes and print results
fit_naive_bayes(input_data, output_data)
