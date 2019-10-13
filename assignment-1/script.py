import sys
import random
import pandas as pd
from tasks import *
from utilities import *


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

# show a random image
get_image_from_array(array=random.choice(input_data)).show()

# fit naive bayes and print results
fit_naive_bayes(x=input_data, y=output_data)

# find k most important attributes and use naive bayes again
input_data_new = get_attribute_importance(x=input_data, y=output_data, k=500)
fit_naive_bayes(x=input_data_new, y=output_data)
