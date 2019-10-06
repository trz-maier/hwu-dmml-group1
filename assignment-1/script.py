import sys
import pandas as pd

# opening the csv file (place assignment files in data folder but do not add them to github as they are too large)
file = open(sys.path[0]+"/data/y_train_smpl_0.csv")

# creating a pandas data-frame from csv file
df = pd.read_csv(file)

print(df)
