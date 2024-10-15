import numpy as np
import pandas as pd
from scipy.io import arff


##### READING DATASET
def read_dataset(directory, dataset_name):


    if dataset_name in ["EQ","JDT","LC", "ML", "PDE","CM1","MW1","PC1", "PC3", "PC4","ar1","ar3","ar4", "ar5", "ar6"]:
        data, meta = arff.loadarff(directory + dataset_name + '.arff')
        X =  pd.DataFrame(data)

        y = X['class']
        y = mapit(y)
        del X['class']
    else:
        data, meta = arff.loadarff(directory + dataset_name + '.arff')
        X =  pd.DataFrame(data)

        y = X['class']
        y = mapit(y)
        del X['class']
    # else:
    #     print("dataset %s does not exist" % dataset_name)


    return np.array(X), np.array(y)
#### MISC
def mapit(vector):

    s = np.unique(vector)

    mapping = pd.Series([x[0] for x in enumerate(s)], index = s)
    vector=vector.map(mapping)
    return vector
