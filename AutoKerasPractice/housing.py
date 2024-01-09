import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak

from sklearn.datasets import fetch_california_housing

house_dataset = fetch_california_housing()
df = pd.DataFrame(
    np.concatenate(
        (house_dataset.data, house_dataset.target.reshape(-1,1)), axis = 1
    ),
    columns=house_dataset.feature_names + ["Price"],
)