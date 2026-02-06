import numpy as np
import pandas as pd

# def load_imgs_to_tensor(paths):



def load_training_csv(path):
    csv_f = pd.read_csv(path)
    return csv_f.to_numpy()