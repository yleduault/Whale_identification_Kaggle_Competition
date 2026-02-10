import os
import pandas as pd
import cv2
from torch.utils.data import Dataset


def load_training_csv(path):
    csv_f = pd.read_csv(path)
    return csv_f.to_numpy()

def load_data(path,csv_path):
    csv_arr = load_training_csv(csv_path)
    whale_id_int_lst = []
    paths_arr = []
    for img_p,w_id in csv_arr:
        if w_id not in whale_id_int_lst:
            whale_id_int_lst.append(w_id)
            idx = len(whale_id_int_lst)
        else:
            idx = whale_id_int_lst.index(w_id)
        if not os.path.isfile(os.path.join(path,'train',img_p)):
            raise Exception("Path to image not found : {}".format(os.path.join(path,'train',img_p)))
        paths_arr.append([idx,os.path.join(path,'train',img_p)])
    return paths_arr,whale_id_int_lst

def load_img(path):
    return cv2.imread(path,cv2.IMREAD_COLOR_RGB)

class WhaleDataset(Dataset):
    def __init__(self,data_folder_path,csv_path,data_augmentation=None):
        super(WhaleDataset).__init__()
        self.paths,self.id_list = load_data(data_folder_path,csv_path)
        self.data_augmentations = data_augmentation

    def __getitem__(self, item):
        idx = self.paths[item][0]
        img = load_img(self.paths[item][1])
        if self.data_augmentations is not None:
            return self.data_augmentations(image=img),idx
        else:
            return img,idx
    def __len__(self):
        return len(self.paths)

