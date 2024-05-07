import os
import random
import numpy as np
import torch
import torch.utils
import torch.utils.data as data
from tqdm import tqdm

class NIST(data.Dataset):
    def __init__(self, exp_path_list, n_frames=5, step_size=1, transform=None):
        super(NIST, self).__init__()

        self.dataset = []
        for exp_path in exp_path_list:
            img_path = os.path.join(exp_path, 'image_seq.npy')
            lab_path = os.path.join(exp_path, 'label_seq.npy')

            img_list = np.load(img_path)
            lab_list = np.load(lab_path)

            print(f"============={exp_path}===============")
            for i in tqdm(range(0, len(lab_list) - n_frames + 1, step_size)):
                img_sample = img_list[i: i + n_frames]
                lab_sample = lab_list[i: i + n_frames]

                self.dataset.append(tuple(img_sample, lab_sample))
        
        random.shuffle(self.dataset)
        self.length = len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return self.length

def load_subfolder_path_list(root):
    path_list = []
    for fold_name in os.listdir(root):
        path_list.append(os.path.join(root, fold_name))
    return path_list

def load_data(
        batch_size, val_batch_size,
        data_root, train_exp_name, valid_exp_name, test_exp_name, 
        num_workers, apply_valid_to_train=False):
    
    train_exp_path = os.path.join(data_root, train_exp_name)
    valid_exp_path = os.path.join(data_root, valid_exp_name)
    test_exp_path = os.path.join(data_root, test_exp_name)

    train_exp_list = load_subfolder_path_list(train_exp_path)
    valid_exp_list = load_subfolder_path_list(valid_exp_path)
    test_exp_list = load_subfolder_path_list(test_exp_path)

    if apply_valid_to_train:
        train_exp_list.append(valid_exp_list)
        valid_exp_list = test_exp_list

    train_set = NIST(exp_path_list=train_exp_list, n_frames=5, step_size=1)
    valid_set = NIST(exp_path_list=valid_exp_list, n_frames=5, step_size=1)
    test_set = NIST(exp_path_list=test_exp_list, n_frames=5, step_size=1)

    dataloader_train = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
    dataloader_validation = data.DataLoader(
        valid_set, batch_size=val_batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)
    dataloader_test = data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)
    
    return dataloader_train, dataloader_validation, dataloader_test
