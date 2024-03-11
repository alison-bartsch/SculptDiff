import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from pointBERT.utils.logger import *

@DATASETS.register_module()
class ClayDemos(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS

        # self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        # test_data_list_file = os.path.join(self.data_root, 'test.txt')
        
        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'ClayDemos')
        # print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'ShapeNet-55')
        # with open(self.data_list_file, 'r') as f:
        #     lines = f.readlines()
        # if self.whole:
        #     with open(test_data_list_file, 'r') as f:
        #         test_lines = f.readlines()
        #     print_log(f'[DATASET] Open file {test_data_list_file}', logger = 'ShapeNet-55')
        #     lines = test_lines + lines
        # self.file_list = []
        # for line in lines:
        #     line = line.strip()
        #     taxonomy_id = line.split('-')[0]
        #     model_id = line.split('-')[1].split('.')[0]
        #     self.file_list.append({
        #         'taxonomy_id': taxonomy_id,
        #         'model_id': model_id,
        #         'file_path': line
        #     })

        self.actions = IO.get(self.pc_path + '/action_normalized.npy').astype(np.float32)
        print_log(f'[DATASET] {len(self.actions)} instances were loaded', logger = 'ClayDemos')

        self.permutation = np.arange(self.npoints)
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        # sample = self.file_list[idx]

        data = IO.get(self.pc_path + '/States/shell_scaled_state' + str(idx) + '.npy').astype(np.float32)

        data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        # return sample['taxonomy_id'], sample['model_id'], data
        return idx, idx, data

    def __len__(self):
        # return len(self.file_list)
        return len(self.actions)