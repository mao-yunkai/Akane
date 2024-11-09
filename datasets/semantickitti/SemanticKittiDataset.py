import os
import numpy as np
import yaml
from torch.utils.data import Dataset

class SemanticKittiDataset(Dataset):
    def __init__(self, data_path, splitset="train"):
        with open('datasets/semantickitti/semantickitti.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.labels = config['labels']
        self.learning_labels = config['learning_labels']
        self.learning_map, self.learning_map_inv = self.learning_label_map()
        self.colors = config['colors']
        self.split = config['split']
        self.splitset = splitset
        self.data_list = []
        self.label_list = []
        for folder in self.split[self.splitset]:
            self.data_list.extend([os.path.join(data_path, str(folder).zfill(2),'velodyne', file) for file in sorted(os.listdir(os.path.join(data_path, str(folder).zfill(2), 'velodyne')))])
            self.label_list.extend([os.path.join(data_path, str(folder).zfill(2), 'labels', file) for file in sorted(os.listdir(os.path.join(data_path, str(folder).zfill(2), 'labels')))])
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data = np.fromfile(self.data_list[index], dtype=np.float32).reshape(-1,4)
        label = np.fromfile(self.label_list[index], dtype=np.int32).reshape(-1,1)
        label = label & 0xFFFF
        label = np.vectorize(self.learning_map.__getitem__)(label)
        return data, label
    
    def learning_label_map(self):
        # 构建映射关系
        learning_map = {}
        learning_map_inv = {}

        for key, value in self.labels.items():
            if value in self.learning_labels.values():
                # 找到对应的learning_labels键
                learning_key = list(self.learning_labels.keys())[list(self.learning_labels.values()).index(value)]
                learning_map_inv[learning_key] = key
                learning_map[key] = learning_key
            else:
                learning_map[key] = 0
        return learning_map, learning_map_inv
