from torch.utils.data import Dataset
import numpy as np

class VoxelDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data, labels = self.dataset[index]
        max_bound = np.percentile(data,100,axis = 0)
        min_bound = np.percentile(data,0,axis = 0)
        
        # if self.fixed_volume_space:
        #     max_bound = np.asarray(self.max_volume_space)
        #     min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        self.grid_size = np.array([10,10,5,5])
        cur_grid_size = self.grid_size
        
        intervals = crop_range/(cur_grid_size-1)
        if (intervals==0).any(): print("Zero interval!")
        
        grid_ind = (np.floor((np.clip(data,min_bound,max_bound)-min_bound)/intervals)).astype(np.int64)

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        # process labels
        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]
        # processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)
        
        data_tuple = (voxel_position,processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_data = data - voxel_centers
        return_xyz = np.concatenate((return_xyz,data),axis = 1)
        return data, labels
    
if __name__ == "__main__":
    from semantickitti.SemanticKittiDataset import SemanticKittiDataset
    dataset = SemanticKittiDataset('data\semantickitti\sequences', 'train')
    print(VoxelDataset(dataset)[0])