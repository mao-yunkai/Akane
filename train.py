from torch.utils.data import DataLoader
from datasets.semantickitti.SemanticKittiDataset import SemanticKittiDataset


if __name__=="__main__":
    train_data = SemanticKittiDataset('data\semantickitti\sequences', 'train')
    valid_data = SemanticKittiDataset('data\semantickitti\sequences', 'valid')

    train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers = 0)
    valid_dataloader = DataLoader(dataset=valid_data, batch_size=64, shuffle=False, num_workers = 0)

    print(next(iter(train_dataloader)))