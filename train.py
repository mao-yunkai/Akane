from torch.utils.data import DataLoader
from datasets.semantickitti.SemanticKittiDataset import SemanticKittiDataset
def main():
    pass

if __name__=="__main__":
    train_data = SemanticKittiDataset('data\semantickitti\sequences', 'train')
    valid_data = SemanticKittiDataset('data\semantickitti\sequences', 'valid')

    train_dataloader = DataLoader(dataset=train_data, batch_size=2, shuffle=True, num_workers = 0, collate_fn=SemanticKittiDataset.collate_fn)
    valid_dataloader = DataLoader(dataset=valid_data, batch_size=2, shuffle=False, num_workers = 0, collate_fn=SemanticKittiDataset.collate_fn)

    print(next(iter(train_dataloader)))

