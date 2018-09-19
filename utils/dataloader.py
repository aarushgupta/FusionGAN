from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from PIL import Image

class YouTubePose(Dataset):
    
    def __init__(self, datapoint_pairs, shapeLoss_datapoint_pairs, dataset_dir, transform=None, mode='train'):
        self.datapoint_pairs = datapoint_pairs
        self.shapeLoss_datapoint_pairs = shapeLoss_datapoint_pairs
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.mode = mode
    
    def __len__(self):
        return len(self.datapoint_pairs)
    
    def __getitem__(self, idx):
        image_pair = self.datapoint_pairs[idx]
        x_gen_path = image_pair[0]
        x_dis_path = image_pair[1]
        y_path = image_pair[2]
        
        identity_pair = self.shapeLoss_datapoint_pairs[idx]
        iden_1_path = identity_pair[0]
        iden_2_path = identity_pair[1]
        
        x_gen = Image.open(self.dataset_dir + self.mode + '/' + x_gen_path)
        x_dis = Image.open(self.dataset_dir + self.mode + '/' + x_dis_path)
        y = Image.open(self.dataset_dir + self.mode + '/' + y_path)
        iden_1 = Image.open(self.dataset_dir + self.mode + '/' + iden_1_path)
        iden_2 = Image.open(self.dataset_dir + self.mode + '/' + iden_2_path)
        
        if self.transform:
            x_gen = self.transform(x_gen)
            x_dis = self.transform(x_dis)
            y = self.transform(y)
            iden_1 = self.transform(iden_1)
            iden_2 = self.transform(iden_2)
            
        sample = {'x_gen' : x_gen, 'x_dis': x_dis, 'y': y, 'iden_1': iden_1, 'iden_2':iden_2}
        return sample
