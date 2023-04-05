import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file, patch_size, scale):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        self.patch_size = patch_size
        self.scale = scale
        self.f = h5py.File(self.h5_file, 'r')
        self.l = len(self.f['lh'])

    def __getitem__(self, idx):
        return np.expand_dims(self.f['lh'][idx][0][0:self.patch_size//self.scale, 0:self.patch_size//self.scale], 0), np.expand_dims(self.f['lh'][idx][1], 0)

    def __len__(self):
        return self.l


class EvalDataset(Dataset):
    def __init__(self, h5_file, patch_size, scale):
        super(EvalDataset, self).__init__
        self.h5_file = h5_file
        self.patch_size = patch_size
        self.scale = scale
        self.f = h5py.File(self.h5_file, 'r')
        self.l = len(self.f['lh'])

    def __getitem__(self, idx):
        return np.expand_dims(self.f['lh'][idx][0][0:self.patch_size//self.scale, 0:self.patch_size//self.scale], 0), np.expand_dims(self.f['lh'][idx][1], 0)

    def __len__(self):
        return self.l
        

# # duplicate
# import h5py
# import numpy as np
# from torch.utils.data import Dataset


# class TrainDataset(Dataset):
#     def __init__(self, h5_file):
#         super(TrainDataset, self).__init__()
#         self.himgs = os.listdir('Datasets/DIV2K_train_HR_y')
#         self.limgs = os.listdir('Datasets/DIV2K_train_LR_y')
#         self.l = len(self.himgs)

#     def __getitem__(self, idx):
#         print(idx)
#         print(np.array(Image.open(os.path.join('Datasets/DIV2K_train_LR_y', self.himgs[idx]))))
#         return np.array(Image.open(os.path.join('Datasets/DIV2K_train_LR_y', self.himgs[idx]))), np.array(Image.open(os.path.join('Datasets/DIV2K_train_HR_y', self.himgs[idx])))

#     def __len__(self):
#         return self.l


# class EvalDataset(Dataset):
#     def __init__(self, h5_file):
#         super(EvalDataset, self).__init__()
#         self.himgs = os.listdir('Datasets/DIV2K_valid_HR_y')
#         self.limgs = os.listdir('Datasets/DIV2K_valid_LR_y')
#         self.l = len(self.himgs)

#     def __getitem__(self, idx):
#         return np.array(Image.open(os.path.join('Datasets/DIV2K_valid_LR_y', self.himgs[idx]))), np.array(Image.open(os.path.join('Datasets/DIV2K_valid_HR_y', self.himgs[idx])))

#     def __len__(self):
#         return self.l

# duplicate

# from PIL import Image

# image_dir = r'Datasets/DIV2K_train_HR'
# image_save_dir = r'Datasets/DIV2K_train_LR_y'
# for img_name in os.listdir(image_dir):
#     img = Image.open(os.path.join(image_dir, img_name)).convert('RGB')
    
#     img = img.resize((img.width // scale, img.height // scale), resample=Image.BICUBIC)
    
#     img = np.array(img).astype(np.float32)
    
#     img = convert_rgb_to_y(img)

#     img = Image.fromarray(img)

#     img.save(os.path.join(image_save_dir, (img_name[:-4] + '.tiff')))