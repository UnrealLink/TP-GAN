from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import yaml
import os

def process_coord(x, y, width, height):
    return y, x, int(height), int(width)

def cut_patches(image, patches):
    '''
    Patch : max_w x max_h
    Left eye : 44x22
    Right eye : 44x22
    Nose : 46x66
    Mouth : 69x25
    '''
    left_eye = transforms.functional.pad(transforms.functional.crop(image, *process_coord(**patches['Left eye'])), 
        (0, 0, 48-int(patches['Left eye']['width']), 24-int(patches['Left eye']['height'])))
    right_eye = transforms.functional.pad(transforms.functional.crop(image, *process_coord(**patches['Right eye'])),
        (0, 0, 48-int(patches['Right eye']['width']), 24-int(patches['Right eye']['height'])))
    nose = transforms.functional.pad(transforms.functional.crop(image, *process_coord(**patches['Nose'])),
        (0, 0, 48-int(patches['Nose']['width']), 64-int(patches['Nose']['height'])))
    mouth = transforms.functional.pad(transforms.functional.crop(image, *process_coord(**patches['Mouth'])),
        (0, 0, 80-int(patches['Mouth']['width']), 24-int(patches['Mouth']['height'])))
    return left_eye, right_eye, nose, mouth

class TrainingSet(Dataset):
    def __init__(self, images_list, images_dir):
        super(TrainingSet, self).__init__()
        with open(images_list, 'r') as rf:
            self.images_list = yaml.safe_load(rf.read())
        self.images_dir = images_dir
        self.keys = list(self.images_list.keys())
        
    def __len__(self):
        return len(self.images_list.keys())
    
    def __getitem__(self, idx):
        # Return a dict with :
        #  - profile and frontal (ground truth) images with size 128x128, 64x64 and 32x32
        #  - profile and frontal (ground truth) patches of eyes, nose and mouth
        
        image_info = self.images_list[self.keys[idx]]
        image   = Image.open(os.path.join(self.images_dir, image_info['img']))
        imageGT = Image.open(os.path.join(self.images_dir, image_info['imgGT']))
        
        batch = dict()
        
        batch['id']       = int(image_info['id'])
        batch['img128']   = image
        batch['img64']    = transforms.functional.resize(image, (64, 64))
        batch['img32']    = transforms.functional.resize(image, (32, 32))
        batch['img128GT'] = imageGT
        batch['img64GT']  = transforms.functional.resize(imageGT, (64, 64))
        batch['img32GT']  = transforms.functional.resize(imageGT, (32, 32))
        batch['left_eye'], batch ['right_eye'], batch['nose'], batch['mouth'] = cut_patches(image, image_info)
        batch['left_eyeGT'], batch ['right_eyeGT'], batch['noseGT'], batch['mouthGT'] = cut_patches(imageGT, self.images_list[image_info['imgGT']])
        
        toTensor = transforms.ToTensor()
        for k in batch.keys():
            if (k == 'id'):
                continue
            batch[k] = toTensor(batch[k])
            #print('{} : {}'.format(k, batch[k].shape))
        
        return batch
