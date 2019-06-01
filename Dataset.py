from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import yaml
import os
from random import random

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
    left_eye = transforms.functional.crop(image, *process_coord(**patches['Left eye']))
    left_eye = transforms.functional.pad(left_eye, (0, 0, 48-left_eye.size[-2], 24-left_eye.size[-1]))
    right_eye = transforms.functional.crop(image, *process_coord(**patches['Right eye']))
    right_eye = transforms.functional.pad(right_eye, (0, 0, 48-right_eye.size[-2], 24-right_eye.size[-1]))
    nose = transforms.functional.crop(image, *process_coord(**patches['Nose']))
    nose = transforms.functional.pad(nose, (0, 0, 48-nose.size[-2], 64-nose.size[-1]))
    mouth = transforms.functional.crop(image, *process_coord(**patches['Mouth']))
    mouth = transforms.functional.pad(mouth, (0, 0, 80-mouth.size[-2], 32-mouth.size[-1]))
    return left_eye, right_eye, nose, mouth

class CustomDataset(Dataset):
    def __init__(self, images_list_selected, images_list_all, images_dir):
        super(CustomDataset, self).__init__()
        self.images_list_selected = images_list_selected
        self.images_list = images_list_all
        self.images_dir = images_dir
        self.keys = list(self.images_list_selected)
        
    def __len__(self):
        return len(self.images_list_selected)
    
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
            if (k == 'id' or k == 'patches'):
                continue
            batch[k] = toTensor(batch[k])
            #print('{} : {}'.format(k, batch[k].shape))
        
        return batch

def createDataset(images_list, images_dir, p_test=0.1):
    with open(images_list, 'r') as rf:
        images_list_all = yaml.safe_load(rf.read())
    
    images_list_test = list()
    images_list_train = list()

    if p_test == 1:
        for k in images_list_all.keys():
            if images_list_all[k]['img'] != images_list_all[k]['imgGT']:
                images_list_test.append(k)
    else:
        nb_total = len(images_list_all)
        counter = 0
        
        for k in images_list_all.keys():
            if counter < nb_total*p_test:
                images_list_test.append(k)
                counter += 1
            else:
                images_list_train.append(k)
    
    
    return CustomDataset(images_list_train, images_list_all, images_dir), CustomDataset(images_list_test, images_list_all, images_dir)

