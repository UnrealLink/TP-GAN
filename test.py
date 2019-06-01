import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from Dataset import createDataset
from Network import Generator, Discriminator
from Loss import LossGenerator, LossDiscriminator
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os

def parse_args():
    parser = argparse.ArgumentParser(description='''Show the output of trained TP-GAN on the input images.
                                                    Input is not sanitized, please be nice. ''')
    parser.add_argument('-l', '--img-list', type=str, default='image_list_reduced.yml', help='yaml file of input processed input images')
    parser.add_argument('-d', '--img-dir', type=str, default='put_cleaned', help='directory of processed input images')
    parser.add_argument('-m', '--model', type=str, default='model_generator_20.pth', help='path to generator checkpoint')
    parser.add_argument('-o', '--output', type=str, default='image.png', help='path to save image output')
    parser.add_argument('-c', action='store_true', default=False, help='cpu only (no cuda)')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    print('Starting...')

    _, testSet = createDataset(args.img_list, args.img_dir, 1)
    testloader = torch.utils.data.DataLoader(testSet, batch_size = 1, shuffle = False, num_workers = 1, pin_memory = True)

    print('Dataset initialized')
    if not(args.c):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.c:
        G = Generator(noise_dim = 64, num_classes = 100)
        G.load_state_dict(torch.load(args.model))
    else:
        G = torch.nn.DataParallel(Generator(noise_dim = 64, num_classes = 100)).to(device)
        G.module.load_state_dict(torch.load(args.model))
        
    print('Network created')

    

    print('Finished loading checkpoints')

    G.eval()

    img_list = list()
    toPIL = transforms.ToPILImage()
    

    for batch in tqdm(testloader):
            
        noise = torch.FloatTensor(np.random.normal(0,0.02,(len(batch['img128']), 64))).to(device)
        img128_fake, img64_fake, img32_fake, encoder_predict, local_fake, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, local_GT = \
            G(batch['img128'], batch['img64'], batch['img32'], batch['left_eye'], batch['right_eye'], batch['nose'], batch['mouth'], noise)

        img_list.append({'input': toPIL(batch['img128'].detach().cpu().reshape(*batch['img128'].shape[1:])), 
                            'fake': toPIL(img128_fake.detach().cpu().reshape(*img128_fake.shape[1:])), 
                            'GT': toPIL(batch['img128GT'].detach().cpu().reshape(*batch['img128GT'].shape[1:])), 
                            'local': toPIL(local_fake.detach().cpu().reshape(*local_fake.shape[1:]))})

    
    columns = 4
    rows = min(10, len(img_list))
    fig=plt.figure(figsize=(16, 4 * rows))
    for i in range(rows):
        images = img_list[i]
        img = images['input']
        fig.add_subplot(rows, columns, 1 + 4*i)
        plt.imshow(img)
        img = images['fake']
        fig.add_subplot(rows, columns, 2 + 4*i)
        plt.imshow(img)
        img = images['local']
        fig.add_subplot(rows, columns, 3 + 4*i)
        plt.imshow(img)
        img = images['GT']
        fig.add_subplot(rows, columns, 4 + 4*i)
        plt.imshow(img)
    plt.tight_layout()
    if args.output != '':
        try:
            fig.savefig(args.output)
        except Exception as e:
            print("Couldn't save figure : {}".format(e))
    plt.show()