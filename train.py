import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from Dataset import createDataset
from Network import Generator, Discriminator
from Loss import LossGenerator, LossDiscriminator
from config import settings
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os

if __name__ == "__main__":

    print('Starting...')

    trainSet, testSet = createDataset(settings['images_list'], settings['images_dir'])
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size = settings['batch_size'], shuffle = True, num_workers = 2, pin_memory = True)
    testloader = torch.utils.data.DataLoader(testSet, batch_size = 1, shuffle = True, num_workers = 2, pin_memory = True)

    print('Dataset initialized')

    device = torch.device(settings['device'])

    if device == "cpu":
        G = Generator(noise_dim = 64, num_classes = 100)
        D = Discriminator()
    else:
        G = torch.nn.DataParallel(Generator(noise_dim = 64, num_classes = 100)).cuda()
        D = torch.nn.DataParallel(Discriminator()).cuda()
        
    print('Network created')

    optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), lr = 1e-4)
    optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr = 1e-4)

    print('Optimizer created')

    #G.module.load_state_dict(torch.load('model/model_generator_1.pth'))
    #D.module.load_state_dict(torch.load('model/model_discriminator_1.pth'))
    #optimizer_G.load_state_dict(torch.load('opt/opt_generator_1.pth'))
    #optimizer_D.load_state_dict(torch.load('opt/opt_discriminator_1.pth'))
    #print('Finished loading checkpoints')

    loss_G = LossGenerator()
    loss_D = LossDiscriminator()

    print('Loss function created')

    print('Starting training...')

    loss_D_histo = list()
    loss_G_histo = list()
    img_fake_histo = list()

    for epoch in tqdm(range(settings['nb_epoch'])):
        for batch in tqdm(trainloader):
            
            noise = torch.FloatTensor(np.random.normal(0,0.02,(len(batch['img128']), 64))).to(device)
            img128_fake, img64_fake, img32_fake, encoder_predict, local_fake, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, local_GT = \
                G(batch['img128'], batch['img64'], batch['img32'], batch['left_eye'], batch['right_eye'], batch['nose'], batch['mouth'], noise)

            for parm in D.parameters():
                parm.requires_grad = True

            optimizer_D.zero_grad()
            LD = loss_D(D, img128_fake, batch)
            LD.backward()
            optimizer_D.step()

            for parm in D.parameters():
                parm.requires_grad = False

            optimizer_G.zero_grad()
            LG, _ = loss_G(G, D, img128_fake, img64_fake, img32_fake, encoder_predict, local_fake, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, local_GT, batch)
            LG.backward()
            optimizer_G.step()

        print("Epoch {}/{} finished".format(epoch+1, settings['nb_epoch']))
        print("Starting testing")
        
        G.eval()
        D.eval()
        
        loss_test_D = 0
        loss_test_G = dict()
        for batch in tqdm(testloader):
            
            noise = torch.FloatTensor(np.random.normal(0,0.02,(len(batch['img128']), 64))).to(device)
            img128_fake, img64_fake, img32_fake, encoder_predict, local_fake, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, local_GT = \
                G(batch['img128'], batch['img64'], batch['img32'], batch['left_eye'], batch['right_eye'], batch['nose'], batch['mouth'], noise)

            LD = loss_D(D, img128_fake, batch)
            LG, losses = loss_G(G, D, img128_fake, img64_fake, img32_fake, encoder_predict, local_fake, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, local_GT, batch)

            loss_test_D += LD.detach()
            
            if loss_test_G == {}:
                for k in losses.keys():
                    try:
                        loss_test_G[k] = losses[k].detach()
                    except:
                        loss_test_G[k] = losses[k]
            else:
                for k in losses.keys():
                    try:
                        loss_test_G[k] += losses[k].detach()
                    except:
                        loss_test_G[k] += losses[k]
            
        loss_D_histo.append(loss_test_D/len(testSet))
        for k in loss_test_G.keys():
                    loss_test_G[k] = loss_test_G[k]/len(testSet)
        loss_G_histo.append(loss_test_G.copy())

        batch = testSet[0]
        for k in batch.keys():
            if k == 'id':
                continue
            batch[k] = batch[k].reshape((1, *batch[k].shape))
        noise = torch.FloatTensor(np.random.normal(0,0.02,(len(batch['img128']), 64))).to(device)
        img128_fake, img64_fake, img32_fake, encoder_predict, local_fake, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, local_GT = \
                G(batch['img128'], batch['img64'], batch['img32'], batch['left_eye'], batch['right_eye'], batch['nose'], batch['mouth'], noise)
        
        toPIL = toPIL = transforms.ToPILImage()
        img_fake_histo.append({'input': toPIL(batch['img128'].detach().cpu().reshape(*batch['img128'].shape[1:])), 
                                'fake': toPIL(img128_fake.detach().cpu().reshape(*img128_fake.shape[1:])), 
                                'GT': toPIL(batch['img128GT'].detach().cpu().reshape(*batch['img128GT'].shape[1:])), 
                                'local': toPIL(local_fake.detach().cpu().reshape(*local_fake.shape[1:]))})
        
        images = img_fake_histo[-1]

        fig=plt.figure(figsize=(16, 4))
        columns = 4
        rows = 1
        img = images['input']
        fig.add_subplot(rows, columns, 1)
        plt.imshow(img)
        img = images['fake']
        fig.add_subplot(rows, columns, 2)
        plt.imshow(img)
        img = images['local']
        fig.add_subplot(rows, columns, 3)
        plt.imshow(img)
        img = images['GT']
        fig.add_subplot(rows, columns, 4)
        plt.imshow(img)
        plt.tight_layout()
        plt.show()
        
        print("End of testing")
        
        G.train()
        D.train()
        
        torch.save( G.module.state_dict() , settings['generator_path'].format(epoch%2))
        torch.save( D.module.state_dict() , settings['discriminator_path'].format(epoch%2))
        torch.save( optimizer_G.state_dict() ,settings['opt_G_path'].format(epoch%2))
        torch.save( optimizer_D.state_dict() , settings['opt_D_path'].format(epoch%2))
            
    with open(os.path.join(settings['histo'], 'loss_D'), 'wb') as f:
        pickle.dump(loss_D_histo, f)
    with open(os.path.join(settings['histo'], 'loss_G'), 'wb') as f:
        pickle.dump(loss_G_histo, f)
    with open(os.path.join(settings['histo'], 'img_fake'), 'wb') as f:
        pickle.dump(img_fake_histo, f)
            



            

