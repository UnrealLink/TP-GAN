import torch
import torch.nn as nn
import torchvision
import numpy as np
from Dataset import TrainingSet
from Network import Generator, Discriminator
from Loss import LossGenerator, LossDiscriminator
from config import settings

if __name__ == "__main__":

    print('Starting...')

    trainSet = TrainingSet(settings['images_list'], settings['images_dir'])
    dataloader = torch.utils.data.DataLoader(trainSet, batch_size = settings['batch_size'], shuffle = True, num_workers = 2, pin_memory = True)
    
    print('Dataset initialized')

    device = torch.device(settings['device'])

    if device == "cpu":
        G = Generator(noise_dim = 64, num_classes = 100)
        D = Discriminator()
    else:
        G = torch.nn.DataParallel(Generator(noise_dim = 64, num_classes = 100)).cuda()
        D = torch.nn.DataParallel(Discriminator()).cuda()

    print('Network created')

    optimizer_G = torch.optim.SGD(filter(lambda p: p.requires_grad, G.parameters()), lr = 1e-4)
    optimizer_D = torch.optim.SGD(filter(lambda p: p.requires_grad, D.parameters()), lr = 1e-4)

    print('Optimizer created')

    loss_G = LossGenerator()
    loss_D = LossDiscriminator()

    print('Loss function created')

    for step, batch in enumerate(dataloader):

        noise = torch.FloatTensor(np.random.uniform(-1,1,(len(batch['img128']), 64))).to(device)
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
        LG = loss_G(G, D, img128_fake, img64_fake, img32_fake, encoder_predict, local_fake, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, local_GT, batch)
        LG.backward()
        optimizer_G.step()

        print('End of batch')



        
