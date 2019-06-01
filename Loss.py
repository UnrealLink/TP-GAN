import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms.functional as tf
from config import settings
from LightCNN import LightCNN_29Layers_v2


device = settings['device']

class LossGenerator(nn.Module):
    def __init__(self):
        super(LossGenerator, self).__init__()
        self.L1Loss = nn.L1Loss().to(device)
        self.MSELoss = nn.MSELoss().to(device)
        self.CrossEntropy = nn.CrossEntropyLoss().to(device)
        self.ExtractFeatures = torch.nn.DataParallel(LightCNN_29Layers_v2(num_classes=80013)).to(settings['device'])
        self.ExtractFeatures.load_state_dict(torch.load(settings['light_cnn'])['state_dict'])
    
    def _pixelwise_loss_global(self, img128_fake, img64_fake, img32_fake, batch):
        l128 = self.L1Loss(img128_fake.cpu(), batch['img128GT'])
        l64 = self.L1Loss(img64_fake.cpu(), batch['img64GT'])
        l32 = self.L1Loss(img32_fake.cpu(), batch['img32GT'])
        return (l128 + l64 + 1.5*l32).to(device)
    
    def _pixelwise_loss_local(self, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, batch):
        lle = self.L1Loss(left_eye_fake.cpu(), batch['left_eyeGT'])
        lre = self.L1Loss(right_eye_fake.cpu(), batch['right_eyeGT'])
        ln  = self.L1Loss(nose_fake.cpu(), batch['noseGT'])
        lm  = self.L1Loss(mouth_fake.cpu(), batch['mouthGT'])
        return (lle + lre + ln + lm).to(device)
    
    def _cut(self, img, patch, name):
        '''
        Patch : max_w x max_h
        Left eye : 44x22
        Right eye : 44x22
        Nose : 46x66
        Mouth : 69x25
        '''
        if name == 'left_eye' or name == 'right_eye':
            return torch.cat([img[i,:, int(patch['y'][i]):int(patch['y'][i])+22, int(patch['x'][i]):int(patch['x'][i])+28] for i in range(img.shape[0])], 0)
        if name == 'nose':
            return torch.cat([img[i,:, int(patch['y'][i]):int(patch['y'][i])+66, int(patch['x'][i]):int(patch['x'][i])+46] for i in range(img.shape[0])], 0)
        if name == 'mouth':
            return torch.cat([img[i,:, int(patch['y'][i]):int(patch['y'][i])+25, int(patch['x'][i]):int(patch['x'][i])+54] for i in range(img.shape[0])], 0)
    
    def _pixelwise_loss_local2(self, img128_fake, batch):
        lle = self.L1Loss(self._cut(img128_fake.cpu(), batch['patches']['left_eye'], 'left_eye'),
                          self._cut(batch['img128GT'].cpu(), batch['patches']['left_eye'], 'left_eye'))
        lre = self.L1Loss(self._cut(img128_fake.cpu(), batch['patches']['right_eye'], 'right_eye'),
                          self._cut(batch['img128GT'].cpu(), batch['patches']['right_eye'], 'right_eye'))
        ln  = self.L1Loss(self._cut(img128_fake.cpu(), batch['patches']['nose'], 'nose'),
                          self._cut(batch['img128GT'].cpu(), batch['patches']['nose'], 'nose'))
        lm  = self.L1Loss(self._cut(img128_fake.cpu(), batch['patches']['mouth'], 'mouth'),
                          self._cut(batch['img128GT'].cpu(), batch['patches']['mouth'], 'mouth'))
        return (lle + lre + ln + lm).to(device)
    
    def pixelwise_loss(self, img128_fake, img64_fake, img32_fake, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, batch):
        global_loss = self._pixelwise_loss_global(img128_fake, img64_fake, img32_fake, batch)
        local_loss  = self._pixelwise_loss_local(left_eye_fake, right_eye_fake, nose_fake, mouth_fake, batch)
        local_loss2 = self._pixelwise_loss_local2(img128_fake, batch)
        return global_loss + 3*local_loss + 3*local_loss2
    
    def symmetry_loss(self, img128_fake, img64_fake, img32_fake):
        img128_fake_mirror = img128_fake.index_select(3, torch.arange(img128_fake.size()[3]-1, -1, -1).long().to(device))
        img128_fake_mirror.detach_()
        img64_fake_mirror = img64_fake.index_select(3, torch.arange(img64_fake.size()[3]-1, -1, -1).long().to(device))
        img64_fake_mirror.detach_()
        img32_fake_mirror = img32_fake.index_select(3, torch.arange(img32_fake.size()[3]-1, -1, -1).long().to(device))
        img32_fake_mirror.detach_()
        symloss128 = self.L1Loss(img128_fake, img128_fake_mirror)
        symloss64 = self.L1Loss(img64_fake, img64_fake_mirror)
        symloss32 = self.L1Loss(img32_fake, img32_fake_mirror)
        return symloss128 + symloss64 + 1.5*symloss32

    def adversarial_loss(self, D, img128_fake):
        return - torch.mean(D(img128_fake))
    
    def identity_preserving_loss(self, img128_fake, batch):
        _, feat_fake = self.ExtractFeatures((img128_fake[:,0,:,:]*0.2126 + img128_fake[:,0,:,:]*0.7152 + img128_fake[:,0,:,:]*0.0722).view(img128_fake.shape[0], 1, img128_fake.shape[2], img128_fake.shape[3]))
        _, feat_GT = self.ExtractFeatures((batch['img128GT'][:,0,:,:]*0.2126 + batch['img128GT'][:,0,:,:]*0.7152 + batch['img128GT'][:,0,:,:]*0.0722).view(batch['img128GT'].shape[0], 1, batch['img128GT'].shape[2], batch['img128GT'].shape[3]))
        return self.L1Loss(feat_fake, feat_GT)
    
    def total_variation_loss(self, img128_fake):
        return torch.mean(torch.abs(img128_fake[:,:,:-1,:] - img128_fake[:,:,1:,:])) + torch.mean(torch.abs(img128_fake[:,:,:,:-1] - img128_fake[:,:,:,1:]))
    
    def cross_entropy_loss(self, encoder_predict, batch):
        return self.CrossEntropy(encoder_predict.cpu(), batch['id']-1).to(device)
    
    def forward(self, G, D, img128_fake, img64_fake, img32_fake, encoder_predict, local_fake, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, local_GT, batch):
        pw_loss  = self.pixelwise_loss(img128_fake, img64_fake, img32_fake, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, batch)
        sym_loss = self.symmetry_loss(img128_fake, img64_fake, img32_fake)
        adv_loss = self.adversarial_loss(D, img128_fake)
        ip_loss  = self.identity_preserving_loss(img128_fake, batch)
        tv_loss  = self.total_variation_loss(img128_fake)
        L_syn    = pw_loss + 0.3*sym_loss + 0.001*adv_loss + 0.003*ip_loss + 0.0001*tv_loss
        ce_loss  = self.cross_entropy_loss(encoder_predict, batch)
        return L_syn + 0.1*ce_loss, {'pw_loss':pw_loss, 'sym_loss':0.3*sym_loss, 'adv_loss':0.001*adv_loss, 'ip_loss':0.003*ip_loss, 'tv_loss':0.0001*tv_loss, 'L_syn':L_syn, 'ce_loss':0.1*ce_loss, 'total':L_syn + 0.1*ce_loss}
    
    
class LossDiscriminator(nn.Module):
    def __init__(self):
        super(LossDiscriminator, self).__init__()
        
    def forward(self, D, img128_fake, batch):
        adv_D_loss = torch.mean(D(img128_fake.detach())) - torch.mean(D(batch['img128GT']))
        alpha = torch.rand(batch['img128GT'].shape[0] , 1 , 1 , 1 ).expand_as(batch['img128GT']).pin_memory().to(device)
        interpolated_x = Variable(alpha * img128_fake.detach().data.to(device) + (1.0 - alpha) * batch['img128GT'].data.to(device), requires_grad = True) 
        out = D(interpolated_x)
        dxdD = torch.autograd.grad(outputs = out, inputs = interpolated_x, grad_outputs = torch.ones(out.size()).to(device), retain_graph = True, create_graph = True, only_inputs = True)[0].view(out.shape[0],-1)
        gp_loss = torch.mean((torch.norm(dxdD, p = 2) - 1)**2)
        return adv_D_loss + 10*gp_loss
