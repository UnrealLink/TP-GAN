import torch
import torch.nn as nn
from torch.autograd import Variable

class LossGenerator(nn.Module):
    def __init__(self):
        super(LossGenerator, self).__init__()
        self.L1Loss = nn.L1Loss().cuda()
        self.MSELoss = nn.MSELoss().cuda()
        self.CrossEntropy = nn.CrossEntropyLoss().cuda()
        #self.ExtractFeatures = nn.DataParallel(feature_extract_model).cuda() # TODO
    
    def _pixelwise_loss_global(self, img128_fake, img64_fake, img32_fake, batch):
        l128 = self.L1Loss(img128_fake.cpu(), batch['img128GT'])
        l64 = self.L1Loss(img64_fake.cpu(), batch['img64GT'])
        l32 = self.L1Loss(img32_fake.cpu(), batch['img32GT'])
        return (l128 + l64 + 1.5*l32).cuda()
    
    def _pixelwise_loss_local(self, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, batch):
        lle = self.L1Loss(left_eye_fake.cpu(), batch['left_eyeGT'])
        lre = self.L1Loss(right_eye_fake.cpu(), batch['right_eyeGT'])
        ln  = self.L1Loss(nose_fake.cpu(), batch['noseGT'])
        lm  = self.L1Loss(mouth_fake.cpu(), batch['mouthGT'])
        return (lle + lre + ln + lm).cuda()
    
    def pixelwise_loss(self, img128_fake, img64_fake, img32_fake, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, batch):
        global_loss = self._pixelwise_loss_global(img128_fake, img64_fake, img32_fake, batch)
        local_loss  = self._pixelwise_loss_local(left_eye_fake, right_eye_fake, nose_fake, mouth_fake, batch)
        return global_loss + 3*local_loss
    
    def symmetry_loss(self, img128_fake, img64_fake, img32_fake):
        img128_fake_mirror = img128_fake.index_select(3, torch.arange(img128_fake.size()[3]-1, -1, -1).long().cuda())
        img128_fake_mirror.detach_()
        img64_fake_mirror = img64_fake.index_select(3, torch.arange(img64_fake.size()[3]-1, -1, -1).long().cuda())
        img64_fake_mirror.detach_()
        img32_fake_mirror = img32_fake.index_select(3, torch.arange(img32_fake.size()[3]-1, -1, -1).long().cuda())
        img32_fake_mirror.detach_()
        symloss128 = self.L1Loss(img128_fake, img128_fake_mirror)
        symloss64 = self.L1Loss(img64_fake, img64_fake_mirror)
        symloss32 = self.L1Loss(img32_fake, img32_fake_mirror)
        return symloss128 + symloss64 + 1.5*symloss32

    def adversarial_loss(self, D, img128_fake):
        return - torch.mean(D(img128_fake))
    
    def identity_preserving_loss(self, img128_fake, batch):
        #feature_GT, fc_GT = self.ExtractFeatures(batch['img128GT'])
        #feature_fake, fc_fake = self.ExtractFeatures(img128_fake)
        #return = self.MSELoss(feature_fake, feature_GF.detach())
        # TO DO
        return 0
    
    def total_variation_loss(self, img128_fake):
        return torch.mean(torch.abs(img128_fake[:,:,:-1,:] - img128_fake[:,:,1:,:])) + torch.mean(torch.abs(img128_fake[:,:,:,:-1] - img128_fake[:,:,:,1:]))
    
    def cross_entropy_loss(self, encoder_predict, batch):
        return self.CrossEntropy(encoder_predict.cpu(), batch['id']-1).cuda()
    
    def forward(self, G, D, img128_fake, img64_fake, img32_fake, encoder_predict, local_fake, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, local_GT, batch):
        pw_loss  = self.pixelwise_loss(img128_fake, img64_fake, img32_fake, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, batch)
        sym_loss = self.symmetry_loss(img128_fake, img64_fake, img32_fake)
        adv_loss = self.adversarial_loss(D, img128_fake)
        ip_loss  = self.identity_preserving_loss(img128_fake, batch)
        tv_loss  = self.total_variation_loss(img128_fake)
        L_syn    = pw_loss + 0.3*sym_loss + 0.001*adv_loss + 0.003*ip_loss + 0.0001*tv_loss
        ce_loss  = self.cross_entropy_loss(encoder_predict, batch)
        return L_syn + 0.1*ce_loss
    
    
class LossDiscriminator(nn.Module):
    def __init__(self):
        super(LossDiscriminator, self).__init__()
        
    def forward(self, D, img128_fake, batch):
        adv_D_loss = torch.mean(D(img128_fake.detach())) - torch.mean(D(batch['img128GT']))
        alpha = torch.rand(batch['img128GT'].shape[0] , 1 , 1 , 1 ).expand_as(batch['img128GT']).pin_memory().cuda()
        interpolated_x = Variable(alpha * img128_fake.detach().data.cuda() + (1.0 - alpha) * batch['img128GT'].data.cuda(), requires_grad = True) 
        out = D(interpolated_x)
        dxdD = torch.autograd.grad(outputs = out, inputs = interpolated_x, grad_outputs = torch.ones(out.size()).cuda(), retain_graph = True, create_graph = True, only_inputs = True)[0].view(out.shape[0],-1)
        gp_loss = torch.mean((torch.norm(dxdD, p = 2) - 1)**2)
        return adv_D_loss + 10*gp_loss
