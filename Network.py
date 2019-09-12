import torch
import torch.nn as nn
from Blocks import ConvBlock, DeConvBlock, ResidualBlock, Sequential

class LocalPathway(nn.Module):
    def __init__(self):
        super(LocalPathway, self).__init__()
        channel_encoder = [64, 128, 256, 512]
        channel_decoder = [256, 128, 64]
        ## encoder blocks
        self.conv0 = Sequential(ConvBlock(
                                       in_channels = 3, out_channels = channel_encoder[0], 
                                       kernel_size = 3, stride = 1, padding = 1, 
                                       activation = nn.LeakyReLU(), use_batchnorm = True),
                                   ResidualBlock(
                                       channel_encoder[0], activation = nn.LeakyReLU()))
        
        self.conv1 = Sequential(ConvBlock(
                                       in_channels = channel_encoder[0], out_channels = channel_encoder[1], 
                                       kernel_size = 3, stride = 2, padding = 1, 
                                       activation = nn.LeakyReLU(), use_batchnorm = True),
                                   ResidualBlock(
                                       channel_encoder[1], activation = nn.LeakyReLU()))
        
        self.conv2 = Sequential(ConvBlock(
                                       in_channels = channel_encoder[1], out_channels = channel_encoder[2], 
                                       kernel_size = 3, stride = 2, padding = 1, 
                                       activation = nn.LeakyReLU(), use_batchnorm = True),
                                   ResidualBlock(
                                       channel_encoder[2], activation = nn.LeakyReLU()))
        
        self.conv3 = Sequential(ConvBlock(
                                       in_channels = channel_encoder[2], out_channels = channel_encoder[3], 
                                       kernel_size = 3, stride = 2, padding = 1, 
                                       activation = nn.LeakyReLU(), use_batchnorm = True),
                                   *[ResidualBlock(
                                       channel_encoder[3], activation = nn.LeakyReLU()) for i in range(2)])
        
        ## decoder blocks
        
        self.deconv0 = DeConvBlock(
                           in_channels = channel_encoder[3], out_channels = channel_decoder[0],
                           kernel_size = 3, stride = 2, padding = 1, output_padding = 1,
                           activation = nn.ReLU(), use_batchnorm = True)
        
        self.decode0 = Sequential(ConvBlock(
                                         in_channels = channel_decoder[0] + self.conv2.out_channels, out_channels = channel_decoder[0], 
                                         kernel_size = 3, stride = 1, padding = 1, 
                                         activation = nn.LeakyReLU(), use_batchnorm = True),
                                     ResidualBlock(
                                         channel_decoder[0], activation = nn.LeakyReLU()))
        
        self.deconv1 = DeConvBlock(
                           in_channels = channel_decoder[0], out_channels = channel_decoder[1],
                           kernel_size = 3, stride = 2, padding = 1, output_padding = 1,
                           activation = nn.ReLU(), use_batchnorm = True)
        
        self.decode1 = Sequential(ConvBlock(
                                         in_channels = channel_decoder[1] + self.conv1.out_channels, out_channels = channel_decoder[1], 
                                         kernel_size = 3, stride = 1, padding = 1, 
                                         activation = nn.LeakyReLU(), use_batchnorm = True),
                                     ResidualBlock(
                                         channel_decoder[1], activation = nn.LeakyReLU()))
        
        self.deconv2 = DeConvBlock(
                           in_channels = channel_decoder[1], out_channels = channel_decoder[2],
                           kernel_size = 3, stride = 2, padding = 1, output_padding = 1,
                           activation = nn.ReLU(), use_batchnorm = True)
        
        self.decode2 = Sequential(ConvBlock(
                                         in_channels = channel_decoder[2] + self.conv0.out_channels, out_channels = channel_decoder[2], 
                                         kernel_size = 3, stride = 1, padding = 1, 
                                         activation = nn.LeakyReLU(), use_batchnorm = True),
                                     ResidualBlock(
                                         channel_decoder[2], activation = nn.LeakyReLU()))
        
        self.output_local = ConvBlock(
                                in_channels = channel_decoder[2], out_channels = 3,
                                kernel_size = 1, stride = 1, padding = 0,
                                activation = nn.Tanh(), use_batchnorm = False)
        
        
    def forward(self, x):
        ## encoding
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        ## decoding
        deconv0 = self.deconv0(conv3)
        decode0 = self.decode0(torch.cat([deconv0, conv2],  1))
        deconv1 = self.deconv1(decode0)
        decode1 = self.decode1(torch.cat([deconv1, conv1] , 1))
        deconv2 = self.deconv2(decode1)
        decode2 = self.decode2(torch.cat([deconv2, conv0],  1))
        
        output_local = self.output_local(decode2)
        
        assert output_local.shape == x.shape, "{} {}".format(output_local.shape , x.shape)
        return output_local , decode2
        

class LocalFuser(nn.Module):
    def __init__(self):
        super(LocalFuser, self).__init__()
        """ 
        Left eye :
        x : 75.62766569186905, y : 38.04095867299554
        Right eye :
        x : 20.889541067281183, y : 38.579423812532795
        Nose :
        x : 47.32572848751807, y : 36.85505589786052
        Mouth :
        x : 41.847959186221765, y : 86.41842353875073
        """        
        
    def forward(self, left_eye, right_eye, nose, mouth):
        p_left_eye = torch.nn.functional.pad(left_eye, (76, 128 - (76 + left_eye.shape[-1]), 38, 128 - (38 + left_eye.shape[-2])))
        p_right_eye = torch.nn.functional.pad(right_eye, (21, 128 - (21 + right_eye.shape[-1]), 38, 128 - (38 + right_eye.shape[-2])))
        p_nose = torch.nn.functional.pad(nose, (47, 128 - (47 + nose.shape[-1]), 37, 128 - (37 + nose.shape[-2])))
        p_mouth = torch.nn.functional.pad(mouth, (42, 128 - (42 + mouth.shape[-1]), 86, 128 - (86 + mouth.shape[-2])))
        return torch.max(torch.stack([p_left_eye, p_right_eye, p_nose, p_mouth], dim = 0), dim = 0)[0]    
        
        
        
class GlobalPathway(nn.Module):
    def __init__(self, noise_dim = 64, local_features_dim = 64):
        super(GlobalPathway, self).__init__()
        channel_encoder = [64, 64, 128, 256, 512]
        channel_decoder_feat = [64,32,16,8] 
        channel_decoder = [512, 256, 128, 64]
        channel_decoder_conv = [64, 32]
        self.noise_dim = noise_dim
        
        ## encoder blocks
        # img size = 128x128
        self.conv0 = Sequential(ConvBlock(
                                       in_channels = 3, out_channels = channel_encoder[0],
                                       kernel_size = 7, stride = 1, padding = 3,
                                       activation = nn.LeakyReLU(), use_batchnorm = True),
                                   ResidualBlock(
                                       in_channels = channel_encoder[0], kernel_size = 7, padding = 3,
                                       activation = nn.LeakyReLU()))
        
        # img size = 64x64
        self.conv1 = Sequential(ConvBlock(
                                       in_channels = channel_encoder[0], out_channels = channel_encoder[1],
                                       kernel_size = 5, stride = 2, padding = 2,
                                       activation = nn.LeakyReLU(), use_batchnorm = True),
                                   ResidualBlock(
                                       in_channels = channel_encoder[0], kernel_size = 5, padding = 2,
                                       activation = nn.LeakyReLU()))
        
        # img size = 32x32
        self.conv2 = Sequential(ConvBlock(
                                       in_channels = channel_encoder[1], out_channels = channel_encoder[2],
                                       kernel_size = 3, stride = 2, padding = 1,
                                       activation = nn.LeakyReLU(), use_batchnorm = True),
                                   ResidualBlock(
                                       in_channels = channel_encoder[2], kernel_size = 3, padding = 1,
                                       activation = nn.LeakyReLU()))
        
        # img size = 16x16
        self.conv3 = Sequential(ConvBlock(
                                       in_channels = channel_encoder[2], out_channels = channel_encoder[3],
                                       kernel_size = 3, stride = 2, padding = 1,
                                       activation = nn.LeakyReLU(), use_batchnorm = True),
                                   ResidualBlock(
                                       in_channels = channel_encoder[3], kernel_size = 3, padding = 1,
                                       activation = nn.LeakyReLU()))
        
        # img size = 8x8
        self.conv4 = Sequential(ConvBlock(
                                       in_channels = channel_encoder[3], out_channels = channel_encoder[4],
                                       kernel_size = 3, stride = 2, padding = 1,
                                       activation = nn.LeakyReLU(), use_batchnorm = True),
                                   *[ResidualBlock(
                                       in_channels = channel_encoder[4], kernel_size = 3, padding = 1,
                                       activation = nn.LeakyReLU()) for i in range(4)])
        
        self.fc1 = nn.Linear( channel_encoder[4]*8*8 , 512)
        self.fc2 = nn.MaxPool1d( 2 , 2 , 0)
        
        ## decoder blocks
        # 8x8
        self.feat8 = DeConvBlock(
                                in_channels = channel_encoder[4]//2 + self.noise_dim, out_channels = channel_decoder_feat[0], 
                                kernel_size = 8, stride = 1, padding = 0, output_padding = 0, 
                                activation = nn.ReLU(), use_batchnorm = True)
        
        self.decode8 = ResidualBlock(
                           in_channels = self.feat8.out_channels + self.conv4.out_channels, 
                           kernel_size = 3, stride = 1, padding = 1, activation = nn.LeakyReLU())
        
        self.reconstruct8 = Sequential(*[ResidualBlock(
                                in_channels = self.feat8.out_channels + self.conv4.out_channels,
                                kernel_size = 3, stride = 1, padding = 1, activation = nn.LeakyReLU()) for i in range(2)])
        
        # 16x16
        self.deconv16 = DeConvBlock(
                            in_channels = self.reconstruct8.out_channels, out_channels = channel_decoder[0],
                            kernel_size = 3, stride = 2, padding = 1, output_padding = 1,
                            activation = nn.ReLU(), use_batchnorm = True)
        
        self.decode16 = ResidualBlock(
                            in_channels = self.conv3.out_channels, activation = nn.LeakyReLU())
        
        self.reconstruct16 = Sequential(*[ResidualBlock(
                                 in_channels = self.deconv16.out_channels + self.decode16.out_channels,
                                 activation = nn.LeakyReLU()) for i in range(2)])
        
        
        # 32x32
        self.feat32 = DeConvBlock(
                                in_channels = channel_decoder_feat[0], out_channels = channel_decoder_feat[1], 
                                kernel_size = 3, stride = 4, padding = 0, output_padding = 1, 
                                activation = nn.ReLU(), use_batchnorm = True)
        
        self.deconv32 = DeConvBlock(
                            in_channels = self.reconstruct16.out_channels, out_channels = channel_decoder[1],
                            kernel_size = 3, stride = 2, padding = 1, output_padding = 1,
                            activation = nn.ReLU(), use_batchnorm = True)
        
        self.decode32 = ResidualBlock(
                            in_channels = self.feat32.out_channels + self.conv2.out_channels + 3, activation = nn.LeakyReLU())
        
        self.reconstruct32 = Sequential(*[ResidualBlock(
                                 in_channels = self.feat32.out_channels + self.conv2.out_channels + 3 + self.deconv32.out_channels,
                                 activation = nn.LeakyReLU() ) for i in range(2)])

        self.output32 = ConvBlock(
                            in_channels = self.reconstruct32.out_channels, out_channels = 3,
                            kernel_size = 3, stride = 1, padding = 1,
                            activation = nn.Tanh(), init_weight = False)
        
        # 64x64
        self.feat64 = DeConvBlock(
                                in_channels = channel_decoder_feat[1], out_channels = channel_decoder_feat[2], 
                                kernel_size = 3, stride = 2, padding = 1, output_padding = 1, 
                                activation = nn.ReLU(), use_batchnorm = True)
        
        self.deconv64 = DeConvBlock(
                            in_channels = self.reconstruct32.out_channels, out_channels = channel_decoder[2],    
                            kernel_size = 3, stride = 2, padding = 1, output_padding = 1,
                            activation = nn.ReLU(), use_batchnorm = True)
        
        self.decode64 = ResidualBlock(
                            in_channels = self.feat64.out_channels + self.conv1.out_channels + 3, 
                            kernel_size = 5, activation = nn.LeakyReLU())
        
        self.reconstruct64 = Sequential(*[ResidualBlock(
                                 in_channels = self.feat64.out_channels + self.conv1.out_channels + 3 + self.deconv64.out_channels + 3,
                                 activation = nn.LeakyReLU() ) for i in range(2)])
        
        self.output64 = ConvBlock(
                            in_channels = self.reconstruct64.out_channels, out_channels = 3,
                            kernel_size = 3, stride = 1, padding = 1,
                            activation = nn.Tanh(), init_weight = False)
        
        # 128x128
        self.feat128 = DeConvBlock(
                                in_channels = channel_decoder_feat[2], out_channels = channel_decoder_feat[3], 
                                kernel_size = 3, stride = 2, padding = 1, output_padding = 1, 
                                activation = nn.ReLU(), use_batchnorm = True)
        
        self.deconv128 = DeConvBlock(
                             in_channels = self.reconstruct64.out_channels, out_channels = channel_decoder[3],    
                             kernel_size = 3, stride = 2, padding = 1, output_padding = 1,
                             activation = nn.ReLU(), use_batchnorm = True)
        
        self.decode128 = ResidualBlock(
                             in_channels = self.feat128.out_channels + self.conv0.out_channels + 3, kernel_size = 7, activation = nn.LeakyReLU())
        
        self.reconstruct128 = ResidualBlock(
                                  in_channels = self.feat128.out_channels + self.conv0.out_channels + 3 + self.deconv128.out_channels + 3 + local_features_dim + 3,
                                  kernel_size = 5, activation = nn.LeakyReLU())
        
        self.decode_conv0 = Sequential(ConvBlock(
                                              in_channels = self.reconstruct128.out_channels, out_channels = channel_decoder_conv[0], 
                                              kernel_size = 5, stride = 1, padding = 2,
                                              activation = nn.LeakyReLU(), use_batchnorm = True),
                                          ResidualBlock(
                                              channel_decoder_conv[0], kernel_size = 3))
        
        self.decode_conv1 = ConvBlock(
                                in_channels = channel_decoder_conv[0], out_channels = channel_decoder_conv[1],
                                kernel_size = 3, stride = 1, padding = 1, 
                                activation = nn.LeakyReLU(), use_batchnorm = True)
        
        self.output128 = ConvBlock(
                             in_channels = channel_decoder_conv[1], out_channels = 3,
                             kernel_size = 3, stride = 1, padding = 1,
                             activation = nn.Tanh(), init_weight = False)
        
    def forward(self, img128, img64, img32, local_predict, local_feature, noise):
        
        ## encoder
        conv0 = self.conv0(img128)  # 128x128
        conv1 = self.conv1(conv0) # 64x64
        conv2 = self.conv2(conv1) # 32x32
        conv3 = self.conv3(conv2) # 16x16
        conv4 = self.conv4(conv3) # 8x8
        
        fc1 = self.fc1(conv4.view(conv4.size()[0], -1))
        fc2 = self.fc2(fc1.view(fc1.size()[0], -1, 2)).view(fc1.size()[0], -1) 
        
        ## decoder
        # 8x8
        feat8        = self.feat8(torch.cat([fc2, noise], 1).view(fc2.size()[0], -1, 1, 1))
        decode8      = self.decode8(torch.cat([feat8,conv4], 1))
        reconstruct8 = self.reconstruct8(decode8)
        assert reconstruct8.shape[2] == 8

        # 16x16
        deconv16      = self.deconv16(reconstruct8)
        decode16      = self.decode16(conv3)
        reconstruct16 = self.reconstruct16(torch.cat([deconv16, decode16], 1))
        assert reconstruct16.shape[2] == 16
        
        # 32x32
        feat32        = self.feat32(feat8)
        deconv32      = self.deconv32(reconstruct16)
        decode32      = self.decode32(torch.cat([feat32, conv2, img32] ,1 ))
        reconstruct32 = self.reconstruct32(torch.cat([deconv32,decode32], 1))
        output32      = self.output32(reconstruct32)
        assert output32.shape[2] == 32

        # 64x64
        feat64        = self.feat64(feat32)
        deconv64      = self.deconv64(reconstruct32)
        decode64      = self.decode64(torch.cat([feat64, conv1, img64], 1))
        reconstruct64 = self.reconstruct64(torch.cat([deconv64, decode64, nn.functional.interpolate(output32.data, (64,64), mode='bilinear', align_corners=False)], 1))
        output64      = self.output64( reconstruct64 )
        assert output64.shape[2] == 64
        
        # 128x128
        feat128        = self.feat128(feat64)
        deconv128      = self.deconv128(reconstruct64)
        decode128      = self.decode128(torch.cat([feat128, conv0, img128], 1))
        reconstruct128 = self.reconstruct128(torch.cat([deconv128, decode128, nn.functional.interpolate(output64, (128,128), mode='bilinear', align_corners=False),  local_feature, local_predict], 1))
        decode_conv0   = self.decode_conv0(reconstruct128)
        decode_conv1   = self.decode_conv1(decode_conv0)
        output128      = self.output128(decode_conv1)
        
        return output128, output64, output32, fc2
    
    
class FeaturePredict(nn.Module):
    def __init__(self, num_classes, global_feature_layer_dim=256, dropout=0.3):
        super(FeaturePredict, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(global_feature_layer_dim, num_classes)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Generator(nn.Module) :
    
    def __init__(self, noise_dim, num_classes) :
        
        super(Generator, self).__init__()
        
        self.path_left_eye = LocalPathway()
        self.path_right_eye = LocalPathway()
        self.path_nose = LocalPathway()
        self.path_mouth = LocalPathway()
        
        self.globalpath = GlobalPathway(noise_dim)
        self.fuser = LocalFuser()
        self.feature_predict = FeaturePredict(num_classes)
        
    def forward(self, img128, img64, img32, left_eye, right_eye, nose, mouth, noise):
        
        # Local Path
        
        left_eye_fake, left_eye_fake_features = self.path_left_eye(left_eye)
        right_eye_fake, right_eye_fake_features = self.path_right_eye(right_eye)
        nose_fake, nose_fake_features = self.path_nose(nose)
        mouth_fake, mouth_fake_features = self.path_mouth(mouth)
        
        # Merge local path
        
        local_features = self.fuser(left_eye_fake_features, right_eye_fake_features, nose_fake_features, mouth_fake_features)
        local_fake = self.fuser(left_eye_fake, right_eye_fake, nose_fake, mouth_fake)
        local_GT = self.fuser(left_eye, right_eye, nose, mouth)
        
        # Global Path
        
        img128_fake, img64_fake, img32_fake, fc2 = self.globalpath(img128, img64, img32, local_fake, local_features, noise)
        encoder_predict = self.feature_predict(fc2)
        
        return img128_fake, img64_fake, img32_fake, encoder_predict, local_fake, left_eye_fake, right_eye_fake, nose_fake, mouth_fake, local_GT
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        layers = []
        channel_output  = [3,64,128,256,512,512] 
        for i in range(5):
            layers.append(ConvBlock(
                              in_channels = channel_output[i], out_channels = channel_output[i+1],
                              kernel_size = 3, stride = 2, padding = 1,
                              activation = nn.LeakyReLU(), use_batchnorm = True))
            if i >= 3:
                layers.append(ResidualBlock(channel_output[i+1], activation = nn.LeakyReLU()))
    
        layers.append(ConvBlock(
                          in_channels = channel_output[-1], out_channels = 1, 
                          kernel_size = 3, stride = 1, padding = 1, 
                          activation = None, init_weight = False))
                          
        self.model = Sequential(*layers)

    def forward(self,x):
        return self.model(x)