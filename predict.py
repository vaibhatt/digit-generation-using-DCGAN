import torch
from model_utils import Discriminator,Generator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#HP
channel_img = 1
features_d = 16
channel_noise = 256
features_g = 16
torch.random.seed()
generator_model_path = r"E:\ML paper implementation\DCGAN\saved_models\Generator_model.bin"
discriminator_model_path = r"E:\ML paper implementation\DCGAN\saved_models\Discriminator_model.bin"
noise = torch.randn(1,channel_noise,1,1)
'''
netD = Discriminator(channel_img,features_d)
netG = Generator(channel_noise,channel_img,features_g)
torch.load_state_dict(torch.load(generator_model_path))
'''
netG = torch.load(generator_model_path)
output_img = netG(noise)
img_array = np.array(output_img[0,0,:,:].detach())
img_array = Image.fromarray(img_array,'L')
img_array.save(r"E:\ML paper implementation\DCGAN\predicted\sample1.jpg")
