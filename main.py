import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
from model_utils import Discriminator,Generator

#hyperparameters
lr = 0.0002
batch_size = 64
image_size = 64 
channel_img = 1
channel_noise = 256
num_epochs = 10

features_d = 16
features_g = 16

my_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
])

dataset = datasets.MNIST(root = r"E:\ML paper implementation\DCGAN\data",train = False, transform=my_transforms,download=False)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netD = Discriminator(channel_img,features_d).to(device)
netG = Generator(channel_noise,channel_img,features_g).to(device)

optimizerD = optim.Adam(netD.parameters(),lr = lr,betas=(0.5,0.999))
optimizerG = optim.Adam(netG.parameters(),lr = lr,betas=(0.5,0.999))

netG.train()
netD.train()

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

fixed_noise = torch.randn(batch_size, channel_noise, 1, 1).to(device)
writer_real = SummaryWriter(r"E:\ML paper implementation\DCGAN\tensorboard write\test_real")
writer_fake = SummaryWriter(r"E:\ML paper implementation\DCGAN\tensorboard write\test_fake")
print("Starting training....")

for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(dataloader):
        #train Discriminator max log(D(x)) + log(1 - G(D(z)))
        data = data.to(device)
        targets = targets.to(device)
        batch_size = data.shape[0]

        netD.zero_grad()
        label = (torch.ones(batch_size)*0.9).to(device) # *0.9 is a hack
        output = netD(data).reshape(-1)
        lossD_real = criterion(output,label)
        D_x = output.mean().item()
        
        noise = torch.randn(batch_size,channel_noise,1,1).to(device)
        fake = netG(noise)
        label = (torch.ones(batch_size)*0.1).to(device)

        output = netD(fake.detach()).reshape(-1)
        lossD_fake = criterion(output,label)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        #train Generator maximize log(G(D(x)))

        netG.zero_grad()
        label = torch.ones(batch_size).to(device)
        output = netD(fake).reshape(-1)

        lossG = criterion(output,label)
        lossG.backward()
        optimizerG.step()

        if batch_idx % 100 == 0:
            print(f"{epoch}/{num_epochs} Batch {batch_idx}/{len(dataloader)} LossD : {lossD} LossG : {lossG}")
        
            with torch.no_grad():
                fake = netG(fixed_noise)
                
                img_grid_real = torchvision.utils.make_grid(data[:32],normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32],normalize=True)

                writer_real.add_image("Mnist real images",img_grid_real)
                writer_real.add_image("Mnist fake images",img_grid_fake)    

torch.save(netG, r"E:\ML paper implementation\DCGAN\saved_models\Generator_model.bin")
torch.save(netD, r"E:\ML paper implementation\DCGAN\saved_models\Discriminator_model.bin")
