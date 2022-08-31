import torch 
import torch.nn as nn
import numpy





def weight_init(m):
    submodule_name = m.__class__.__name__
    if submodule_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif submodule_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
        Generator for the images
        input is nz-dim random noise 
        out put is an image of size nc * 28 * 28
    
    """
    def __init__(self, dim_z, ngf, nc):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(

            # nz * 1 * 1
            nn.ConvTranspose2d(dim_z, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf * 8) * 4 * 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True), 
            # (ngf * 4) * 8 * 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf * 2) * 16 * 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf ) * 32 * 32

            ##################################
            nn.Conv2d(ngf, ngf, 4, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf),
            nn.Conv2d(ngf, nc, 2, 1, bias=False),
            nn.Tanh()
            ## (nc) * 28 * 28
            
    
        )
    def forward(self, x):
        return self.layers(x)



class Discriminator(nn.Module):
    def __init__(self, nc, ndf) -> None:
        super().__init__()
        self.convLayer = nn.Sequential(
            ## nc * 28 * 28
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ## nc * 14 * 14 
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2 , padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ## nc * 7 * 7
            
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=1 , padding=0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
  
        )
    
    def forward(self, x):
        x = self.convLayer(x)
        return x.view(-1)


        

