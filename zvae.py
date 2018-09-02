import torch
import numpy as np
import argparse, os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

latent_dim  = 20

# Visdom
# Tensorboard
# Training Plot
# Visualization
# NSML Integration
# Save Bind Model

# VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sample_z(z_mean, z_log_var)
        x_reconst = self.decoder(z)
        return x_reconst, z_mean, z_log_var

    def sample_z(self, z_mean, z_log_var):
        '''
        samples latent variable z ~ N(mu, sigma^2) with reparameterization trick
        '''
        std_z = torch.randn_like(z_mean)
        z_sigma = torch.exp(z_log_var/2)
        return z_mean + z_sigma * std_z

class Encoder(nn.Module):
    # CNN version
    # def __init__(self, input_dim, hidden_dim, z_dim):
    #     super(Encoder, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    #     self.conv1_bn = nn.BatchNorm2d(10)
    #     self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #     self.conv2_bn = nn.BatchNorm2d(20)
    #     self.conv2_drop = nn.Dropout2d()
    #     self.fc1 = nn.Linear(320, 50)
    #     self.fc1_bn = nn.BatchNorm1d(50)
    #     self.fc2 = nn.Linear(50, 30)
    #     self.fc2_bn = nn.BatchNorm1d(30)
    #
    #     self.fc3 = nn.Linear(30, z_dim)
    #     self.fc4 = nn.Linear(30, z_dim)
    #
    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
    #
    #     x = F.max_pool2d(self.conv2_drop(self.conv2_bn(self.conv2(x))), 2)
    #     x = F.relu(x)
    #     x = x.view(-1, 320)
    #     x = F.relu(self.fc1_bn(self.fc1(x)))
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2_bn(self.fc2(x))
    #
    #     return self.fc3(x), self.fc4(x)

    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, z_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        return self.linear2(x), self.linear3(x)
class Decoder(nn.Module):
    # CNN version
    # def __init__(self, z_dim, hidden_dim, input_dim):
    #     super(Decoder, self).__init__()
    #     # input is Z, going into a convolution
    #     self.layer1 = nn.Sequential(
    #         # input is Z, going into a convolution: [64, 100, 1, 1]
    #         nn.Linear(z_dim, 30),
    #         nn.BatchNorm1d(30),
    #         nn.ReLU(True),
    #         nn.Linear(30, 50),
    #         nn.BatchNorm1d(50),
    #         nn.ReLU(True),
    #         nn.Linear(50, 320),
    #         nn.BatchNorm1d(320),
    #         nn.ReLU(True)
    #         )
    #     self.layer2 = nn.Sequential(
    #         nn.ConvTranspose2d(20, 10, 3, 2, 1, 1),  #10*8*8
    #         nn.BatchNorm2d(10),
    #         nn.ReLU(True),
    #         nn.ConvTranspose2d(10, 1, 3, 2, 2, 1), #1*16*16
    #         nn.ReLU(True),
    #         nn.ConvTranspose2d(1, 1, 3, 2, 1, 1), #1*16*16
    #         nn.Sigmoid(),
    #     )
    #
    # def forward(self,x):
    #     x = self.layer1(x)
    #     x = x.view(-1,20,4,4)
    #     x = self.layer2(x)
    #     return x

    def __init__(self, z_dim, hidden_dim, input_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.sigmoid(self.linear2(x))

def train(model, device, data_loader, optimizer, num_epochs, log_interval, per_epoch=5):
    model.train()
    for epoch in range(num_epochs):
        print("-"*50)
        sample_x = None
        for batch_idx, (x, _) in enumerate(data_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x = x.view(x.size()[0], -1) # comment this for CNN version
            x_reconst, z_mean, z_log_var = model(x)

            # Compute reconstruction loss and kl divergence
            criterion = nn.BCELoss(size_average=False) # or nn.MSELoss(size_average=False) but BCELoss works better
            reconst_loss = criterion(x_reconst,x)#/torch.exp(z_log_var)
            kl_div = 0.5 * torch.sum(z_mean.pow(2) + z_log_var.exp() - z_log_var - 1)

            # Backprop and optimize
            loss = reconst_loss + kl_div
            loss.backward()
            # If there are exploding gradient issues uncomment this line and set the clipping threshold
            # torch.nn.utils.clip_grad_norm(model.parameters(),5)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Total Loss: {:.4f}, Reconst Loss: {:.4f}, KL Div: {:.4f}"
                       .format(epoch+1, num_epochs, batch_idx+1, len(data_loader), loss.item(), reconst_loss.item(), kl_div.item()))
        # Save the reconstructed images
        if (epoch+1) % per_epoch ==0:
            x_concat = torch.cat([x.view(-1, 1, 28, 28), model(x)[0].view(-1, 1, 28, 28)], dim=3)
            save_image(x_concat, os.path.join('.''', 'reconst_epoch_{}.png'.format(epoch+1)))

def generate(model, device, latent_dim, grid_size=64):
    model.eval()
    with torch.no_grad():
        # Save the sampled images
        z = torch.randn(grid_size, latent_dim).to(device)
        out = model.decoder(z).view(-1, 1, 28, 28)
        save_image(out, os.path.join('.', 'sampled.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoder')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--latent_dim', type=int, default=20, metavar='N', help='Size of latent dimenion')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    # MNIST dataset
    dataloader = torch.utils.data.DataLoader(
                    datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
                    batch_size=args.batch_size, shuffle=True)

    model = VAE(784, 500, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Use MultiGPU if available
    if torch.cuda.device_count() > 1:
      print("Using", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model)

    train(model, device, dataloader, optimizer, args.epochs, args.log_interval)
    generate(model, device, args.latent_dim)
        # test(args, model, device, test_loader)
