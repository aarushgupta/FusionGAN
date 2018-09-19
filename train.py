
# coding: utf-8

# In[1]:


import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils

import time
import numpy as np
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
plt.ion()

from model.generator import *
from model.discriminator import NLayerDiscriminator as Discriminator
from utils.dataloader import YouTubePose
from utils.loss_functions import lossIdentity, lossShape


# In[2]:


dataset_dir = './Dataset/'
batch_size = 8
epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.01
alpha = 1
beta = 1


# In[3]:


with open('./dataset_lists/train_datapoint_triplets.pkl', 'rb') as f:
    datapoint_pairs = pickle.load(f)


# In[4]:


with open('./dataset_lists/train_shapeLoss_pairs.pkl', 'rb') as f:
    shapeLoss_datapoint_pairs = pickle.load(f)


# In[5]:


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# In[6]:


train_dataset = YouTubePose(datapoint_pairs, shapeLoss_datapoint_pairs, dataset_dir, transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=0)
dataset_sizes = [len(train_dataset)]


# In[8]:


generator = Generator(ResidualBlock)


# In[9]:


discriminator = Discriminator(3)


# In[10]:


generator = generator.to(device)
discriminator = discriminator.to(device)


# In[11]:


optimizer_gen = optim.SGD(generator.parameters(), lr = learning_rate, momentum=0.9)
optimizer_disc = optim.SGD(discriminator.parameters(), lr = learning_rate, momentum=0.9)


# In[12]:


def train_model(gen, disc, loss_i, loss_s, optimizer_gen, optimizer_disc, alpha = 1, beta = 1, num_epochs = 10):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print('-'*10)
        dataloader = train_dataloader
        gen.train()
        disc.train()
        since = time.time()
        running_loss_iden = 0.0
        running_loss_s1 = 0.0
        running_loss_s2a = 0.0
        running_loss_s2b = 0.0
        running_loss = 0.0
        
        for i_batch, sample_batched in enumerate(dataloader):
            x_gen, y, x_dis = sample_batched['x_gen'], sample_batched['y'], sample_batched['x_dis']
            iden_1, iden_2 = sample_batched['iden_1'], sample_batched['iden_2']
            x_gen = x_gen.to(device)
            y = y.to(device)
            x_dis = x_dis.to(device)
            iden_1 = iden_1.to(device)
            iden_2 = iden_2.to(device)
            
            optimizer_gen.zero_grad()
            optimizer_disc.zero_grad()
            
            with torch.set_grad_enabled(True):
                x_generated = gen(x_gen, y)
                print('forward 1 done')
                fake_op, fake_pooled_op = disc(x_gen, x_generated)
                real_op, real_pooled_op = disc(x_gen, x_dis)
                loss_identity_gen = -loss_i(real_pooled_op, fake_pooled_op)
                print('Loss calculated')
                loss_identity_gen.backward(retain_graph=True)
                optimizer_gen.step()
                print('backward 1.1 done')
                
                optimizer_disc.zero_grad()
                loss_identity_disc = loss_i(real_op, fake_op)
                print('Loss calculated')
                loss_identity_disc.backward(retain_graph=True)
                optimizer_disc.step()
                print('backward 1.2 done')

                optimizer_gen.zero_grad()
                optimizer_disc.zero_grad()
                x_ls2a = gen(y, x_generated)
                x_ls2b = gen(x_generated, y)
                print('forward 2 done')

                loss_s2a = loss_s(y, x_ls2a)
                loss_s2b = loss_s(x_generated, x_ls2b)
                loss_s2 = loss_s2a + loss_s2b
                print('Loss calculated')

                loss_s2.backward()
                optimizer_gen.step()
                print('backward 2 done')

                optimizer_gen.zero_grad()
                optimizer_disc.zero_grad()
                
                x_ls1 = generator(iden_1, iden_2)
                print('forward 3 done')

                loss_s1 = loss_s(iden_2, x_ls1)
                print('Loss calculated')
                loss_s1.backward()
                optimizer_gen.step()
                print('backward 5 done')
                print()
            running_loss_iden += loss_identity_disc.item() * x_gen.size(0)
            running_loss_s1 += loss_s1.item() * x_gen.size(0)
            running_loss_s2a += loss_s2a.item() * x_gen.size(0) 
            running_loss_s2b += loss_s2b.item() * x_gen.size(0)
            running_loss = running_loss_iden +  beta * (running_loss_s1 + alpha * (running_loss_s2a + running_loss_s2b))
            print(str(time.time() - since))
            since = time.time()
        epoch_loss_iden = running_loss_iden / dataset_sizes[0]
        epoch_loss_s1 = running_loss_s1 / dataset_sizes[0]
        epoch_loss_s2a = running_loss_s2a / dataset_sizes[0]
        epoch_loss_s2b = running_loss_s2a / dataset_sizes[0]
        epoch_loss = running_loss / dataset_sizes[0]
        print('Identity Loss: {:.4f} Loss Shape1: {:.4f} Loss Shape2a: {:.4f}                Loss Shape2b: {:.4f}'.format(epoch_loss_iden, epoch_loss_s1,
                                           epoch_loss_s2a, epoch_loss_s2b))
        print('Epoch Loss: {:.4f}'.format(epoch_loss))
        print('Time Taken: ' + str(time.time() - since))
    return gen, disc


# In[13]:


generator, discriminator = train_model(generator, discriminator, lossIdentity, lossShape, optimizer_gen, optimizer_disc, alpha=alpha, beta=beta, num_epochs=epochs)

