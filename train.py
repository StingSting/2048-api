#!/usr/bin/env python
# coding: utf-8

# # Pytorch Tutorial

# Pytorch is a popular deep learning framework and it's easy to get started.



import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import time
import csv
from tqdm import tqdm
import numpy as np

BATCH_SIZE = 128
NUM_EPOCHS = 60
CHANNEL=12
OUT_SHAPE=(4,4)
def grid_ohe(input):
    out=[]
    each_c=[]
    for counter in range(len(input)):
        oneofinput = input[counter, :]
        for i in range(CHANNEL):
            ret=np.zeros(shape=(4,4),dtype=int)
            for r in range(4):
                for c in range(4):
                    if i==oneofinput[4*r+c]:
                        ret[r,c]=1
            each_c.append(ret)
        out.append(each_c)
        each_c = []
    return out#shape:4*4*channel
# with open('data_test.csv','r') as csvfile:
#     reader = csv.reader(csvfile)
#     rows = [row for row in reader]
# input = np.array(rows).astype('float')
# input = torch.FloatTensor(input)
# train_loader = data.DataLoader(input, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
data_read= pd.read_csv('data_train.csv')
data_train=data_read.values
board_train = data_train[:,0:16]
board_train=grid_ohe(np.int_(board_train))
direction_train=np.int_(data_train[:,16])
board_train = torch.FloatTensor(board_train)
direction_train= torch.LongTensor(direction_train)
train_dataset = data.TensorDataset(board_train,direction_train)
train_loader =data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)

data_read= pd.read_csv('data_test.csv')
data_test=data_read.values
board_test = data_test[:,0:16]
board_test=grid_ohe(np.int_(board_test))
direction_test=np.int_(data_test[:,16])
board_test = torch.FloatTensor(board_test)
direction_test= torch.LongTensor(direction_test)
test_dataset = data.TensorDataset(board_test,direction_test)
test_loader =data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=False)


# TODO:define model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(CHANNEL, 64, kernel_size=(1, 4), padding=(0, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(4, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(4, 4), padding=(2, 2)),
            nn.ReLU()
             )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(128 * 5 * 5),
            nn.Linear(128*5*5, 2048),
            nn.ReLU(),
#            nn.Dropout(0.3),
             # nn.BatchNorm1d(2048),
             # nn.Linear(2048,2048),#去掉这个全连接层
             # nn.ReLU(),
  #          nn.Dropout(0.5),
  
            nn.BatchNorm1d(2048),
             nn.Linear(2048,256),
             nn.ReLU(),
    #         nn.Dropout(0.5),
             nn.BatchNorm1d(256),
             nn.Linear(256,4)  
   #          nn.BatchNorm1d(2048),  
   #          nn.Linear(2048,1024),
   #          nn.ReLU(),
   # #         nn.Dropout(0.5),
   #          nn.BatchNorm1d(1024),
   #          nn.Linear(1024,4)            
        )
    def forward(self, x):
        out = self.conv1(x)
        out = out.view(out.shape[0], -1)  # reshape
        out = self.fc(out)
        return out
#class SimpleNet(nn.Module):
#    def __init__(self):
#        super(SimpleNet, self).__init__()
#        self.conv = nn.Sequential(
#            nn.Conv2d(1, 6, 2,padding=(1,1)), # in_channels, out_channels, kernel_size
#            nn.Sigmoid(),
#            #nn.MaxPool2d(2, 2), # kernel_size, stride
#            nn.Conv2d(6, 16, 3),
#            nn.Sigmoid(),
#            #nn.MaxPool2d(2, 2)
#        )
#        self.fc = nn.Sequential(
#            nn.Linear(144, 120),
#            nn.Sigmoid(),
#            nn.Linear(120, 84),
#            nn.Sigmoid(),
#            nn.Linear(84, 16)
#        )
#
#    def forward(self, img):
#        feature = self.conv(img)
#        output = self.fc(feature.view(img.shape[0], -1))
#        return output
model = SimpleNet()
#model.load_state_dict(torch.load('model.pkl'))#继承
model=model.cuda()
# TODO:define loss function and optimiter

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),0.001)

# Next, we can start to train and evaluate!



for epoch in range(NUM_EPOCHS):
    model.train()
    for i, (image,label) in enumerate(tqdm(train_loader)):
        image, label = Variable(image).cuda(), Variable(label).cuda()
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
    print('\n Epoch [%d/%d],  Loss_item: %.4f'% (epoch + 1, NUM_EPOCHS, loss.item()))
    #if loss.item()<0.20:
    if epoch > 20:
     torch.save(model.state_dict(), 'epoch{}.pkl'.format(epoch+1))#save low loss model
    

# Save the Trained Model
torch.save(model.state_dict(), 'model.pkl')


model.eval()

correct=0.00
total=0.00
for i, (image,label) in enumerate(tqdm(test_loader)):
    image, label = Variable(image).cuda(), Variable(label).cuda()    
    outputs=model(image)
    # print('outputs')
    # print(outputs)
    _,predicted=torch.max(outputs.data,1)
    # print('predicted')
    # print(predicted)
    total+=label.size(0)
    correct+=(predicted==label).sum()
print('\n Accuracy of test=%.4f'% (correct/total))
   
