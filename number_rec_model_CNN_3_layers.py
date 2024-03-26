# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:15:00 2024

@author: Alexa
"""
#import libraries
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import pandas as pd

#Download the testing and training data
train_data = datasets.MNIST(root = 'data', train = True, transform=ToTensor(), download=True)
test_data = datasets.MNIST(root = 'data', train = False, transform=ToTensor(), download=True)
#Define dataloader for train and test data
loaders = {'train': torch.utils.data.DataLoader(train_data, batch_size = 100, shuffle = True), 
           'test': torch.utils.data.DataLoader(test_data,batch_size = 100, shuffle = True)}
#Create list to loop multiple model parameters through
channels_size = [[1,16,16,48,48,32,5,3],
                 [1,16,16,48,48,32,4,4],
                 [1,16,16,48,48,32,3,5],
                 [1,32,32,64,64,32,5,3],
                 [1,32,32,64,64,32,4,4],
                 [1,32,32,64,64,32,3,5],
                 [1,48,48,64,64,32,5,3],
                 [1,48,48,64,64,32,4,4],
                 [1,48,48,64,64,32,3,5],
                 [1,64,64,48,48,32,5,3],
                 [1,64,64,48,48,32,4,4],
                 [1,64,64,48,48,32,3,5],
                 [1,8,8,16,16,32,5,3],
                 [1,8,8,16,16,32,4,4],
                 [1,8,8,16,16,32,3,5]] 
#Define lists for storing model loss and accuracy data
data_channels = []
data_epoch = []
data_loss = []
data_accuracy = []
#Create loop to model different parameters from the channel_size array 
for k in range(len(channels_size)):   
    #Create class for CNN model
    class CNN(nn.Module):
        def __init__(self):
            super(CNN,self).__init__()
            #Create first conv layer and normalize it
            self.conv1 = nn.Sequential(nn.Conv2d(channels_size[k][0],channels_size[k][1],channels_size[k][6],1,2),nn.ReLU(),nn.MaxPool2d(2))
            self.batchnorm1 = nn.BatchNorm2d(channels_size[k][1])
            #Create second conv layer and normalize it
            self.conv2 = nn.Sequential(nn.Conv2d(channels_size[k][2],channels_size[k][3],channels_size[k][6],1,2),nn.ReLU(),nn.MaxPool2d(2))  
            self.batchnorm2 = nn.BatchNorm2d(channels_size[k][3])
            #Create third conv layer and normalize it
            self.conv3 = nn.Sequential(nn.Conv2d(channels_size[k][4],channels_size[k][5],channels_size[k][6],1,2),nn.ReLU(),nn.MaxPool2d(2))
            self.batchnorm3 = nn.BatchNorm2d(channels_size[k][5])
            #Create out layer 
            self.out = nn.Linear(channels_size[k][5]*channels_size[k][7]*channels_size[k][7], 10, bias=True)
        #Pass data through the convolutions    
        def forward(self,x):
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = self.conv3(x)
            x = self.batchnorm3(x)
            #flatten and return the output
            x = x.view(x.size(0),-1)
            output = self.out(x)
            return output, x
    #Create model variable
    cnn = CNN()
    #Create definition to train the model
    def train(num_epochs, cnn, loaders):
        #Train the new model
        cnn.train()
        #run multiple epochs
        for epoch in range(num_epochs):
            #Train model and find the loss function
            for i, (images,labels) in enumerate(loaders['train']):
                b_x = Variable(images)
                b_y = Variable(labels)
                output = cnn(b_x)[0]
                loss = loss_func(output,b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pass
            #Print epoch and loss
            print ('Epoch [{}/{}] Loss: {:.10f} '.format(epoch + 1, num_epochs, loss.item()))
            #Find accuracy of model with test data
            test_accuracy() 
            #save Model
            torch.save(cnn.state_dict(), 'Models\cnn_epochs_{}_channelsize_{}.pt'.format(epoch+1,k))
            #Add model data to lists
            data_channels.append(channels_size[k])
            data_epoch.append(epoch+1)
            data_loss.append(loss.item())
        pass
    #Create function to find the test accuracy of the model
    def test_accuracy():
        # Test the model
        cnn.eval()
        #create variable for counting correct images
        correct = 0
        #Create loop to run all test images through the model
        with torch.no_grad():
            for images, labels in loaders['test']:
                test_output, last_layer = cnn(images)
                #find model perdiction
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                #add all correct ones together
                correct += sum(pred_y == labels).item()
                pass
            #find the accuracy
            test_acc = correct/len(test_data)
            #add accuracy datat to list
            data_accuracy.append(test_acc)
            #print the test accuracy
            print('Test Accuracy: {:.10f}'.format(test_acc))
            pass
    #create a loss function
    loss_func = nn.CrossEntropyLoss()
    #Define optimization function
    optimizer = optim.Adam(cnn.parameters(),lr = 0.01)
    #Define number of epochs
    num_epochs = 20
    #Train model
    train(num_epochs, cnn, loaders)
pass
#Take all model data and combine it for dataframe convertion
graph_data_df = list(zip(data_channels,data_epoch,data_loss,data_accuracy))
#Create dataframe
graph_data_df = pd.DataFrame(graph_data_df, columns = ['ChannelSet-Up', 'NumberOfEpochs','Loss','TestAccuracy'])
#export dataframe to excel
graph_data_df.to_excel("model_graph_data_test.xlsx")