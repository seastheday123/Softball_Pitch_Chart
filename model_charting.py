# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:32:52 2024

@author: Alexa
"""

#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Import data from models
model_data = pd.read_excel('model_graph_data_test.xlsx')
#Find different models
model_labels = model_data['ChannelSetUp']
model_labels.drop_duplicates(inplace = True)

#graph the data
modelGraph, modelPlot = plt.subplots(2)

#Set up colors
colors = sns.color_palette('husl', len(model_labels))
modelPlot[0].set_prop_cycle('color', colors)
modelPlot[1].set_prop_cycle('color', colors)

#Plot each model with loss and accuracy reletive to the number of Epochs
for modelLabel in model_labels:
    modelPlot[0].plot(model_data[model_data.ChannelSetUp==modelLabel].NumberOfEpochs,model_data[model_data.ChannelSetUp==modelLabel].Loss,label=modelLabel)
    modelPlot[1].plot(model_data[model_data.ChannelSetUp==modelLabel].NumberOfEpochs,model_data[model_data.ChannelSetUp==modelLabel].TestAccuracy,label=modelLabel)

#Format the graphs
modelGraph.suptitle('Graphs of Loss and Accuracy for all Channel Set-Ups')
modelGraph.set_figwidth(15)
modelGraph.set_figheight(15)
modelPlot[0].set_facecolor((.95, .95, .95))
modelPlot[0].set_xlabel('Epoch')
modelPlot[0].set_ylabel('Loss')
modelPlot[0].legend(loc='center left', bbox_to_anchor=(1, 0), facecolor=(.95, .95, .95), title = "Channel Parameters")
modelPlot[1].set_facecolor((.95, .95, .95))
modelPlot[1].set_xlabel('Epoch')
modelPlot[1].set_ylabel('Accuracy')

#Add list for models that have the closest to zero loss after 20 epochs
model_channel_data_loss = []
for n in range(len(model_data)):
    if model_data.at[n, 'NumberOfEpochs'] == 20 and model_data.at[n, 'Loss'] < 0.0006:
        model_channel_data_loss.append(model_data.at[n,'ChannelSetUp'])

#Create Plot for loss and accuracy of best converging loss models
modelGraphLoss, modelPlotLoss = plt.subplots(2)

#Set up colors
colors = sns.color_palette('husl', len(model_labels))
modelPlotLoss[0].set_prop_cycle('color', colors)
modelPlotLoss[1].set_prop_cycle('color', colors)

#Plot each model with loss and accuracy reletive to the number of Epochs
for modelLabel in model_channel_data_loss:
    modelPlotLoss[0].plot(model_data[model_data.ChannelSetUp==modelLabel].NumberOfEpochs,model_data[model_data.ChannelSetUp==modelLabel].Loss,label=modelLabel)
    modelPlotLoss[1].plot(model_data[model_data.ChannelSetUp==modelLabel].NumberOfEpochs,model_data[model_data.ChannelSetUp==modelLabel].TestAccuracy,label=modelLabel)

#Format the graphs
modelGraphLoss.suptitle('Graphs of Loss and Accuracy for Loss Functions Converging on 0')
modelGraphLoss.set_figwidth(15)
modelGraphLoss.set_figheight(15)
modelPlotLoss[0].set_facecolor((.95, .95, .95))
modelPlotLoss[0].set_xlabel('Epoch')
modelPlotLoss[0].set_ylabel('Loss')
modelPlotLoss[0].legend(loc='center left', bbox_to_anchor=(1, 0), facecolor=(.95, .95, .95), title = "Channel Parameters")
modelPlotLoss[1].set_facecolor((.95, .95, .95))
modelPlotLoss[1].set_xlabel('Epoch')
modelPlotLoss[1].set_ylabel('Accuracy')

#Add list for models that have the best test accuracy after 20 epochs
model_channel_data_acc = []
for n in range(len(model_data)):
    if model_data.at[n, 'NumberOfEpochs'] == 20 and model_data.at[n, 'TestAccuracy'] > 0.9907:
        model_channel_data_acc.append(model_data.at[n,'ChannelSetUp'])

#Create Plot for loss and accuracy of best accuracy models
modelGraphAcc, modelPlotAcc = plt.subplots(2)

#Set up colors
colors = sns.color_palette('husl', len(model_labels))
modelPlotAcc[0].set_prop_cycle('color', colors)
modelPlotAcc[1].set_prop_cycle('color', colors)

#Plot each model with loss and accuracy reletive to the number of Epochs
for modelLabel in model_channel_data_acc:
    modelPlotAcc[0].plot(model_data[model_data.ChannelSetUp==modelLabel].NumberOfEpochs,model_data[model_data.ChannelSetUp==modelLabel].Loss,label=modelLabel)
    modelPlotAcc[1].plot(model_data[model_data.ChannelSetUp==modelLabel].NumberOfEpochs,model_data[model_data.ChannelSetUp==modelLabel].TestAccuracy,label=modelLabel)

#Format the graphs
modelGraphAcc.suptitle('Graphs of Loss and Accuracy for Accuracy Greater Than 0.99')
modelGraphAcc.set_figwidth(15)
modelGraphAcc.set_figheight(15)
modelPlotAcc[0].set_facecolor((.95, .95, .95))
modelPlotAcc[0].set_xlabel('Epoch')
modelPlotAcc[0].set_ylabel('Loss')
modelPlotAcc[0].legend(loc='center left', bbox_to_anchor=(1, 0), facecolor=(.95, .95, .95), title = "Channel Parameters")
modelPlotAcc[1].set_facecolor((.95, .95, .95))
modelPlotAcc[1].set_xlabel('Epoch')
modelPlotAcc[1].set_ylabel('Accuracy')   

#Save Graphs 
modelGraph.savefig('graphs/all_loss_accuracy.png', bbox_inches="tight")
modelGraphLoss.savefig('graphs/zero_loss.png', bbox_inches="tight")
modelGraphAcc.savefig('graphs/greater_accuracy.png',bbox_inches="tight")