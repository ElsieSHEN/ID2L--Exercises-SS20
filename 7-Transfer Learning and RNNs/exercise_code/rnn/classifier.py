import pickle
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from .rnn_nn import *
from .base_classifier import *


class RNN_Classifier(Base_Classifier):
    
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()

    ############################################################################
    #  TODO: Build a RNN classifier                                            #
    ############################################################################

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(self.input_size, self.hidden_size,)
        self.fc = nn.Linear(self.hidden_size, classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


    def forward(self, x):
    ############################################################################
    #  TODO: Perform the forward pass                                          #
    ############################################################################   

        x, _ = self.rnn(x)
        x = self.fc(x[-1])

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
        return x


class LSTM_Classifier(Base_Classifier):

    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################
           
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, classes)

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################


    def forward(self, x):

        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################    

        x, _ = self.lstm(x)
        x = self.fc(x[-1])

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return x
