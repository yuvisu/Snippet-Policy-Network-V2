import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class BaseCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, fc_size = 4):
        super(BaseCNN, self).__init__()
        
        self.fc_size = fc_size
        
        self.conv= nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5,stride=1)
        
        self.conv_pad_1_64 =  nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(64,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_2_64 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(64,momentum=0.1),
            nn.ReLU()
        )
        
        self.conv_pad_1_128 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(128,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_2_128 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(128,momentum=0.1),
            nn.ReLU()
        )
        
        self.conv_pad_1_256 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(256,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_2_256 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(256,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_3_256 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(256,momentum=0.1),
            nn.ReLU()
        )
        
        self.conv_pad_1_512 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(512,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_2_512 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(512,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_3_512 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(512,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_4_512 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(512,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_5_512 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(512,momentum=0.1),
            nn.ReLU()
        )
        self.conv_pad_6_512 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(512,momentum=0.1),
            nn.ReLU()
        )
        
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3,stride=3) 
        self.maxpool_2 = nn.MaxPool1d(kernel_size=3,stride=3) 
        self.maxpool_3 = nn.MaxPool1d(kernel_size=3,stride=3) 
        self.maxpool_4 = nn.MaxPool1d(kernel_size=3,stride=3) 
        self.maxpool_5 = nn.MaxPool1d(kernel_size=3,stride=3) 

        # *4 for the series length 1000
        # *2 for the series length 500
        self.dense1 = nn.Linear(512 * self.fc_size, 1024)
        self.dense2 = nn.Linear(1024, hidden_size)
        
        self.dense_final = nn.Linear(hidden_size, num_classes) #no use
        
    def forward(self, x):
        #print(x.shape)
        x = x.permute(0,2,1)
        x = self.conv_pad_1_64(x)
        x = self.conv_pad_2_64(x)
        x = self.maxpool_1(x)
        
        
        x = self.conv_pad_1_128(x)
        x = self.conv_pad_2_128(x)
        x = self.maxpool_2(x)
        
        x = self.conv_pad_1_256(x)
        x = self.conv_pad_2_256(x)
        x = self.conv_pad_3_256(x)
        x = self.maxpool_3(x)
        
        x = self.conv_pad_1_512(x)
        x = self.conv_pad_2_512(x)
        x = self.conv_pad_3_512(x)
        x = self.maxpool_4(x)
        
        x = self.conv_pad_4_512(x)
        x = self.conv_pad_5_512(x)
        x = self.conv_pad_6_512(x)
        x = self.maxpool_5(x)

        # *4 for the series length 1000
        # *2 for the series length 500
        x = x.view(-1, 512 * self.fc_size)
        #print(x.shape)
        x = self.dense1(x)
        #print(x.shape)
        x= self.dense2(x)
        
        return x

class BaseRNN(nn.Module):

    def __init__(self,
                 N_FEATURES,
                 HIDDEN_DIM,
                 CELL_TYPE="LSTM",
                 N_LAYERS=1):
        super(BaseRNN, self).__init__()

        # --- Mappings ---
        if CELL_TYPE in ["RNN", "LSTM", "GRU"]:
            self.rnn = getattr(nn, CELL_TYPE)(N_FEATURES,
                                              HIDDEN_DIM,
                                              N_LAYERS)
        else:
            try: 
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[CELL_TYPE]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was
                                 supplied, options are ['LSTM', 'GRU',
                                 'RNN_TANH' or 'RNN_RELU']""")

            self.rnn = nn.RNN(N_FEATURES,
                              HIDDEN_DIM,
                              N_LAYERS,
                              nonlinearity=nonlinearity)
        self.tanh = nn.Tanh()

    def forward(self, x_t, hidden):
        output, h_t = self.rnn(x_t, hidden)
        return output, h_t

class Discriminator(nn.Module):

    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        
        self.fc = nn.Linear(input_size, output_size)
        
        self.softmax = nn.LogSoftmax(dim=1) #test sofmax

    def forward(self, h_t):

        y_hat = self.fc(h_t)
        y_hat = self.softmax(y_hat)
        
        return y_hat