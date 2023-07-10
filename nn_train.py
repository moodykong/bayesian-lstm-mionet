'''
This module is for for machine learning training of Ausgrid data. 
'''

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from nn_lib import Pendulum_Dataset, save_ckp, load_ckp, StandardScaler, draw_loss, draw_valid
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.cuda.amp import autocast, GradScaler

__version__ = '1.0.1'
__author__ = "Zhihao Kong"
__email__ = 'kong89@purdue.edu'
__date__ = '2023-06-27'


class LSTM_DeepONet(nn.Module):
    def __init__(self):
        super(LSTM_DeepONet, self).__init__()
        d_latent = 150
        d_rnn = 50
        self.layernorm=nn.LayerNorm([d_latent])
        self.layernorm_x_n=nn.LayerNorm([d_latent])
        self.layernorm_delta_t=nn.LayerNorm([d_latent])
        self.lstm = nn.LSTM(d_latent,d_rnn,2,batch_first=True,dropout=0.)

        input_mlp = nn.Sequential(
            
            nn.Linear(1, 50),
            nn.ReLU(),

            nn.Linear(50, 100),
            nn.ReLU(),

            nn.Linear(100, d_latent),
            
        )

        cell_mlp = nn.Sequential(
            nn.Linear(d_rnn, 100),
            nn.ReLU(),
            nn.Linear(100, d_latent),
        )

        hidden_mlp = nn.Sequential(
            nn.Linear(d_rnn, 100),
            nn.ReLU(),
            nn.Linear(100, d_latent),
        )

        x_n_mlp = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),

            nn.Linear(50, 100),
            nn.ReLU(),

            nn.Linear(100, d_latent),
        )

        delta_t_mlp = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),

            nn.Linear(50, 100),
            nn.ReLU(),

            nn.Linear(100, d_latent),
        )

        self.cell_mlp=cell_mlp
        
        self.input_mlp=input_mlp
        self.delta_t_mlp=delta_t_mlp
        self.x_n_mlp=x_n_mlp
        self.bias=nn.Parameter(torch.zeros(1))

    def forward(self, x, metadata, mask=None):
        pred = self.forward_1(x, metadata, mask)
        return pred

    def forward_1(self, x, metadata, mask):
        
        t_s = 0.01
        t = metadata[:,-3].type(torch.int64)
        x_n = x[:,t].diagonal()

        x = x.unsqueeze(-1)
        x_n = x_n.unsqueeze(-1).unsqueeze(-1)
        
        latent_x = self.input_mlp(x)
        latent_x_n = self.x_n_mlp(x_n)
        
        delta_t = metadata[:,[-2]] * t_s
        delta_t = delta_t.unsqueeze(-1)
        
        # normalize
        latent_x = self.layernorm(latent_x)

        # define the mask for the padded sequence
        mask_len = mask.sum(axis=-1).type(torch.int64).cpu() if mask is not None else None
        latent_x_packed = pack_padded_sequence(latent_x, mask_len, batch_first=True, enforce_sorted=False) if mask_len is not None else latent_x
        
        output, (h_n, c_n) = self.lstm(latent_x_packed)
        memory = self.cell_mlp(c_n[-1]).unsqueeze(1)
        
        latent_delta_t = self.delta_t_mlp(delta_t)
        latent_delta_t = self.layernorm_delta_t(latent_delta_t)
        latent_x_n = self.layernorm_x_n(latent_x_n)
        pred = (memory * latent_x_n * latent_delta_t).sum(dim=-1).view(-1,1) + self.bias

        return pred

def train_loop(dataloader, model, loss_fn, optimizer, input_transform=None, target_transform=None):
    num_batches = len(dataloader)
    train_loss, train_accuracy = 0., 0.
    pred_list=torch.tensor([],device=torch.device('cpu'))
    y_list=torch.tensor([],device=torch.device('cpu'))

    for batch, (X, metadata) in enumerate(dataloader):

        # Zero the gradients
        optimizer.zero_grad()
        
        y = metadata[:,-1].view(-1,1)
        t = metadata[:,-3]
        mask = (X!=0.).type(torch.bool)
        
        with autocast():
            # Compute prediction and loss
            pred = model(X,metadata,mask)
            
            batch_loss = loss_fn(pred, y)
            train_loss += batch_loss

            pred_list = torch.cat((pred_list,pred.cpu()),0)
            y_list = torch.cat((y_list,y.cpu()),0)

        # Backpropagation
        scaler.scale(batch_loss).backward()
        # Perform optimization
        scaler.step(optimizer)
        scaler.update()

    train_loss=train_loss.item()/num_batches
    train_accuracy=((pred_list > y_list * (1-accuracy_threshold)) * (pred_list < y_list * (1+accuracy_threshold))).sum()
    train_accuracy=train_accuracy.item()/torch.numel(y_list)*100
    #train_mape = (torch.abs(pred_list-y_list)/torch.abs(y_list)).mean().item()*100
    train_l2 = torch.sqrt(((pred_list-y_list)**2).sum() / (y_list**2).sum()) * 100
    train_accuracy = 100 - train_l2.item()
    
    r2=1-((pred_list-y_list)**2).sum()/((y_list-y_list.mean())**2).sum()
    
    return (pred_list, y_list), (train_loss, train_accuracy, r2)
   

def valid_loop(dataloader, model, loss_fn, input_transform=None, target_transform=None):
    num_batches = len(dataloader)
    valid_loss, valid_accuracy = 0., 0.

    with torch.no_grad():
        pred_list=torch.tensor([],device=torch.device('cpu'))
        y_list=torch.tensor([], device=torch.device('cpu'))
        
        for batch, (X, metadata) in enumerate(dataloader):
            
            y = metadata[:,-1].view(-1,1)
            t = metadata[:,-3]
            mask = (X!=0.).type(torch.bool)
            with autocast():
                # Compute prediction and loss
                pred = model(X,metadata,mask)
                
                batch_loss = loss_fn(pred, y)
                valid_loss += batch_loss

                pred_list = torch.cat((pred_list,pred.cpu()),0)
                y_list=torch.cat((y_list,y.cpu()),0)
    
    valid_loss=valid_loss.item()/num_batches
    valid_accuracy=((pred_list > y_list * (1-accuracy_threshold)) * (pred_list < y_list * (1+accuracy_threshold))).sum()
    valid_accuracy=valid_accuracy.item()/torch.numel(y_list)*100
    valid_mape = (torch.abs(pred_list-y_list)/torch.abs(y_list)).mean().item()*100
    #valid_accuracy = valid_mape

    valid_l2 = torch.sqrt(((pred_list-y_list)**2).sum() / (y_list**2).sum()) * 100
    valid_accuracy = 100 - valid_l2.item()
    r2=1-((pred_list-y_list)**2).sum()/((y_list-y_list.mean())**2).sum()

    return (pred_list,y_list), (valid_loss, valid_accuracy, r2)

loss_epoch = {}  # loss history
loss_epoch['train'] = []
loss_epoch['val'] = []
accuracy_epoch = {}
accuracy_epoch['train'] = []
accuracy_epoch['val'] = []

epochs_tonow = []
accuracy_threshold=.15
fig_train = plt.figure()
fig_valid = plt.figure()

loss_valid_min=np.Inf
checkpoint_path='models/recent_model.pt'
best_model_path='models/best_model'

device_glob = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

if __name__ == "__main__":
    
    standarize_X = None 
    standarize_metadata = None 

    torch.manual_seed(999)
    np.random.seed(999)

    train_dataset = Pendulum_Dataset(
        filepath='data/lorenz_random_init.pkl',
        search_len=2,
        search_num=40,
        use_padding=True,
        search_random = True,
        device=device_glob,
        transform=standarize_X,
        target_transform=standarize_metadata,
    )
   
    train_size = int(0.7 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=500, drop_last=True, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=500, drop_last=True, shuffle=True)

    model = LSTM_DeepONet()
    model.to(device=device_glob)
    model = nn.DataParallel(model,output_device=device_glob)

    learning_rate = 1e-3
    epoch_num = 2000
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    stop_flag = 0
    stop_threshold = 80
    
    for epoch in range(epoch_num):
        epochs_tonow.append(epoch+1)
        model.train()
        train_output, train_statistics = train_loop(train_dataloader, model, loss_fn, optimizer,standarize_X, standarize_metadata)
        loss_train, accuracy_train, r2_train = train_statistics[0], train_statistics[1], train_statistics[2]
        loss_epoch['train'].append(loss_train)
        accuracy_epoch['train'].append(accuracy_train)

        model.eval()
        valid_output, valid_statistics = valid_loop(valid_dataloader, model, loss_fn, standarize_X, standarize_metadata)
        loss_valid, accuracy_valid, r2_valid = valid_statistics[0], valid_statistics[1], valid_statistics[2]
        loss_epoch['val'].append(loss_valid) 
        accuracy_epoch['val'].append(accuracy_valid)
        
        draw_loss(fig_train,epochs_tonow,loss_epoch,accuracy_epoch,accuracy_threshold)

        checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': torch.tensor([loss_valid]),
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'transform': standarize_X,
                'target_transform': standarize_metadata,
            }
        
        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # save the model if validation loss has decreased
        if loss_valid <= loss_valid_min:
            print(f"\n-------------------------------\nEpoch {epoch+1}\n-------------------------------")
            print(f"\nTrain Loss: {loss_train:.2f} \nTrain Accuracy: {accuracy_train:.2f} %\nR2: {r2_train:.2f}")
            print(f"\nValid Loss: {loss_valid:.2f} \nValid Accuracy: {accuracy_valid:.2f} % \nR2: {r2_valid:.2f}")
            print('Validation loss decreased ({:.4f} --> {:.4f}).  \nSaving model ...'.format(loss_valid_min,loss_valid))
            valid_pred = valid_output[0]
            valid_y = valid_output[1]
            
            draw_valid(fig_valid,valid_pred,valid_y,accuracy_threshold)

            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path + '_'+ str(epoch+1) + '.pt')
            loss_valid_min = loss_valid

            stop_flag = 0
        stop_flag += 1

        if stop_flag > stop_threshold:
            print("\nEarly stop. Training Done.")
            break

        if epoch==epoch_num-1:
            print("All epoches training done!")
    