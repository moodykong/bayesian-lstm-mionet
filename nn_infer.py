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
from nn_train import LSTM_DeepONet
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MultipleLocator
import matplotlib.animation as animation
import os

__version__ = '1.0.1'
__author__ = "Zhihao Kong"
__email__ = 'kong89@purdue.edu'
__date__ = '2023-06-27'


if __name__ == "__main__":
    
    print(f"\n-------------------------------\nInferring ...")
    device_glob = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(999)
    np.random.seed(999)

    # reconstruct the model class
    model = LSTM_DeepONet()
    model.to(device=device_glob)
    model = nn.DataParallel(model)

    # load the trained model
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_path='models/best_model_150.pt'

    model, optimizer, start_epoch, valid_loss_min, standarize_X, standarize_metadata = load_ckp(model_path, model, optimizer)
    
    # switch the model to eval mode
    model.eval()

    # define the dataset
    dataset = Pendulum_Dataset(
        filepath='data/pendulum_u_test_random_init_stat.pkl',
        search_len=2,
        search_num=40,
        use_padding=True,
        search_random=True,
        device=device_glob,
        transform=standarize_X,
        target_transform=standarize_metadata,
    )

    dataloader = DataLoader(dataset, batch_size=500, shuffle=False)
    size = len(dataloader.dataset)
    dsize = dataloader.dataset.dsize
    
    accuracy_threshold=0.15
    infer_accuracy=0

    # start inferencing
    with torch.no_grad():
        pred_list=torch.tensor([],device=torch.device('cpu'))
        y_list=torch.tensor([], device=torch.device('cpu'))
        t_list=torch.tensor([], device=torch.device('cpu'))
        X_list=torch.tensor([], device=torch.device('cpu'))

        for X, metadata in dataloader:
            
            y = metadata[:,-1].view(-1,1)
            delta_t = metadata[:,-2].view(-1,1)
            t = metadata[:,0].view(-1,1)
            t_search = t + delta_t
            mask = (X!=0).type(torch.bool)

            pred = model(X,metadata,mask)

            pred_list = torch.cat((pred_list,pred.cpu()),0)
            y_list=torch.cat((y_list,y.cpu()),0)
            t_list=torch.cat((t_list,t_search.cpu()),0)
            X_list=torch.cat((X_list,X.cpu()),0)
            
        # Draw Figures
        xylim=torch.cat((y_list,pred_list),0).max()*1.05
    
        cm = plt.get_cmap('gist_rainbow')
        num_colors = size +1
        fig_name_suffix=''
        fig, ax=plt.subplots()
        fig.set_size_inches(6.4,4.8)
        ax.set_prop_cycle(color=[cm(1.*i/num_colors) for i in range(num_colors)])
        ax.set_xlabel('t')
        ax.set_ylabel(r'$\theta(t)$')
        #ax.set_ylim(ymin=-2,ymax=2)
        ax.set_xlim(xmin=0,xmax=10)
        ax.grid(True)

        line1 = None
        line2 = None
        line3 = None

        def update_fig(frame=0):
            idxs = np.arange(frame*dataset.search_num*dsize,(frame+1)*dataset.search_num*dsize)
            global X_list, t_list, pred_list, y_list
            global line1, line2, line3
            X_memory = X_list[idxs[::dataset.search_num]]
            X_memory = np.where(X_memory!=0,X_memory,np.nan)
            if line1 and line2 and line3:
                for l in line1: l.remove()
                for l in line3: l.remove()
            
            idxs_sort = np.argsort(t_list[idxs].squeeze())
            t_list_sorted = t_list[idxs].squeeze()[idxs_sort]
            pred_list_sorted = pred_list[idxs].squeeze()[idxs_sort]
            y_list_sorted = y_list[idxs].squeeze()[idxs_sort]
            
            line1 = ax.plot(np.arange(X_memory.shape[1]),X_memory.T,marker='none',color ='green',label='X')
            line2 = ax.plot(t_list_sorted,pred_list_sorted,marker='.',color='red',linestyle ='--',label='pred')
            line3 = ax.plot(t_list_sorted,y_list_sorted,marker='.',color='green',alpha=0.7,linestyle ='--',label='label')
            
            return line1[0],line2[0],line3[0]
        
        frame_num = int(size/dataset.search_num/dsize)
        #ani = animation.FuncAnimation(fig, update_fig, frames=frame_num, interval=100, blit=True)
        #ani.save('figures/infer_anim'+ fig_name_suffix +'.gif', writer='pillow', fps=1, dpi=300)
        
        #ax.plot(np.arange(X_list.shape[1]-1)/100,X_list[[-1],:-1].T,linestyle='-',marker='none',color ='green',label='X')
        #ax.plot(t_list/100,pred_list,linestyle='none',marker='.',color='red',label='pred')
        ##ax.plot(t_list,y_list,linestyle='--',color='green',alpha=0.7,label='label')

        # calculate inference statistics
        infer_accuracy = ((pred_list > y_list * (1-accuracy_threshold)) * (pred_list < y_list * (1+accuracy_threshold))).sum()
        infer_accuracy = infer_accuracy.item()/torch.numel(pred_list)*100
        r2=1-((pred_list-y_list)**2).sum()/((y_list-y_list.mean())**2).sum()
        rmse=np.sqrt(((pred_list-y_list)**2).mean(axis=(0,1)))
        mae=np.abs((pred_list-y_list)).mean(axis=(0,1))
        idxs_select = torch.abs(y_list) > 1e-6
        
        #mape=torch.abs(pred_list[idxs_select]-y_list[idxs_select])/torch.abs(y_list[idxs_select])
        #mape = torch.nanmean(mape)*100

        # calculate inference L2 norm
        diff = pred_list[idxs_select]-y_list[idxs_select]
        l2_norm_diff = torch.sqrt((diff**2).sum(axis=0))
        l2_norm_y = torch.sqrt((y_list[idxs_select]**2).sum(axis=0))
        l2_percent = l2_norm_diff/l2_norm_y * 100
        # reshape pred_list and y_list to 2D array with shape (-1, 40)
        pred_list_mat = pred_list.reshape(-1,dataset.search_num)
        y_list_mat = y_list.reshape(-1,dataset.search_num)
        pred_list_mat = torch.where(torch.abs(y_list_mat)>1e-6,pred_list_mat,torch.nan)
        y_list_mat = torch.where(torch.abs(y_list_mat)>1e-6,y_list_mat,torch.nan)
        diff_mat = pred_list_mat - y_list_mat
        l2_norm_diff_mat = torch.sqrt(torch.nansum(diff_mat**2,dim=1))
        l2_norm_y_mat = torch.sqrt(torch.nansum(y_list_mat**2,dim=1))
        l2_percent_mat = l2_norm_diff_mat/l2_norm_y_mat * 100
        
        # output and figure drawing
        print(f"-------------------------------")
        print_txt = f"Model Name:\n{os.path.split(model_path)[1]}\nTrain Datasets:\n \nInference Datasets:\n \nRMSE: {rmse:.2f} \nMAE: {mae:.2f} \nL2: mean = {(l2_percent_mat.mean(dim=0)):.2f} %, std = {(l2_percent_mat.std(dim=0)):.2f}  %\nR2: {r2:.2f}\n"
        print(print_txt)
        info_txt = print_txt
        at = AnchoredText(info_txt, prop=dict(size=6),loc='lower center', frameon=False)

        ax.add_artist(at)
        #ax.legend(loc='lower right')
        plt.tight_layout()
        fig.savefig('figures/infer'+ fig_name_suffix +'.jpg',dpi=300)

        # draw figures
        ax.cla()
        num_colors=size+1
        ax.set_prop_cycle(color=[cm(1.*i/num_colors) for i in range(num_colors)])
        ax.scatter(pred_list,y_list,marker='.')    

        #ax.plot([0,xylim],[[0,0,0],[xylim,xylim*(1-accuracy_threshold),xylim*(1+accuracy_threshold)]],linestyle='dashed')
        #ax.set_xlim([0,xylim])
        #ax.set_ylim([0,xylim])
        ax.set_xlabel('Pred')
        ax.set_ylabel('Label')
        ax.add_artist(at)
        #ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        fig.savefig('figures/scatter_plot/infer_satter'+ fig_name_suffix +'.jpg',dpi=300)
        