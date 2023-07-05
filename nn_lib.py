import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import shutil


__version__ = '1.0.1'
__author__ = "Zhihao Kong"
__email__ = 'kong89@purdue.edu'
__date__ = '2023-06-27'

class Ausgrid_Dataset(Dataset):
    def __init__(
        self,
        
        filepath = 'data/lorenz.pkl',
        time_range = (20, 999),
        mask_len = None,
        search_len = 10,
        search_num = 5,
        search_random = True,
        use_padding = True,
        device = torch.device("cpu"), 
        transform = None, 
        target_transform = None):
        
        super().__init__()
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.time_range = time_range
        self.mask_len = mask_len
        self.search_len = search_len
        self.search_num = search_num
        self.search_random = search_random
        self.use_padding = use_padding if self.mask_len is None else False

        data_pd = pd.read_pickle(filepath)
        len_data = data_pd.shape[0]

        t = data_pd['t'].values
        x = data_pd['x'].values + 20.
        t_s = t[1] - t[0]
        
        data = x.reshape(1,-1)

        splines = []
        for i in range(data.shape[0]):
            spline = interp1d(np.arange(t.shape[0]), data[i], kind='cubic', fill_value=1e-6, bounds_error=False)
            splines.append(spline)
        
        data_idxs = np.arange(data.shape[0],dtype=int).reshape(-1,1)
        # get the metadata
        metadata = np.zeros((data.shape[0], 1))
        y = data.copy()
        
        dsize = data.shape[0]
        self.dsize = dsize

        if self.mask_len is not None:
            # this is usually used for inference
            data[:, self.mask_len:] = 0.
            metadata[:, -1] = self.mask_len - 1

        if self.use_padding:
            # this is usually used for training
            seg_len = 1
            repeats = data.shape[1] // seg_len - 1
            data = np.tile(data,(repeats,1))
            y = data.copy()
            metadata = np.tile(metadata,(repeats,1))
            data_idxs = np.tile(data_idxs,(repeats,1))
            
            for i in range(repeats):
                data[i*dsize:(i+1)*dsize,(i+1)*seg_len:] = 0.
                metadata[i*dsize:(i+1)*dsize,-1] = i

        # select the data in the time range
        time_range/=t_s
        t_range_idxs = (metadata[:,-1] >= time_range[0]) & (metadata[:,-1] <= time_range[1])
        data = data[t_range_idxs]
        metadata = metadata[t_range_idxs]
        y = y[t_range_idxs]
        data_idxs = data_idxs[t_range_idxs]

        # search for future data at random delta time
        # get a list of random float numbers in the range of 0 to search_len
        delta_idxs = np.ones((search_num,1))
        delta_idxs = np.tile(delta_idxs, (data.shape[0], 1))
        if search_random:
            delta_idxs = np.random.rand(*delta_idxs.shape) * search_len
        else:
            delta_idxs = search_len * .5 * np.ones(delta_idxs.shape)
        # repeat the data to search_num times
        data = np.repeat(data, search_num, axis=0)
        metadata = np.repeat(metadata, search_num, axis=0)
        y = np.repeat(y, search_num, axis=0)
        data_idxs = np.repeat(data_idxs, search_num, axis=0)
        search_idxs = delta_idxs + metadata[:, [-1]] 
        # get y_interp at the search_idxs for each y using spline interpolation
        y_interp = np.zeros((y.shape[0], 1))
        
        for i in range(y.shape[0]):
            spline = splines[data_idxs[i,0]]
            y_interp[i] = spline(search_idxs[i])
        metadata = np.concatenate((metadata, delta_idxs, y_interp), axis=1)
        self.data = torch.from_numpy(data).float()
        # (metadata, t, search_idxs, y_interp)
        self.metadata = torch.from_numpy(metadata).float()

        print(f"Data shape: {self.data.shape}")
        print(f"Data memory: {(self.data.element_size() * self.data.nelement() + self.metadata.element_size() * self.metadata.nelement()) / 1024 ** 2:.2f} MB")

        self.data = self.data.to(self.device)
        self.metadata = self.metadata.to(self.device)

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        metadata = self.metadata[idx]
        return data, metadata

class StandardScaler:
    def __init__(self,axis=0):
        self.mean = None
        self.std = None
        self.axis = axis
    def fit(self, x):
        mask = np.abs(x) > 1e-7
        x = np.where(mask,x,np.nan)
        self.mean =  np.nanmean(x[mask],axis=self.axis)
        self.std = np.nanstd(x[mask],axis=self.axis)
        
    def transform(self, x):
        x = (x - self.mean) / (self.std + 1e-7)
        return x
    def fit_transform(self, x):
        self.fit(x)
        x = self.transform(x)
        return x
    def back_transform(self, x):
        if self.std.shape[-1] != x.shape[-1]:
            x = x * (self.std.reshape(x.shape) + 1e-7) + self.mean.reshape(x.shape)
        else:
            x = x * (self.std + 1e-7) + self.mean
        return x

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath,map_location=torch.device('cpu'))
    # initialize state_dict from checkpoint to model
    model_state_dict = checkpoint['state_dict']
    #model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()} # remove 'module.' of dataparallel
    # initialize state_dict from checkpoint to model
    model.load_state_dict(model_state_dict)
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    #train_set = checkpoint['train_set']
    transform = checkpoint['transform']
    target_transform = checkpoint['target_transform']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item(), transform, target_transform

def draw_loss(fig_train,epochs_tonow,loss_epoch,accuracy_epoch,accuracy_threshold=.2):
    
    fig_train.clear()
    fig_train.set_size_inches(10, 4)
    ax0 = fig_train.add_subplot(121)
    ax1 = fig_train.add_subplot(122)

    plt.ioff()
    
    ax0.plot(epochs_tonow, loss_epoch['train'], 'b-', label='train')
    ax0.plot(epochs_tonow, loss_epoch['val'], 'r-', label='val')
    ax0.xaxis.set_major_locator(MultipleLocator(50))
    ax0.xaxis.set_major_formatter('{x:.0f}')
    ax0.xaxis.set_minor_locator(MultipleLocator(10))
    ax0.set_title("loss")
    ax0.legend()
    
    ax1.plot(epochs_tonow, accuracy_epoch['train'], 'b-', label='train')
    ax1.plot(epochs_tonow, accuracy_epoch['val'], 'r-', label='val')
    ax1.xaxis.set_major_locator(MultipleLocator(50))
    ax1.xaxis.set_major_formatter('{x:.0f}')
    ax1.xaxis.set_minor_locator(MultipleLocator(10))
    ax1.set_title(f"accuracy (\u00B1{accuracy_threshold*100:.2f}%)")
    ax1.legend()
    if epochs_tonow == 0:
        ax0.legend()
        ax1.legend()
    
    fig_train.savefig('figures/train.jpg',dpi=300)

def draw_valid(fig_valid,pred_list,y_list,accuracy_threshold=.2):
    
    fig_valid.clear()
    fig_valid.set_size_inches(6.4, 4.8)
    ax = fig_valid.add_subplot()
    ax.scatter(pred_list,y_list,marker='.')
    xylim = torch.cat((pred_list,y_list),dim=0).max()*1.05
    ax.plot([0,xylim],[[0,0,0],[xylim,xylim*(1 - accuracy_threshold),xylim*(1 + accuracy_threshold)]])
    ax.set_xlim([0,xylim])
    ax.set_ylim([0,xylim])
    ax.set_xlabel('Pred')
    ax.set_ylabel('Label')
    fig_valid.savefig('figures/validation.jpg',dpi=300)

if __name__ == '__main__':
    pass   