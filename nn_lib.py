import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.linalg import cholesky
from scipy.interpolate import CubicSpline
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import shutil
import pickle


__version__ = '1.0.1'
__author__ = "Zhihao Kong"
__email__ = 'kong89@purdue.edu'
__date__ = '2023-06-27'

class Customize_Dataset(Dataset):
    def __init__(
        self,
        
        filepath = 'data/pendulum.pkl',
        search_len = 10,
        search_num = 5,
        search_random = True,
        use_padding = True,
        offset = 0,
        device = torch.device("cpu"), 
        transform = None, 
        target_transform = None):
        
        super().__init__()
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.search_len = search_len
        self.search_num = search_num
        self.offset = offset
        self.search_random = search_random
        self.use_padding = use_padding

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        t = data['t']
        x = data['theta']
        y = data['omega']
        u = data['u'] if 'u' in data else x.copy()
        t_s = t[1] - t[0]
        splines_x =[]
        splines_y =[]
        # Remove explosion points
        idxs_select = (np.isnan(x).sum(axis=1)==0) & (np.isnan(y).sum(axis=1)==0) & (np.isnan(u).sum(axis=1)==0)
        x = x[idxs_select] + 1e-6
        y = y[idxs_select] + 1e-6
        u = u[idxs_select] + 1e-6
        
        data_len = x.shape[0]
        self.dsize = data_len
        
        for i in range(data_len):
            spline_x = interp1d(np.arange(t.size), x[i], kind='cubic', fill_value=1e-6, bounds_error=False)
            splines_x.append(spline_x)
            spline_y = interp1d(np.arange(t.size), y[i], kind='cubic', fill_value=1e-6, bounds_error=False)
            splines_y.append(spline_y)
        
        if self.search_random:
            local_idxs = np.random.rand(search_num,data_len,3)
            offset = self.offset
            local_idxs[:,:,0] = (offset + local_idxs[:,:,0] * (t.size - offset - search_len)).astype(int)
            local_idxs[:,:,1] = local_idxs[:,:,1] * search_len
            local_idxs[:,:,2] = local_idxs[:,:,0] + local_idxs[:,:,1]
        else:
            local_idxs = np.ones((search_num,data_len,3))
            offset = self.offset
            idx_end = 1000 - search_len
            local_idxs[:,:,0] = (offset + np.linspace(1, idx_end , search_num)).reshape(-1,1).astype(int)
            local_idxs[:,:,1] = local_idxs[:,:,1] * 0.5 * search_len
            local_idxs[:,:,2] = local_idxs[:,:,0] + local_idxs[:,:,1]

        mask_idxs = np.ones((search_num,data_len,t.size))
        mask_idxs *= np.arange(t.size)
        mask_idxs = (mask_idxs > local_idxs[:,:,[0]]) 
        
        # Mask the input data
        data_input = u
        data_masked = np.repeat(np.expand_dims(data_input,axis=0),search_num,axis=0)
        data_masked[mask_idxs] = 0.
        
        data_idxs = np.arange(data_len,dtype=int)
        data_idxs = np.repeat(np.expand_dims(data_idxs,axis=0),search_num,axis=0)
        
        # Unfold the data
        data_masked = data_masked.reshape(-1,t.size)
        local_idxs = local_idxs.reshape(-1,local_idxs.shape[-1])
        data_idxs = data_idxs.reshape(-1)
        
        # Get the metadata
        metadata = np.hstack((local_idxs,np.zeros((local_idxs.shape[0],1))))
        x_tn = np.zeros(metadata.shape[0])
        y_tn = np.zeros(metadata.shape[0])

        # Get the next state
        for i in range(metadata.shape[0]):
            spline_x = splines_x[data_idxs[i]]
            spline_y = splines_y[data_idxs[i]]
            t_next = metadata[i,2]
            metadata[i,2] = spline_x(t_next)
            metadata[i,3] = spline_y(t_next)
            x_tn[i] = x[data_idxs[i],int(metadata[i,0])]
            y_tn[i] = y[data_idxs[i],int(metadata[i,0])]

        # Get the data
        data = data_masked
        metadata = np.insert(metadata,1,x_tn,axis=1)
        metadata = np.insert(metadata,2,y_tn,axis=1)
        
        # Remove the points close to zero
        #idxs_select = (np.abs(metadata[:,-1]) > 1e-6)
        #data = data[idxs_select]
        #metadata = metadata[idxs_select]

        self.data = torch.from_numpy(data).float()
        # metadata: (t, x_tn, y_tn, delta_t, x_next, y_next)
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
    fig_train.set_size_inches(10, 4.8)
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
    #ax1.set_title(f"accuracy (\u00B1{accuracy_threshold*100:.2f}%)")
    ax1.set_title(f"L2 (%)")
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
    xymax = torch.cat((pred_list,y_list),dim=0).max()*1.05
    xymin = torch.cat((pred_list,y_list),dim=0).min()*1.05
    #ax.plot([0,xymax],[[0,0,0],[xymax,xymax*(1 - accuracy_threshold),xymax*(1 + accuracy_threshold)]])
    ax.set_xlim([xymin,xymax])
    ax.set_ylim([xymin,xymax])
    ax.set_xlabel('Pred')
    ax.set_ylabel('Label')
    fig_valid.savefig('figures/validation.jpg',dpi=300)


def grf_1d(a=0.01, nu=1):
    # Correlation function
    def rho(h, a=a, nu=nu):
        return np.exp(- (h / a)**(2*nu))

    # Space discretization
    x = np.linspace(-10, 10, 2000)
    dx = x[1] - x[0]

    # Distance matrix
    H = np.abs(x[:, np.newaxis] - x)

    # Covariance matrix
    sigma = rho(H)
    sigma += 1e-6 * np.eye(sigma.shape[0])

    # Cholesky factorization
    L = cholesky(sigma, lower=True)

    # Independent standard Gaussian random variables
    z = np.random.normal(size=x.size)

    # Gaussian random field
    y = np.dot(L, z)
    spline = CubicSpline(x, y)
    return spline

class dotdict(dict):
    """dot.notation access to dictionary attributes."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def runge_kutta(f, x, u, h):
    """f(x, u) -> next_x."""
    k1 = f(x, u)
    k2 = f(x + 0.5 * h * k1, u)
    k3 = f(x + 0.5 * h * k2, u)
    k4 = f(x + h * k3, u)
    next_x = x + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6
    return next_x

def integrate(method, f, control, x0, h, N):
    soln = dotdict()
    soln.x = []
    soln.t = []
    soln.u = []

    x = x0

    t = 0 * h
    u = control(t, x)
    soln.x.append(x)


    for n in range(1, N):
        # log previous control
        soln.t.append(t)
        soln.u.append(u)
        # compute next state
        x_next = runge_kutta(f, x, u, h)
        # log next state
        soln.x.append(x_next)
        x = x_next
        t = n * h
        u = control(t, x)

    soln.x = np.vstack(soln.x)
    soln.t = np.vstack(soln.t)
    soln.u = np.vstack(soln.u)

    return soln


if __name__ == '__main__':
    pass   