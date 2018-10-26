"""
Convolutional Neural Network for Saccade Detection
Bellet et al. 2018
Contact: marie-estelle.bellet@student.uni-tuebingen.de
"""
import numpy as np
import os
from skimage.measure import label
from sklearn import metrics
from scipy import io
#from IPython import display
# Pytorch imports:
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim



################################
# Neural Network Architecture: #
################################
class UNet(nn.Module):
    def __init__(self,dim=2,ks=5,mp=5):
        super(UNet,self).__init__()
        ''''
        dim: number of output features
        ks: kernel size of convolutional operations
        mp: kernel size of max pooling
        '''
        Ch = [1,10,20,40,dim] # number of features
        pd = int((ks-1)/2) # number of bins for border padding on each side
 
        self.c0 = nn.Sequential(
            nn.Conv2d(Ch[0],Ch[1],(ks,2),stride=1,padding=(pd,0)),
            nn.ReLU(True),
            nn.BatchNorm2d(Ch[1],affine=True),
        )
        self.c1 = nn.Sequential(
            nn.Conv1d(Ch[1],Ch[2],ks,stride=1,padding=pd),
            nn.ReLU(True),
            nn.BatchNorm1d(Ch[2],affine=True),
        )
        self.p1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=mp,stride=mp),
        )
        self.c2 = nn.Sequential(
            nn.Conv1d(Ch[2],Ch[2],ks,stride=1,padding=pd),
            nn.ReLU(True),
            nn.BatchNorm1d(Ch[2],affine=True),
        )
        self.p2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=mp,stride=mp),
        )
        self.c3 = nn.Sequential(  
            nn.Conv1d(Ch[2],Ch[2],ks,stride=1,padding=pd),
            nn.ReLU(True),
            nn.BatchNorm1d(Ch[2],affine=True), 
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(Ch[2],Ch[2],mp,stride=mp,padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(Ch[2],affine=True),
        )
        self.c4 = nn.Sequential(  
            nn.Conv1d(Ch[3],Ch[2],ks,stride=1,padding=pd),
            nn.ReLU(True),
            nn.BatchNorm1d(Ch[2],affine=True),  
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(Ch[2],Ch[2],mp,stride=mp,padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(Ch[2],affine=True),
        )
        self.c5 = nn.Sequential(
            nn.Conv1d(Ch[3],Ch[2],ks,stride=1,padding=pd),
            nn.ReLU(True),
            nn.BatchNorm1d(Ch[2],affine=True),
        )
        self.c6 = nn.Sequential(
            nn.Conv1d(Ch[2],Ch[1],ks,stride=1,padding=pd),
            nn.ReLU(True),
            nn.BatchNorm1d(Ch[1],affine=True),
        ) 
        self.c7 = nn.Conv1d(Ch[1],Ch[4],1,stride=1,padding=0)     
        
        self.sftmax = nn.Softmax(dim=1)
        
    def forward(self,input,outkeys):
        
        out = {}
        out['in'] = input
        out['c0'] = self.c0(input).squeeze(3)
        out['c1'] = self.c1(out['c0'])
        out['p1'] = self.p1(out['c1'])
        out['c2'] = self.c2(out['p1'])
        out['p2'] = self.p2(out['c2'])
        out['c3'] = self.c3(out['p2'])
        out['up1'] = self.up1(out['c3']) 
        out['c4'] = self.c4(torch.cat((out['p1'],out['up1']),1))
        out['up2'] = self.up2(out['c4']) 
        out['c5'] = self.c5(torch.cat((out['c1'],out['up2']),1))
        out['c6'] = self.c6(out['c5'])
        out['out'] = self.sftmax(self.c7(out['c6']))
        
        return [out[key] for key in outkeys]

        
        
        
###############################
###### helper functions: ######
###############################
 
# Initialization of weights of convolutional layers
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    
# Multi-class loss
class MCLoss(nn.Module):
    '''
    Torch cost function with foward method. 
    '''
    def __init__(self):
        super(MCLoss,self).__init__()
           
    def forward(self,prediction,target):
        
        epsilon = 1e-7
        prediction = torch.clamp(prediction,min=epsilon)
        prediction = torch.clamp(prediction,max=1-epsilon)  
        E = - torch.mean( target * torch.log(prediction) )

        return E
    
    
def lr_decay(optimizer, lr_decay=0.5):
    '''
    Decay learning rate by a factor of lr_decay
    '''
 
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
        
    return optimizer


def add_noise(X,Y,sd):
    """
    Add white noise to data for data augmentation
    
    Parameters
    ----------
    X: array-like, horizontal eye trace
    Y: array-like, vertical eye trace
    sd: list of floats, shape=2, standard deviation of the white noise in a range [min,max]
    
    Output
    ------
    Xnoise: array-like, horizontal eye trace + white noise
    Ynoise: array-like, vertical eye trace + white noise
    
    """
    t = X.shape[1]
    N = X.shape[0]
    np.random.seed(20)
    
    trial_noise_level = np.tile(np.random.rand(N,1) * (sd[1]-sd[0]) + sd[0], (1,t));
    noise4X = np.multiply(np.random.randn(N,t),trial_noise_level)
    noise4Y = np.multiply(np.random.randn(N,t),trial_noise_level)
    
    return X+noise4X , Y+noise4Y

def merge_saccades(Prediction,samp_freq,min_sacc_dist=10):
    '''
    Merge saccades that are too close in time
    
    Parameters
    ----------
    Prediction: array-like, shape={(n_timepoints),(n_samples,n_timepoints)}, binary saccade prediction
    samp_freq: int, sampling frequency in Hz
    min_sacc_dist: int, minimum saccade distance for merging of saccades, default=10 ms

    Output
    ------
    Prediction_new = array-like, shape={(n_timepoints),(n_samples,n_timepoints)}, binary saccade prediction
    '''
    Prediction2 = (Prediction.copy()==0).astype(int)
    Prediction_new = Prediction.copy()
    
    if len(Prediction.shape)<2: # case where network output is a vector
        l = label(Prediction2)
        first_label = 1 + int(l[0]==1)
        last_label = np.max(l) - int(l[-1]==np.max(l))
        for i in range(first_label,last_label+1):
            if np.sum(l==i)<int(min_sacc_dist*(samp_freq/1000)):
                Prediction_new[l==i] = 1
                    
    else:  # case where network output is a matrix
        for n in range(Prediction.shape[0]):
            l = label(Prediction2[n,:])
            first_label = 1 + int(l[0]==1)
            last_label = np.max(l) - int(l[-1]==np.max(l))
            for i in range(first_label,last_label+1):
                if np.sum(l==i)<int(min_sacc_dist*(samp_freq/1000)):
                    Prediction_new[n,l==i] = 1
                    
    return Prediction_new        
            
    
def binary_prediction(output,samp_freq,p=0.5,min_sacc_dist=1,min_sacc_dur=1): 
    '''
    Predict saccades from network probability output.
    Apply threshold on softmax and delete saccade of length < min_sacc_dur.
    
    Parameters
    ----------
    output: array-like, shape={(n_timepoints), (n_samples,n_timepoints)}, network softmax output
    samp_freq: int, sampling frequency in Hz
    p: float, threshold for saccade class, default=0.5
    min_sacc_dur: int, minimum saccade duration for removal of small saccades, default=1 (==no filtering)
    
    Output
    ------
    S_predic: array-like, shape={(n_timepoints), (n_samples,n_timepoints)}, fixation(=0) and saccade(=1) prediction for each time_bin in the network output
    
    '''
    S_predic = (output>p).astype(int)
    if min_sacc_dist!=1:
        # merge saccades with distance < min_sacc_dist
        S_predic = merge_saccades(S_predic,samp_freq,min_sacc_dist)
    
    # delete small saccades                
    if min_sacc_dur>1:
        if len(output.shape)<2:
            # case where network output is a vector
            l = label(S_predic)
            for j in range(1,np.max(l)+1):
                s = np.sum(l==j)
                if s<min_sacc_dur:
                    S_predic[l==j] = 0
        else:
            # case where network output is a matrix; assume: first dimension: number of samples
            for n in range(output.shape[0]):
                l = label(S_predic[n,:])
                for j in range(1,np.max(l)+1):
                    s = np.sum(l==j)
                    if s<min_sacc_dur:
                        S_predic[n,l==j] = 0                        
    return S_predic  


def cluster_belonging (logical_array,index):
    '''
    return the cluster of 'True' values that contain a time point
    logical_array: 1D array of 'True' and 'False'
    index value between 0 and length of logical_array
    '''
    
    labels = label(logical_array)
    
    cluster=(np.arange(0,len(logical_array))*0).astype(bool)
    cluster[labels==labels[index]]=True              
    return cluster


def accuracy (predicted,truth):
    '''
    Computes accuracy of prediction compared to groundtruth (in terms of true and false positive and false negative detections)
    Implemented for saccades only (no pso)
    
    Parameters
    ----------
    predicted: array-like, shape={(n_timepoints),(n_samples,n_timepoints)}, fixation(=0) and saccade(=1) prediction
    truth: array-like, shape={(n_timepoints),(n_samples,n_timepoints)}, fixation(=0) and saccade(=1) labels
    
    Output
    ------
    true_pos: float, number of correctly detected saccades
    false_pos: float, number of wrongly detected saccades
    false_neg: float, number of missed saccades
    on_distance: array-like, time-bin difference between true and predicted saccade onset for all true positives
    off_distance: array-like, time-bin difference between true and predicted saccade offset for all true positives
    
    '''
    
    if len(truth.shape)>1:
        # assume first dimension is number of samples
        batchsize = predicted.shape[0]
    else:
        # if input=vector, set batchsize to 1
        batchsize = 1
        # enforce two dimensions
        truth = np.atleast_2d(truth)
        predicted = np.atleast_2d(predicted)
        
    truth_copy = truth.copy()
    truth = truth.astype(int)
    pred_copy = predicted.copy()
    
    # Velocity
    Diff_p = np.diff(predicted,axis=1)
    Diff_t = np.diff(truth,axis=1)
    
    # count variables:
    true_pos = 0
    false_neg = 0
    false_pos = 0
    on_distance = []
    off_distance = []
    
    # loop trough samples:
    for i in range(batchsize):
        trace_t = Diff_t[i,:]
        trace_p = Diff_p[i,:]
        
        # find start and end of saccades
        # labels from prediction
        start_p = np.argwhere(trace_p==1)+1 # add one as the diff is shifted one bin early
        end_p = np.argwhere(trace_p==-1)+1 # add one as python indexing doesn't include last index in range
        # labels from ground truth
        start_t = np.argwhere(trace_t==1)+1 # add one as the diff is shifted one bin early
        end_t = np.argwhere(trace_t==-1)+1 # add one as python indexing doesn't include last index in range
        
        
        # exclude border saccades in ground truth data:
        if (truth[i,0]==1):
            end_t = end_t[1:]
        if (truth[i,-1]==1):
            start_t = start_t[:-1]
    
        if len(start_t)!=0: # if there is at least one saccade
            
            # if all conditions met, loop through each true saccade
            for j in range(len(start_t)):
                
                content_of_pred = pred_copy[i,int(start_t[j]):int(end_t[j])] #content of prediction during true saccade duration 
                
                if content_of_pred!=[]:
                    if np.mean(content_of_pred)==0:
                        # checks if no saccade has been detected by network
                        # add false negative detection
                        false_neg += 1
                    else:
                        
                        if content_of_pred[0]==1: # the matching predicted saccade start earlier than the ground truth
                            
                            # if the predicted label starts at the signal boundary
                            pred_start_ind = np.argwhere(np.diff(pred_copy[i,:])==1)
                            ind_before_true_start = pred_start_ind[pred_start_ind<start_t[j]] # predicted start before true start
                                                         
                            if len(ind_before_true_start)==0:
                                start = 0; # set predicted start to first bin because it is before the start of the trace
                            else:

                                start = ind_before_true_start[-1] + 1

                            diff_pred_copy = np.diff(pred_copy[i,:])
                            if pred_copy[i,-1]==1:
                                #if the prediction ends with a saccade, set the last bind of the differential to -1 to add an end
                                diff_pred_copy[-1] = -1 
                                
                            # get the predicted end bin:    
                            pred_end_ind = np.argwhere(diff_pred_copy==-1) +1 #get all saccade end bins in trace
                            ind_after_start = pred_end_ind[pred_end_ind>start] #only consider end bins that occur after start
                            end = ind_after_start[0] # use first end bin after predicted saccade start as saccade end

                            pred_copy[i,cluster_belonging (pred_copy[i,:].astype(bool),start_t[j])]=0   # delete matching pairs
                            truth_copy[i,cluster_belonging (truth_copy[i,:],start_t[j])]=0 # to not consider them for the next iteration
                            true_pos += 1
                            
                        else:
                            pred_start_ind = np.argwhere(np.diff(pred_copy[i,:])==1)
                            # predicted start from true start onwards
                            ind_from_true_start = pred_start_ind[pred_start_ind>=start_t[j]] 
                             
                            start = ind_from_true_start[0] +1 # set predicted start index to first found after true start
                            
                            diff_pred_copy=np.diff(pred_copy[i,:])
                            if pred_copy[i,-1]==1:
                                diff_pred_copy[-1]=-1
                                
                            # get the predicted end bin:     
                            pred_end_ind = np.argwhere(diff_pred_copy==-1) +1 #get all saccade end bins in trace
                            ind_after_start = pred_end_ind[pred_end_ind>start] #only consider end bins that occur after start
                            end = ind_after_start[0] # use first end bin after predicted saccade start as saccade end

                            pred_copy[i,cluster_belonging (pred_copy[i,:],start)]=0   # delete matching pairs
                            truth_copy[i,cluster_belonging (truth_copy[i,:],start)]=0 
                            
                            true_pos += 1
                        
                        on_distance.append(int(start)-int(start_t[j]))
                        off_distance.append(int(end)-int(end_t[j]))
    
        # remove border sacades from predicted trace
        if (pred_copy[i,0]==1):
            pred_copy[i,cluster_belonging (pred_copy[i,:],0)]=0 
        if (pred_copy[i,-1]==1):
            pred_copy[i,cluster_belonging (pred_copy[i,:],len(pred_copy[i,:])-1)]=0 
            
        # count false positives 
        false_pos=false_pos+len(np.argwhere(np.diff(pred_copy[i,:])==1))
        
    return true_pos,false_pos,false_neg,on_distance,off_distance


def EM_saccade_detection(X,Y,lambda_param=6,min_sacc_dist=5,min_sacc_dur=10,sampfreq=1000):
    # Addapted from Engbert, Ralf, and Konstantin Mergenthaler. "Microsaccades are triggered by low retinal image slip." Proceedings of the National Academy of Sciences 103.18 (2006): 7192-7197.

    n_samples,n_time = X.shape
    Vx=np.zeros((n_samples,n_time))
    Vy=np.zeros((n_samples,n_time))
    Eta_x=np.zeros(n_samples)
    Eta_y=np.zeros(n_samples)
    Sacc_out=np.zeros((n_samples,n_time),dtype=bool)
    for i in np.arange(0,np.shape(X)[0]):
        Vx[i,:]=np.convolve(X[i,:], np.divide([-1,-1,0,1,1],6), mode='same') # running average over 5 bins
        Vy[i,:]=np.convolve(Y[i,:], np.divide([-1,-1,0,1,1],6), mode='same')
        # set edges values to the last values
        Vx[i,0:2]=Vx[i,3]
        Vy[i,0:2]=Vy[i,3]
        Vx[i,n_time-2:n_time]=Vx[i,n_time-3]
        Vy[i,n_time-2:n_time]=Vy[i,n_time-3]
        # compute thresholds
        Eta_x[i]=np.median((Vx[i,:]-np.median(Vx[i,0:2]))**2);
        Eta_y[i]=np.median((Vy[i,:]-np.median(Vy[i,0:2]))**2);
        # saccade estimate
        
        Sacc_out[i,:]=(np.divide(Vx[i,:],(lambda_param*np.sqrt(Eta_x[i])))**2+np.divide(Vy[i,:],(lambda_param*np.sqrt(Eta_y[i])))**2)>1
        
        # merge saccades closer than min_sacc_dist (ms)
        Sacc_out[i,:] = merge_saccades(Sacc_out[i,:],sampfreq,min_sacc_dist)
        
        labels = label(Sacc_out[i,:])
        for j in np.arange(1,np.max(labels)+1):
            
            if np.sum(labels==j)<int(min_sacc_dur*(sampfreq/1000)): # remove saccades smaller than min_sacc_dur (ms)
                
                Sacc_out[i,labels==j]=False
    return Sacc_out
