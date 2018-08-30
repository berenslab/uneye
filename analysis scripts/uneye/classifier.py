"""

Convolutional Neural Network for Saccade Detection
Bellet et al. 2018
Contact: marie-estelle.bellet@student.uni-tuebingen.de
"""
from uneye.functions import *
import numpy as np
import os
import math
from skimage.measure import label
from scipy import io
from IPython import display
# Pytorch imports:
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score as cohenskappa

###############################
############ U'n'Eye: ###########
###############################

class DNN():
    '''
    Convolutional Neural Network (U-Net Variant) for Saccade Detection
    
    Parameters
    ----------
    max_iter: int, maximum number of epochs during training, default=500
        
    sampfreq: {float,int}, sampling frequency of data in Hz, default=1000 Hz
    
    lr: float, learning rate, default=0.001
    
    weights_name: str, filename of weights to save or load, will automatically be stored and load in local folder 'training', default: 'weights'
    
    n_classes: int, number of classes to predict or present in ground truth, default=2
    
    min_sacc_dist: int, minimum distance between two saccades in ms for merging of saccades, default=1
    
    min_sacc_dur: int, minimum saccade duration in ms for removal of small events, default=6ms
        
    augmentation: bool, whether or not to use data augmentation for training. Default: True
    
    '''
    def __init__(self, max_iter=500, sampfreq=1000,
                 lr=0.001, weights_name='weights',
                classes=2,min_sacc_dist=1,
                 min_sacc_dur=6,augmentation=True,
                 ks=5,mp=5):
        
        if max_iter<10:
            max_iter = 10
        self.max_iter = max_iter
        self.sampfreq = sampfreq
        self.lr = lr
        self.weights_name = weights_name
        self.classes = classes
        self.min_sacc_dist = min_sacc_dist
        self.min_sacc_dur = min_sacc_dur
        self.augmentation = augmentation
        self.net = UNet(classes,ks,mp)
        self.mp = mp
        self.use_gpu = torch.cuda.is_available()

    def train(self,X,Y,Labels,seed=1):
        '''
        Train the model according to the given training data and store the trained weights.
        
        Parameters
        ----------
        X,Y: array-like, shape: (n_samples, n_timepoints), horizontal and vertical eye positions in degree
        
        Labels: array-like, shape: (n_samples, n_timepoints), event labels: fixation=0, saccade=1 (other classes optional)
        
        seed: int, seed for weights initialization and batch shuffling, default=1
        
        '''
        # set random seed
        np.random.seed(1)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) #fixed seed to control random data shuffling in each epoch
        
        classes = self.classes

        # check if data has right dimensions (2)
        xdim,ydim,ldim = X.ndim,Y.ndim,Labels.ndim
        if any((xdim!=2,ydim!=2,ldim!=2)):
            # reshape into matrix with trials of length=1sec
            trial_len = int(self.sampfreq) #trials of 1 sec
            time_points = len(X)
            n_trials = int(time_points/trial_len)
            X = np.reshape(X[:n_trials*trial_len],(n_trials,trial_len))
            Y = np.reshape(Y[:n_trials*trial_len],(n_trials,trial_len))
            Labels = np.reshape(Labels[:n_trials*trial_len],(n_trials,trial_len))

            
        n_samples,n_time = X.shape
        
        # multi class labels
        Labels_mc = np.zeros((n_samples,n_time,classes))
        for c in range(classes):
            Labels_mc[:,:,c] = Labels==c
        Labels = Labels_mc

        # data shuffling 
        randind = np.random.permutation(n_samples)
        X = X[randind,:]
        Y = Y[randind,:]
        Labels = Labels[randind,:]
        
        # check if number of timebins is multiple of the maxpooling kernel size squared, otherwise cut:
        n_time2 = X.shape[1]
        if n_time%(self.mp**2)!=0:
            X = X[:,:int(np.floor(n_time/(self.mp**2))*(self.mp**2))]
            Y = Y[:,:int(np.floor(n_time/(self.mp**2))*(self.mp**2))]
            Labels = Labels[:,:int(np.floor(n_time/(self.mp**2))*(self.mp**2)),:]
     
        # validation and training set
        # 50 samples of training data used for validation
        n_validation = 50 #fixed number of validation samples independent of number of training samples
        n_training = n_samples - n_validation
        Xval = X[:n_validation,:]
        Yval = Y[:n_validation,:]
        Lval = Labels[:n_validation,:]
        Xtrain = X[n_validation:,:]
        Ytrain = Y[n_validation:,:]
        Ltrain = Labels[n_validation:,:]

        if self.augmentation==True:
            # data augmentation: signal rotation
            theta = np.arange(0.25,2,0.5)
            r = np.sqrt(Xtrain**2+Ytrain**2)
            x = Xtrain.copy()
            y = Ytrain.copy()
            for t in theta:
                x2 = x.copy()*math.cos(np.pi * t) + y.copy()*math.sin(np.pi * t)
                y2 = -x.copy()*math.sin(np.pi * t) + y.copy()*math.cos(np.pi * t)
                Xtrain = np.concatenate((Xtrain.copy(),x2),0)
                Ytrain = np.concatenate((Ytrain.copy(),y2),0)
                Ltrain = np.concatenate((Ltrain,Ltrain),0)

                          
        n_training = Xtrain.shape[0]
        
        # Velocity:
        # training data
        Xdiff = np.diff(Xtrain,axis=-1)
        Xdiff = np.concatenate((np.zeros((n_training,1)),Xdiff),1)
        Xdiff[np.isinf(Xdiff)] = 1.5
        Xdiff[np.isnan(Xdiff)] = 0
        Ydiff = np.diff(Ytrain,axis=-1)
        Ydiff[np.isnan(Ydiff)] = 0
        Ydiff[np.isinf(Ydiff)] = 1.5
        Ydiff = np.concatenate((np.zeros((n_training,1)),Ydiff),1)  
        # input matrix:
        V = np.tile((Xdiff,Ydiff),1)
        V = np.swapaxes(np.swapaxes(V,0,1),1,2) 
        # torch Variable:
        Vtrain = Variable(torch.FloatTensor(V).unsqueeze(1),requires_grad=False)
        Ltrain = np.swapaxes(Ltrain,1,2)
        Ltrain = Variable(torch.FloatTensor(Ltrain.astype(float)),requires_grad=False)
        
        # validation data
        Xdiff = np.diff(Xval,axis=-1)
        Xdiff = np.concatenate((np.zeros((n_validation,1)),Xdiff),1)
        Xdiff[np.isinf(Xdiff)] = 1.5
        Xdiff[np.isnan(Xdiff)] = 0
        Ydiff = np.diff(Yval,axis=-1)
        Ydiff[np.isnan(Ydiff)] = 0
        Ydiff[np.isinf(Ydiff)] = 1.5
        Ydiff = np.concatenate((np.zeros((n_validation,1)),Ydiff),1)  
        # input matrix:
        V = np.tile((Xdiff,Ydiff),1)
        V = np.swapaxes(np.swapaxes(V,0,1),1,2) 
        # torch Variable:
        Vval = Variable(torch.FloatTensor(V).unsqueeze(1),requires_grad=False)
        Lval = np.swapaxes(Lval,1,2)
        Lval = Variable(torch.FloatTensor(Lval.astype(float)),requires_grad=False)
        
        # model
        self.net.apply(weights_init)
        self.net.train()
        # send to gpu is cuda enabled
        
        if self.use_gpu:
            self.net.cuda()
            Vval = Vval.cuda()
            Lval = Lval.cuda()
            
        # learning parameters
        criterion = MCLoss()
        optimizer = optim.Adam(self.net.parameters(),lr=self.lr)
        l2_lambda = 0.001 #factor for L2 penalty
        iters = 10 #iterations per epoch
        batchsize = int(np.floor(n_training/iters))
        
        # output folder:
        out_folder = './training'
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            
        epoch = 1
        Loss_val = [] #validation loss storage
        Loss_train = [] #training loss storage
        key = ['out']
        save_weights = False
        getting_worse = 0
        while epoch<=self.max_iter:

            # shuffle training data in each epoch:
            rand_ind = torch.randperm(n_training)
            Vtrain = Vtrain[rand_ind,:]
            Ltrain = Ltrain[rand_ind,:]
            
            loss_train = np.zeros(iters)
            for niter in range(iters):    
                # Minibatches:
                if niter!=iters-1:
                    Vbatch = Vtrain[niter*batchsize:(niter+1)*batchsize,:]
                    Lbatch = Ltrain[niter*batchsize:(niter+1)*batchsize,:]        
                else:
                    Vbatch = Vtrain[niter*batchsize:,:]
                    Lbatch = Ltrain[niter*batchsize:,:]
                if self.use_gpu:
                    Vbatch = Vbatch.cuda()
                    Lbatch = Lbatch.cuda()
                optimizer.zero_grad()
                
                out = self.net(Vbatch,key)[0] # network output
                loss = criterion(out,Lbatch) #loss
                reg_loss = 0
                for param in self.net.parameters():
                    reg_loss += torch.sum(param**2) #L2 penalty
            
                loss += l2_lambda * reg_loss
                loss_train[niter] = loss.data.cpu().numpy() #loss storage in each iteration
                loss.backward() #back propagation
                optimizer.step()
                
            Loss_train.append(np.mean(loss_train)) #store training loss
            
            print('Iteration: '+str(epoch)+'/'+str(self.max_iter))
            display.clear_output(wait=True)
            
            # test in every epoch:
            # validation loss
            out_val = self.net(Vval,key)[0]
            loss_val = criterion(out_val,Lval)
            reg_loss_val = 0
            for param in self.net.parameters():
                reg_loss_val += torch.sum(param**2) #L2 penalty
            loss_val += l2_lambda * reg_loss_val
            Loss_val.append(loss_val.data.numpy())
            if len(Loss_val)>3:
                if Loss_val[-1]<float(np.mean(Loss_val[-4:-1])): #validation performance better than average over last 3
                    getting_worse = 0
                    if Loss_val[-1]<best_loss:
                        best_loss = Loss_val[-1]
                        uneye_weights = self.net.state_dict() #store weights
                        save_weights = True
                    else:
                        self.net.load_state_dict(uneye_weights) #load weights of last time when loss was lower
                        
                else: #validation performance worse than last
                    #learning rate decay:
                    optimizer = lr_decay(optimizer) #reduce learning rate by a fixed step
                    getting_worse += 1 
                    self.net.load_state_dict(uneye_weights) #load weights of last time when loss was lower
            else:
                best_loss = np.min(Loss_val)
                uneye_weights = self.net.state_dict()
                
            if getting_worse>3:
                # stop the training if the loss is increasing for the validation set
                self.net.load_state_dict(uneye_weights) #get back best weights
                print('Early stopping at epoch '+str(epoch-1)+' before overfitting occurred.')
                epoch = self.max_iter+1 
            epoch += 1
  
        # validate after training to ensure saving best weights
        out_val = self.net(Vval,key)[0]
        loss_val = criterion(out_val,Lval).data[0]
        if loss_val<best_loss:            
            uneye_weights = self.net.state_dict()
            save_weights = True
        # save weights
        if save_weights:
            self.net.load_state_dict(uneye_weights)
            if self.use_gpu:
                K = list(uneye_weights.keys())
                for i,k in enumerate(K):
                    uneye_weights[k] = uneye_weights[k].cpu()
                self.net.cpu()
            torch.save(uneye_weights,os.path.join(out_folder,self.weights_name))
            print("Model parameters saved to",os.path.join(out_folder,self.weights_name))
        else:
            print("Model parameters could not be saved due to early overfitting. Try to reduce learning rate or increase number of training samples.")

        
        self.loss_val = Loss_val
        self.loss_train = Loss_train
        
        return self
        
        
        
    def predict(self,X,Y):       
        '''
        Predict Saccades with trained weights.
        
        Parameters
        ----------
        X,Y: array-like, shape: {(n_timepoints),(n_samples, n_timepoints)}, horizontal and vertical eye positions in degree
                
        Output
        ------
        Pred: array-like, shape: {(classes, n_timepoints),(n_samples, n_timepoints)}, class prediction, values in range [0 classes-1], fixation=0, saccades=1
        
        Prob: array-like, shape: {(classes, n_timepoints),(n_samples, classes, n_timepoints)}, class probabilits (network softmax output)
        
        '''
        n_dim = len(X.shape)
        classes = self.classes
        if n_dim==1:
            X = np.atleast_2d(X)
            Y = np.atleast_2d(Y)
        if X.shape[1]<25:
            raise ValueError('Input is to small along dimension 1. Expects input of the form (n_samples x n_bins) or (n_bins).')
        
        n_samples,n_time = X.shape

         # differentiated signal:
        Xdiff = np.diff(X,axis=-1)
        Xdiff = np.concatenate((np.zeros((n_samples,1)),Xdiff),1)
        Xdiff[np.isinf(Xdiff)] = 1.5
        Xdiff[np.isnan(Xdiff)] = 0
        Ydiff = np.diff(Y,axis=-1)
        Ydiff[np.isnan(Ydiff)] = 0
        Ydiff[np.isinf(Ydiff)] = 1.5
        Ydiff = np.concatenate((np.zeros((n_samples,1)),Ydiff),1)
        
        # input matrix:
        V = np.tile((Xdiff,Ydiff),1)
        V = np.swapaxes(np.swapaxes(V,0,1),1,2)
        # torch Variable:
        V = Variable(torch.FloatTensor(V).unsqueeze(1),requires_grad=False)
        
        # load pretrained model
        if os.path.isabs(self.weights_name):
            weights = torch.load(self.weights_name)
        else:
            weights = torch.load(os.path.join('training',self.weights_name))
        self.net.load_state_dict(weights)   
        self.net.eval()
        
        # send to gpu if cuda enabled
        if self.use_gpu:
            self.net.cuda()
        
        #predict in batches so that batchnorm works
        batchsize = 50
        iters = int(np.ceil(n_samples/batchsize))
        n_time2 = V.size()[2]
        Pred = np.zeros((n_samples,n_time))
        Prob = np.zeros((n_samples,classes,n_time))
        for niter in range(iters):    
            # Minibatches:
            if niter!=iters-1:
                Vbatch = V[niter*batchsize:(niter+1)*batchsize,:]
            else:
                Vbatch = V[niter*batchsize:,:]
                
            # send to gpu if cuda is enabled
            if self.use_gpu:
                Vbatch = Vbatch.cuda()
                
            # check if number of timepoints is a multiple of the maxpooling kernel size squared:
            remaining = n_time2%(self.mp**2)
            # if not, evaluate segment-wise and concatenante output
            if remaining!=0:
                first_time_batch = int(np.floor(n_time2/(self.mp**2))*(self.mp**2))
                Vbatch1 = Vbatch[:,:,:first_time_batch,:]
                Vbatch2 = Vbatch[:,:,-(self.mp**2):,:]
                Out1 = self.net(Vbatch1,['out'])[0].data.cpu().numpy()
                Out2 = self.net(Vbatch2,['out'])[0].data.cpu().numpy()
                Out = np.concatenate((Out1,Out2[:,:,-remaining:]),2)
            
            else:
                Out = self.net(Vbatch,['out'])[0].data.cpu().numpy() 
            
            # Prediction:
            if classes==2:
                Prediction = binary_prediction(Out[:,1,:],
                                               self.sampfreq,
                                               min_sacc_dist=self.min_sacc_dist,
                                               min_sacc_dur=int(self.min_sacc_dur/(1000/self.sampfreq)))
                
            else:
                Prediction = np.argmax(Out,1)
            Probability = Out
            if niter!=iters-1:
                Pred[niter*batchsize:(niter+1)*batchsize,:] = Prediction
                Prob[niter*batchsize:(niter+1)*batchsize,:] = Probability
            else:
                Pred[niter*batchsize:,:] = Prediction
                Prob[niter*batchsize:,:] = Probability
        # if input one dimensional, reduce back to one dimension:
        if n_dim==1:
            Pred = Pred[0,:]   
            Prob = Prob[0,:]
        
        
        return Pred,Prob 
    
    
    def test(self,X,Y,Labels):       
        '''
        Predict Saccades with trained weights and test performance against given labels.

        Parameters
        ----------
        X,Y: array-like, shape: {(n_timepoints),(n_samples, n_timepoints)}, horizontal and vertical eye positions in degree
        
        Labels: array-like, shape: {(n_timepoints),(n_samples, n_timepoints)}, class labels in range [0 classes-1], fixation=0, saccades=1
                
        Output
        ------
        Pred: array-like, shape: {(classes, n_timepoints),(n_samples, n_timepoints)}, class prediction, values in range [0 classes-1], fixation=0, saccades=1
        
        Prob: array-like, shape: {(classes, n_timepoints),(n_samples, classes, n_timepoints)}, class probabilits (network softmax output)
        
        Performance: dict with keys: 
            {'kappa': cohen's kappa values for all classes, 
            'fpr': false positive rate (saccades),
            'tpr': true positive rate (saccades),
            'auroc': area under the ROC (saccades),
            'f1': harmonic mean of precision and recall (saccades)
            'on': onset difference in timebins for true positive saccades
            'off': offset difference in timebins for true positive saccades }
            
        '''
        classes = self.classes
        n_dim = len(X.shape)
        if n_dim==1:
            X = np.atleast_2d(X)
            Y = np.atleast_2d(Y)
        if X.shape[1]<25:
            raise ValueError('Input is to small along dimension 1. Expects input of the form (n_samples x n_bins) or (n_bins).')
        
        n_samples,n_time = X.shape
   
        # differentiated signal:
        Xdiff = np.diff(X,axis=-1)
        Xdiff = np.concatenate((np.zeros((n_samples,1)),Xdiff),1)
        Xdiff[np.isinf(Xdiff)] = 1.5
        Xdiff[np.isnan(Xdiff)] = 0
        Ydiff = np.diff(Y,axis=-1)
        Ydiff[np.isnan(Ydiff)] = 0
        Ydiff[np.isinf(Ydiff)] = 1.5
        Ydiff = np.concatenate((np.zeros((n_samples,1)),Ydiff),1)
        
        # input matrix:
        V = np.tile((Xdiff,Ydiff),1)
        V = np.swapaxes(np.swapaxes(V,0,1),1,2)
        # torch Variable:
        V = Variable(torch.FloatTensor(V).unsqueeze(1),requires_grad=False)
        
        # load pretrained model
        if os.path.isabs(self.weights_name):
            weights = torch.load(self.weights_name)
        else:
            weights = torch.load(os.path.join('training',self.weights_name))
        self.net.load_state_dict(weights)   
        self.net.eval()
        
        # send to gpu if cuda enabled
        if self.use_gpu:
            self.net.cuda()
            
        #predict in batches
        batchsize = 50
        iters = int(np.ceil(n_samples/batchsize))
        n_time2 = V.size()[2]
        Pred = np.zeros((n_samples,n_time))
        if classes==2:
            Prob = np.zeros((n_samples,n_time))
        else:
            Prob = np.zeros((n_samples,classes,n_time))
            
        for niter in range(iters):    
            # Minibatches:
            if niter!=iters-1:
                Vbatch = V[niter*batchsize:(niter+1)*batchsize,:]
            else:
                Vbatch = V[niter*batchsize:,:]
                
            # send to gpu if cuda is enabled
            if self.use_gpu:
                Vbatch = Vbatch.cuda()
                
            # check if number of timepoints is a multiple of the maxpooling kernel size squared:
            remaining = n_time2%(self.mp**2)
            if remaining!=0:
                first_time_batch = int(np.floor(n_time2/(self.mp**2))*(self.mp**2))
                Vbatch1 = Vbatch[:,:,:first_time_batch,:]
                Vbatch2 = Vbatch[:,:,-(self.mp**2):,:]
                Out1 = self.net(Vbatch1,['out'])[0].data.cpu().numpy()
                Out2 = self.net(Vbatch2,['out'])[0].data.cpu().numpy()
                Out = np.concatenate((Out1,Out2[:,:,-remaining:]),2)
            
            else:
                Out = self.net(Vbatch,['out'])[0].data.cpu().numpy()
            
            
            # Prediction:
            if classes==2:
                Prediction = binary_prediction(Out[:,1,:],
                                               self.sampfreq,
                                               min_sacc_dist=self.min_sacc_dist,
                                               min_sacc_dur=int(self.min_sacc_dur/(1000/self.sampfreq)))
                Probability = Out[:,1,:]
            else:
                Prediction = np.argmax(Out,1)
                Probability = Out
                
            if niter!=iters-1:
                Pred[niter*batchsize:(niter+1)*batchsize,:] = Prediction
                Prob[niter*batchsize:(niter+1)*batchsize,:] = Probability
            else:
                Pred[niter*batchsize:,:] = Prediction
                Prob[niter*batchsize:,:] = Probability
        
        # PERFORMANCE
        # Cohen's Kappa, AUROC and f1
        if classes==2: 
            # if only two target classes (fixation and saccade), calculate performance measures for saccade detection
            pred = Pred==1
            Kappa = cohenskappa((Labels==1).astype(float).flatten(),pred.astype(float).flatten())
            print('Binary Cohens Kappa: ',np.round(Kappa,3))
            true_pos,false_pos,false_neg,on_distance,off_distance = accuracy(Pred.astype(float),(Labels==1).astype(float))
        else:
            # if multiple target classes, get cohen's kappa for all classes and auroc and f1 for saccades only (class 1)
            # cohen's kappa
            Kappa = np.zeros(classes)
            for c in range(classes):
                pred = Pred==c
                kappa = cohenskappa((Labels==c).astype(float).flatten(),pred.astype(float).flatten())
                Kappa[c] = kappa
                print('Cohens Kappa class',c,': ',np.round(kappa,3))
            true_pos,false_pos,false_neg,on_distance,off_distance = accuracy((Pred==1).astype(float),(Labels==1).astype(float))
        f1 = (2 * true_pos)/(2 * true_pos + false_neg + false_pos)    
        print('F1:',np.round(f1,3))
        
        Performance = {
            'kappa': Kappa,
            'f1': f1,
            'on': on_distance,
            'off': off_distance
            }
        
        # if input one dimensional, reduce back to one dimension:
        if n_dim==1:
            Pred = Pred[0,:]   
            Prob = Prob[0,:]
        
        
        return Pred, Prob, Performance
    
    
    def crossvalidate(self,X,Y,Labels,X_val,Y_val,Labels_val,Labels_test=None,K=10):
        '''
        Use K-fold cross validation.
        Measuring performance in terms of Cohen's Kappa, F1 and AUROC.
        
        Parameters
        ----------
        X,Y: array-like, shape: {(n_timepoints),(n_samples, n_timepoints)}, horizontal and vertical eye positions in degree
        
        Labels: array-like, shape: {(n_timepoints),(n_samples, n_timepoints)}, class labels in range [0 classes-1], fixation=0, saccades=1
        
        X_val,Y_val: array-like, shape: {(n_timepoints),(n_samples, n_timepoints)}, additional horizontal and vertical eye positions in degree for validation
        
        Labels_val: array-like, shape: {(n_timepoints),(n_samples, n_timepoints)}, additional class labels in range [0 classes-1], fixation=0, saccades=1 for validation
        
        Labels_test: array-like, shape: {(n_timepoints),(n_samples, n_timepoints)}, if test Labels different from training labels (for training with missing labels only), optional
        
        K: float, number of folds of cross validation
                
        
        Output
        ------

        Performance: dict with keys: 
            {'kappa': cohen's kappa values for all classes, 
            'fpr': false positive rate (saccades),
            'tpr': true positive rate (saccades),
            'auroc': area under the ROC (saccades),
            'f1': harmonic mean of precision and recall (saccades)
            'on': onset difference in timebins for true positive saccades
            'off': offset difference in timebins for true positive saccades }
            
        
        ''' 

        # check if data has right dimensions (2)
        xdim,ydim,ldim = X.ndim,Y.ndim,Labels.ndim
        if any((xdim!=2,ydim!=2,ldim!=2)):
            # reshape into matrix with trials of length=1sec
            # training set
            trial_len = int(1000 * self.sampfreq/1000)
            time_points = len(X_val)
            n_trials = int(time_points/trial_len)
            X = np.reshape(X[:n_trials*trial_len],(n_trials,trial_len))
            Y = np.reshape(Y[:n_trials*trial_len],(n_trials,trial_len))
            Labels = np.reshape(Labels[:n_trials*trial_len],(n_trials,trial_len))
            # validation set
            time_points = len(X_val)
            n_trials = int(time_points/trial_len)
            X_val = np.reshape(X_val[:n_trials*trial_len],(n_trials,trial_len))
            Y_val = np.reshape(Y_val[:n_trials*trial_len],(n_trials,trial_len))
            Labels_val = np.reshape(Labels_val[:n_trials*trial_len],(n_trials,trial_len))

        n_samples,n_time = X.shape
        classes = self.classes
        
        Labels_mc = np.zeros((n_samples,n_time,classes))
        Labels_mc_val = np.zeros((Labels_val.shape[0],n_time,classes))
        for c in range(classes):
            Labels_mc[:,:,c] = Labels==c
            Labels_mc_val[:,:,c] = Labels_val==c
        Labels = Labels_mc
        Labels_val = Labels_mc_val

        if Labels_test is None: #if no alternative test labels given, use training labels for testing
            Labels_test = Labels.copy() 
      
        # check if number of timebins is multiple of the maxpooling kernel size squared, otherwise cut:
        fac = (self.mp**2)
        if n_time%fac!=0:
            X = X[:,:int(np.floor(n_time/fac)*fac)]
            Y = Y[:,:int(np.floor(n_time/fac)*fac)]
            X_val = X_val[:,:int(np.floor(n_time/fac)*fac)]
            Y_val = Y_val[:,:int(np.floor(n_time/fac)*fac)]

            Labels = Labels[:,:int(np.floor(n_time/fac)*fac),:]
            Labels_test = Labels_test[:,:int(np.floor(n_time/fac)*fac),:]
            Labels_val = Labels_val[:,:int(np.floor(n_time/fac)*fac),:]
            n_time = X.shape[1]

         # prepare validation data: same for all cross validations
        Lval = Labels_val.copy()
        n_val_samp = X_val.shape[0]
        # differentiated signal:
        Xdiff = np.diff(X_val,axis=-1)
        Xdiff = np.concatenate((np.zeros((n_val_samp,1)),Xdiff),1)
        Xdiff[np.isinf(Xdiff)] = 1.5
        Xdiff[np.isnan(Xdiff)] = 0
        Ydiff = np.diff(Y_val,axis=-1)
        Ydiff[np.isnan(Ydiff)] = 0
        Ydiff[np.isinf(Ydiff)] = 1.5
        Ydiff = np.concatenate((np.zeros((n_val_samp,1)),Ydiff),1) 
        # input matrix:
        V = np.tile((Xdiff,Ydiff),1)
        V = np.swapaxes(np.swapaxes(V,0,1),1,2) 
        # torch Variable:
        Vval = Variable(torch.FloatTensor(V).unsqueeze(1),requires_grad=False)
        Lval = np.swapaxes(Lval,1,2)
        Lval = Variable(torch.FloatTensor(Lval.astype(float)),requires_grad=False)
        
        n_test = int(n_samples/K)
        np.random.seed(1) # fixed seed for comparable cross validations
        
        indices = np.random.permutation(n_samples)
        # Cross Validation:  

        Kappa = np.zeros((K,classes))
        F1 = np.zeros(K)
        On = []
        Off = []
        for i in range(K):
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1) #fixed seed to control random data shuffling in each epoch
            print(str(i+1)+'. cross validation...')
            ind_train = indices.copy()
            if i==K-1:
                ind_train = np.array(np.delete(ind_train,range(n_test*i,n_samples)))
                ind_test = indices[n_test*i:]
            else:
                ind_train = np.array(np.delete(ind_train,range(n_test*i,n_test*(i+1))))
                ind_test = indices[n_test*i:n_test*(i+1)]
            # training and test set
            Xtrain = X[ind_train,:].copy()
            Ytrain = Y[ind_train,:].copy()
            Xtest = X[ind_test,:].copy()
            Ytest = Y[ind_test,:].copy()
            
            Ltrain = Labels[ind_train,:].copy()  
            Ltest = Labels_test[ind_test,:].copy()

            Ltrain = np.swapaxes(Ltrain,1,2)
            Ltest = np.swapaxes(Ltest,1,2)
                
            # data augmentation: signal rotation
            theta = np.arange(0.25,2,0.5)
            r = np.sqrt(Xtrain**2+Ytrain**2)
            x = Xtrain.copy()
            y = Ytrain.copy()
            for t in theta:
                x2 = x.copy()*math.cos(np.pi * t) + y.copy()*math.sin(np.pi * t)
                y2 = -x.copy()*math.sin(np.pi * t) + y.copy()*math.cos(np.pi * t)
                Xtrain = np.concatenate((Xtrain.copy(),x2),0)
                Ytrain = np.concatenate((Ytrain.copy(),y2),0)
                Ltrain = np.concatenate((Ltrain,Ltrain),0)
            
            # Prepare Training data:
            n_training = Xtrain.shape[0]
            # differentiated signal:
            Xdiff = np.diff(Xtrain,axis=-1)
            Xdiff = np.concatenate((np.zeros((n_training,1)),Xdiff),1)
            Xdiff[np.isinf(Xdiff)] = 1.5
            Xdiff[np.isnan(Xdiff)] = 0
            Ydiff = np.diff(Ytrain,axis=-1)
            Ydiff[np.isnan(Ydiff)] = 0
            Ydiff[np.isinf(Ydiff)] = 1.5
            Ydiff = np.concatenate((np.zeros((n_training,1)),Ydiff),1) 
            # input matrix:
            V = np.tile((Xdiff,Ydiff),1)
            V = np.swapaxes(np.swapaxes(V,0,1),1,2) 
            # torch Variable:
            Vtrain = Variable(torch.FloatTensor(V).unsqueeze(1),requires_grad=False)
            Ltrain = Variable(torch.FloatTensor(Ltrain.astype(float)),requires_grad=False)
            
            # Prepare Test data:
            n_test_samp = Xtest.shape[0]
            # differentiated signal:
            Xdiff = np.diff(Xtest,axis=-1)
            Xdiff = np.concatenate((np.zeros((n_test_samp,1)),Xdiff),1)
            Xdiff[np.isinf(Xdiff)] = 1.5
            Xdiff[np.isnan(Xdiff)] = 0
            Ydiff = np.diff(Ytest,axis=-1)
            Ydiff[np.isnan(Ydiff)] = 0
            Ydiff[np.isinf(Ydiff)] = 1.5
            Ydiff = np.concatenate((np.zeros((n_test_samp,1)),Ydiff),1) 
            # input matrix:
            V = np.tile((Xdiff,Ydiff),1)
            V = np.swapaxes(np.swapaxes(V,0,1),1,2) 
            # torch Variable:
            Vtest = Variable(torch.FloatTensor(V).unsqueeze(1),requires_grad=False)
            Ltest = Ltest.astype(float)
            
            # model
            self.net.apply(weights_init)
            self.net.train()
            
            # send to gpu is cuda enabled
            if self.use_gpu:
                self.net.cuda()
                Vtest = Vtest.cuda()
                Vval = Vval.cuda()
                Lval = Lval.cuda()
                
            # learning parameters
            criterion = MCLoss()
            optimizer = optim.Adam(self.net.parameters(),lr=self.lr)
            l2_lambda = 0.001     
            iters = 10 #iterations per epoch
            batchsize = int(np.floor(n_training/iters))
                
            epoch = 1
            L = [] #validation loss storage
            Loss_train = [] #training loss storage
            key = ['out'] #layer to output
            getting_worse = 0
            while epoch<self.max_iter:                
                # shuffle training data in each epoch:
                rand_ind = torch.randperm(n_training)
                Vtrain = Vtrain[rand_ind,:]
                Ltrain = Ltrain[rand_ind,:]
                loss_train = np.zeros(iters) #preallocate vector for loss storage
                for niter in range(iters):    
                    # Minibatches:
                    if niter!=iters-1:
                        Vbatch = Vtrain[niter*batchsize:(niter+1)*batchsize,:]
                        Lbatch = Ltrain[niter*batchsize:(niter+1)*batchsize,:]        
                    else:
                        Vbatch = Vtrain[niter*batchsize:,:]
                        Lbatch = Ltrain[niter*batchsize:,:]
                        
                    # send to gpu if cuda is enabled
                    if self.use_gpu:
                        Vbatch = Vbatch.cuda()
                        Lbatch = Lbatch.cuda()
                        
                    optimizer.zero_grad()
                    out = self.net(Vbatch,key)[0]
                    loss = criterion(out,Lbatch)
                    loss_train[niter] = loss.data.cpu().numpy() #store loss in each iteration
                    
                    reg_loss = 0
                    for param in self.net.parameters():
                        reg_loss += torch.sum(param**2)
                
                    loss += l2_lambda * reg_loss
                    loss.backward()
                    optimizer.step()
                    
                Loss_train.append(np.mean(loss_train)) #append average loss over all iterations
                               
                # validate every epoch:
                # validation loss
                out_val = self.net(Vval,key)[0]
                loss_val = criterion(out_val,Lval)
                reg_loss_val = 0
                for param in self.net.parameters():
                    reg_loss_val += torch.sum(param**2) #L2 penalty
                loss_val += l2_lambda * reg_loss_val
                L.append(loss_val.data[0])
                if len(L)>3:
                    if L[-1]<np.mean(L[-4:-1]): #validation performance better than last
                        getting_worse = 0
                        if L[-1]<best_loss:
                            best_loss = L[-1]
                            uneye_weights = self.net.state_dict() #store weights
                            save_weights = True
                        else:
                            self.net.load_state_dict(uneye_weights) #load best weights
                            
                    else: #validation performance worse than last
                        #learning rate decay:
                        optimizer = lr_decay(optimizer) #reduce learning rate by a fixed step
                        getting_worse +=1 
                        self.net.load_state_dict(uneye_weights) #load best weights
                else:
                    best_loss = np.min(L)
                    uneye_weights = self.net.state_dict()
                    
                if getting_worse>3:
                    epoch = self.max_iter+1 # stop the training if the loss is increasing for the validation set
                    self.net.load_state_dict(uneye_weights) 
                    print('early stop')  
                    
                epoch += 1
                
            # Evaluate on test set
            self.net.eval()
            out_test = self.net(Vtest,key)[0]

            if self.classes==2:
                Prediction = binary_prediction(out_test[:,1,:].data.cpu().numpy(),
                                               self.sampfreq,
                                               min_sacc_dist=self.min_sacc_dist,
                                               min_sacc_dur=int(self.min_sacc_dur/(1000/self.sampfreq)))
            else:
                Prediction = np.argmax(out_test.data.cpu().numpy(),1) # predict class that maximizes the softmax
            for c in range(classes):
                pred = Prediction==c
                kappa = cohenskappa(Ltest[:,c,:].flatten(),pred.astype(float).flatten())
                Kappa[i,c] = kappa
                print(kappa)

            # f1 value for saccades
            true_pos,false_pos,false_neg,on_distance,off_distance = accuracy((Prediction==1).astype(float),Ltest[:,1,:])
            f1 = (2 * true_pos)/(2 * true_pos + false_neg + false_pos)
            print('F1:',np.round(f1,3))
            F1[i] = f1
            On.append(on_distance)
            Off.append(off_distance)

            # FREE UP GPU
            del optimizer,criterion,Vtrain,Ltrain,Vtest,out,out_val,out_test,loss
        
            # save weights of last validation
            uneye_weights = self.net.state_dict()
            out_folder = './crossvalidation/'+self.weights_name
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            
            # weights to cpu
            if self.use_gpu:
                Keys = list(uneye_weights.keys())
                for k,key in enumerate(Keys):
                    uneye_weights[key] = uneye_weights[key].cpu()
                    
            torch.save(uneye_weights,os.path.join(out_folder,'crossvalidation_'+str(i)))
        
        print("Weights saved to",self.weights_name)
        return 
    
    
    

