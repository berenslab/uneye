import uneye
import numpy as np
from scipy import io
import sys, getopt, ast
import os

###############################
########### System: ###########
###############################
   

def main(argv):
    
    try:
        opts, args = getopt.getopt(argv, "m:x:y:l:c:w:f:",
                               ["mode=","x=","y=","labels=","classes=","weights=","sampfreq="])
    except getopt.GetoptError:
        print('UnEye.py -x <xfile> -y <yfile>')
        sys.exit(2)
    
    # default values:
    mode = 'predict' #default mode
    weights = 'weights'
    classes = 2
    
    # Arguments
    for opt,arg in opts:
        if opt in ('-m','--mode'):
            mode = arg
        elif opt == '-x':
            X = arg
        elif opt == '-y':
            Y = arg
        elif opt in ('-l','--labels'):
            L = arg
        elif opt in ('-c','--classes'):
            classes = int(arg)
        elif opt in ('-w','--weights'):
            weights = arg   
        elif opt in ('-f','--sampfreq'):
            sampfreq = int(arg)   


    if mode=='hello':
        print('You successfully installed uneye.')
        sys.exit(2)
        
    ##### Training mode #####
    elif mode=='train':
        print('########### TRAINING ###########')
        # check input arguments
        try:
            print('X: ', X)
            print('Y: ', Y)
            print('Labels: ', L)
            print('sampling frequency: ',sampfreq)
        except:
            print('At least one input argument is missing: UnEye.py -x <xfile> -y <yfile> -l <saccfile> -f <samplingfreq>')
            sys.exit(2)
        print('weights file: ',weights)

        # load data
        # CSV
        if X.endswith('csv'):
            X,Y,Labels = np.loadtxt(X,delimiter=","),np.loadtxt(Y,delimiter=","),np.loadtxt(L,delimiter=",")
        # MATLAB 
        elif X.endswith('mat'):
            X,Y,Labels = io.loadmat(X)['X'],io.loadmat(Y)['Y'],io.loadmat(L)['Sacc']

        # check if data has right dimensions (2)
        xdim,ydim,ldim = X.ndim,Y.ndim,Labels.ndim
        if any((xdim!=2,ydim!=2,ldim!=2)):
            # reshape into matrix with trials of length=1sec
            trial_len = int(1000 * sampfreq/1000)
            time_points = len(X)
            n_trials = int(time_points/trial_len)
            X = np.reshape(X[:n_trials*trial_len],(n_trials,trial_len))
            Y = np.reshape(Y[:n_trials*trial_len],(n_trials,trial_len))
            Labels = np.reshape(Labels[:n_trials*trial_len],(n_trials,trial_len))
            
        # split data into training and test set (90% vs 10%)
        randind = np.random.permutation(X.shape[0])
        ntrain = int(X.shape[0]*0.9)
        ntest = X.shape[0] - ntrain
        Xtrain,Ytrain,Ltrain = X[randind[:ntrain],:],Y[randind[:ntrain],:],Labels[randind[:ntrain],:]
        Xtest,Ytest,Ltest = X[randind[ntrain:],:],Y[randind[ntrain:],:],Labels[randind[ntrain:],:]
        
        print('Starting training. Please wait.')
        model = uneye.DNN(weights_name=weights,sampfreq=sampfreq,
                        classes=classes)

        model.train(Xtrain,Ytrain,Ltrain)
        
        print('########### TEST ###########')
        _,_,perf = model.test(Xtrain,Ytrain,Ltrain)
        # print performance 
        kappa = perf['kappa']
        f1 = perf['f1']
        on = np.array(perf['on'])*1000.0/float(sampfreq)
        off = np.array(perf['off'])*1000.0/float(sampfreq)
        print(ntest,'samples used for testing.')
        print("Performance (Cohen's kappa) on test set:",kappa)
        print("Performance (F1) on test set:",f1)
        print("Performance (onset difference) on test set:",np.mean(np.abs(on)))
        print("Performance (offset difference) on test set:",np.mean(np.abs(off)))
        if kappa<0.7:
            print("Bad performance can be due to an insufficient size of the training set, high noise in the data or incorrect labels. Check your data and contact us for support.")

        return kappa,f1,on,off #return performance measures    
    
    ##### Prediction mode #####
    elif mode=='predict':
        
        try:
            print('X: ', X)
            print('Y: ', Y)
            print('sampling frequency: ',sampfreq)
        except:
            print('Missing input arguments: UnEye.py -x <xfile> -y <yfile>')
            sys.exit(2)
        print('weights file: ',weights)
        
        
        # load data
        # CSV
        if X.endswith('csv'):
            Xarr,Yarr = np.loadtxt(X,delimiter=","),np.loadtxt(Y,delimiter=",")
        # MATLAB 
        elif X.endswith('mat'):
            Xarr,Yarr = io.loadmat(X)['X'],io.loadmat(Y)['Y']
                
        model = uneye.DNN(weights_name=weights,sampfreq=sampfreq,classes=classes)        
        Prediction,Probability = model.predict(Xarr,Yarr)
        
        # save output
        if type(X)==str:
            parent = os.path.dirname(X)
            if X.endswith('mat'): 
                io.savemat(parent+'/Labels_pred',{'Sacc':Prediction})
                io.savemat(parent+'/Labels_prob',{'Prob':Probability})
            elif X.endswith('csv'):
                np.savetxt(parent+"/Labels_pred.csv", Prediction, delimiter=",")
                np.savetxt(parent+"/Labels_prob.csv", Probability, delimiter=",")
            print('Saccade prediction saved to '+parent)
                   
    else:
        print('Unknown mode. Use train or predict')
        sys.exit(2)


if __name__ == '__main__':
    '''
    callable from command line
    Can open data of the following file format:
    - .mat
    
    '''
    print(sys.argv)
    main(sys.argv[1:])
