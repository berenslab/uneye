import numpy as np
import sys, getopt
from scipy import io
import os
import datetime
from scipy.signal import butter, lfilter, freqz, resample
import numpy as np
from scipy.signal import butter, lfilter, freqz
from skimage.measure import label



def saccade_model(t, eta, c, amplitude):
    # The saccade generation is addapted from:
    # W. Dai, I. Selesnick, J.-R. Rizzo, J. Rucker and T. Hudson.
    # 'A parametric model for saccadic eye movement.'
    # IEEE Signal Processing in Medicine and Biology Symposium (SPMB), December 2016.
    # DOI: 10.1109/SPMB.2016.7846860.

    tau = amplitude/eta;       # tau: amplitude parameter (amplitude = eta*tau)

    # Default off-set values
    t0 = -tau/2;           # t0: saccade onset time

    s0 = 0;                # s0: initial saccade angle
    
    f = lambda t:  np.multiply(t,t>=0) + np.multiply(0.25*np.exp(-2*t),t>=0) + np.multiply(0.25*np.exp(2*t),t<0)
    f_vel = lambda t:  (t>=0) - np.multiply(0.5*np.exp(-2*t),t>=0) + 0.5*np.multiply(np.exp(2*t),t<0)
    fpvel = lambda A: eta*(1 - np.exp(-A/c))

    waveform = c*f(eta*(t - t0)/c) - c*f(eta*(t - t0 - tau)/c) + s0;
    velocity = eta*f_vel(eta*(t - t0)/c) - eta*f_vel(eta*(t - t0 - tau)/c);
    peak_velocity = fpvel(amplitude);
    waveform=waveform[0]
    
    return waveform#, velocity, peak_velocity, t



def saccade_amplitude_distribution(min_amp=.03,max_amp=90):
    """
    min_amp is the smallest saccade that can be generated in degree
    max_amp is the biggest saccade that can be generated in degree
    """
    amp=max_amp+1;
    m = 20;
    v = 1000000;
    mu = np.log((m**2)/np.sqrt(v+m**2));
    sigma = np.sqrt(np.log(v/(m**2)+1));
    while amp>max_amp or amp<min_amp:
        amp = np.random.lognormal(mu,sigma,[1,1]);
    return amp



def noise_amplitude_distribution(trial_nb,level):
    m = level;
    v = 1;
    mu = np.log((m**2)/np.sqrt(v+m**2));
    sigma = np.sqrt(np.log(v/(m**2)+1));
    amp = np.random.lognormal(mu,sigma,[trial_nb,1])
    return amp

def generate_eye_traces(N,min_amp=.5,max_amp=60):
    """ Generate waveforms saccades
    Input:
     -N is the number of trace simulated
     -min_amp is the minimum amplitude of saccades generated in degree
     -max_amp is the maximum amplitude of saccades generated in degree
    """
    
    # The saccade generation is addapted from:
    # W. Dai, I. Selesnick, J.-R. Rizzo, J. Rucker and T. Hudson.
    # 'A parametric model for saccadic eye movement.'
    # IEEE Signal Processing in Medicine and Biology Symposium (SPMB), December 2016.
    # DOI: 10.1109/SPMB.2016.7846860.
    # The saccade model corresponds to the 'main sequence' formula
    #    Vp = eta*(1 - np.exp(-A/c))
    # where Vp is the peak saccadic velocity and A is the saccadic amplitude.
    #
    # In this simulation, the main sequence parameters 'eta' and 'c'
    # and the amplitude of each saccade are chosen randomly in a given range.
    # The intervals between the saccades are also random.
    
    eta_min = 300; # 500                     # main sequence parameter (degree/sec)
    eta_max = 900; # 590
    
    c_min = 4.5;   # 4.5                     # main sequence parameter (no units)
    c_max = 7.5;   # 7.5
    
    # the bigger the variability in main sequence parameter the more general
    # the network will be after training (hopefully)
    ####
 
    Fs = 1000                           # sampling rate (samples/sec)
    t = np.arange(-0.1,0.1,1/Fs)                   # time axis (sec)
    tmax = 1*Fs #duration in msec
    min_velocity=3;        # any time where the simulated waveform saccade is 
                           # faster than "min_velocity" is going to be
                           # included. (in deg/sec)
                           
    amp_proportion=.3;     # for small microsaccades that are slower than "min_velocity"
                           # some proportion of the saccade is going to be
                           # included (between 0 and .5)
    blink_proportion=.01; #proportion of blink relative to saccades. Set it high to train the network.
    
    eye_trace_x = np.zeros((N,tmax));
    eye_trace_y = np.zeros((N,tmax));
    ground_truth = np.zeros((N,tmax),dtype=bool);
    blinks = np.zeros((N,tmax),dtype=bool);
    n=0;
    for i in np.arange(0,N,1):
        next_possible_start=0;
        for j in np.arange(0,tmax,1):
            if (np.random.rand(1)<1/333) & (next_possible_start<1):
                if np.random.rand(1)>blink_proportion: # generate a saccade
                    eta = eta_min + (eta_max - eta_min)*np.random.rand(1)   # (degree/sec)
                    c = c_min + (c_max - c_min)*np.random.rand(1)           # (no units)
                    Amp = saccade_amplitude_distribution(min_amp,max_amp)      # amplitude (degree)
                    Vp = eta * (1 - np.exp(-Amp/c));         # peak velocity (degree/sec)
                    waveform = saccade_model(t, eta, c, Amp)
                    saccade_time=np.concatenate(([False], abs(np.diff(waveform))>min_velocity/Fs))|((waveform>Amp[0]*amp_proportion) & (waveform<Amp[0]*(1-amp_proportion)));
                    direction = 2*np.pi*np.random.rand(1)
                    waveform_x=np.cos(direction)*waveform[saccade_time]-np.cos(direction)*waveform[np.argwhere(saccade_time)[0]]
                    waveform_y=np.sin(direction)*waveform[saccade_time]-np.sin(direction)*waveform[np.argwhere(saccade_time)[0]]
                    eye_trace_x[i,j:min(j+len(waveform_x),999)]=eye_trace_x[i,j]+waveform_x[0:min(len(waveform_x),999-j)]
                    eye_trace_y[i,j:min(j+len(waveform_x),999)]=eye_trace_y[i,j]+waveform_y[0:min(len(waveform_x),999-j)]
                    ground_truth[i,np.arange(j,min(j+len(waveform_x)-1,999))]=True
                    eye_trace_x[i,min(j+len(waveform_x),999):tmax]=eye_trace_x[i,min(j+len(waveform_x),999)-1]
                    eye_trace_y[i,min(j+len(waveform_x),999):tmax]=eye_trace_y[i,min(j+len(waveform_x),999)-1]
                    next_possible_start=len(waveform_y)
                    n+=1
                else: # generate a blink
                    length_blink=np.random.randint(300)+1
                    blink_x=np.ones((length_blink))*-np.random.randint(5)-10
                    blink_y=np.ones((length_blink))*-np.random.randint(5)-10
                    eye_trace_x[i,j:min(j+len(blink_x)-1,999)]=eye_trace_x[i,j]+blink_x[1:min(len(blink_x),999-j+1)]
                    eye_trace_y[i,j:min(j+len(blink_x)-1,999)]=eye_trace_y[i,j]+blink_y[1:min(len(blink_x),999-j+1)]
                    blinks[i,j:min(j+len(blink_x)-1,999)]=True
                    if j>1:
                        eye_trace_x[i,min(j+len(blink_x)-1,tmax):tmax]=eye_trace_x[i,j-1]
                        eye_trace_y[i,min(j+len(blink_x)-1,tmax):tmax]=eye_trace_y[i,j-1]
                    else:
                        eye_trace_x[i,min(j+len(blink_x)-1,tmax):tmax]=0
                        eye_trace_y[i,min(j+len(blink_x)-1,tmax):tmax]=0

                    next_possible_start=len(blink_x)

            next_possible_start-=1

    return  eye_trace_x,eye_trace_y,ground_truth


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def add_noise(X,Y,sd):
    """input: sd is the standard deviation of the white noise in a range [min,max] or just one value
    """
    if np.shape(sd)==():
        sd=[sd,sd];
        
    t = X.shape[1]
    N = X.shape[0]
    
    trial_noise_level = np.tile(np.random.rand(N,1) * (sd[1]-sd[0]) + sd[0], (1,t));
    noise4X = np.multiply(np.random.randn(N,t),trial_noise_level)
    noise4Y = np.multiply(np.random.randn(N,t),trial_noise_level)
    
    return X+noise4X , Y+noise4Y


def add_drift(X,Y,sd):
    """input: sd is the standard deviation of the white noise in a range [min,max] or just one value
              the white noise is then low passed at .5 Hz to generate drifts 
    """
    if np.shape(sd)==():
        sd=[sd,sd];
    t = X.shape[1]
    N = X.shape[0]
    
    trial_noise_level = np.tile(np.random.rand(N,1)*(sd[1]-sd[0])+sd[0],(1,t*2))
    
    # add drift to X signal
    drift4X = np.multiply(np.random.randn(N,t*2),trial_noise_level)
    drift4X = butter_lowpass_filter(drift4X,.5,t)
    drift4X = drift4X[:,int(t-t/2):int(t*2-t/2)]
    
    # add drift to Y signal
    drift4Y = np.multiply(np.random.randn(N,t*2),trial_noise_level)
    drift4Y = butter_lowpass_filter(drift4Y,.5,t)
    drift4Y = drift4Y[:,int(t-t/2):int(t*2-t/2)]
    
    return X+drift4X , Y+drift4Y

def add_smooth_pursuit(X,Y,degree_per_sec):
    """input: degree_per_sec is the the range [min,max] or just one value
              of a constant value 
    """
    if np.shape(degree_per_sec)==():
        degree_per_sec=[degree_per_sec,degree_per_sec];
    trial_ramp_speedX=np.zeros((X.shape[0],X.shape[1]))
    trial_ramp_speedY=np.zeros((Y.shape[0],Y.shape[1]))
    for i in np.arange(0,X.shape[0]):
        trial_ramp_speedX[i,:]=np.linspace(0,np.random.rand()*(degree_per_sec[1]-degree_per_sec[0])+degree_per_sec[0],1000)*np.sign(np.random.rand()-.5)
        trial_ramp_speedY[i,:]=np.linspace(0,np.random.rand()*(degree_per_sec[1]-degree_per_sec[0])+degree_per_sec[0],1000)*np.sign(np.random.rand()-.5)
        
    return X+trial_ramp_speedX,Y+trial_ramp_speedY

def add_fixation_noise_from_real_data(X,Y,pow_filepath):
    """this function add noise to artificial traces. The noise resembles that of real data that is used as an input.
    pow_filepath is a .mat file containing X_fft and Y_fft size nb_traces*1000
    the absolute value of the fft of 1 s fixations traces without saccades or blinks
    """
    t = X.shape[1]
    n = X.shape[0]
    power = io.loadmat(pow_filepath)
    example_nb = power['X_fft'].shape[0];
    xpower = power['X_fft'][:,:t]
    ypower = power['Y_fft'][:,:t]
    X2add = np.zeros((n,t))
    Y2add = np.zeros((n,t))
    for i in range(0,n):
        created_signalX = abs(np.fft.ifft(np.multiply(xpower[np.mod(i,example_nb),:],np.exp(1j*np.random.rand(1,t)*2*np.pi))))[0]
        created_signalX = created_signalX-np.mean(created_signalX)
        X2add[i,:] = created_signalX
        created_signalY = abs(np.fft.ifft(np.multiply(ypower[np.mod(i,example_nb),:],np.exp(1j*np.random.rand(1,t)*2*np.pi))))[0]
        created_signalY = created_signalY-np.mean(created_signalY)
        Y2add[i,:] = created_signalY
         
    return X+X2add,Y+Y2add

  
def downsample(X,Y,f_target):
    '''
    downsampling data recorded at sampling frequency of 1000 Hz to lower frequency f_target
    '''
    Xresamp = resample(X,f_target,axis=1)
    Yresamp = resample(Y,f_target,axis=1)
    
    return Xresamp,Yresamp


def upsample(X,Y):
    '''
    interpolating data recorded at lower sampling frequency to target frequency of 1000 Hz
    '''
    Xresamp = resample(X,1000,axis=1)
    Yresamp = resample(Y,1000,axis=1)
    
    return Xresamp,Yresamp



def traces(X,Y,smooth=False):
    '''
    Preprocessing of eye traces.
    Computes velocity vector for X and Y axis and stores positive and negative movement in separate channels.
    Splits whole recording into vectors of size = trial_size
    ^
    Params
    ------
    X: x axis position of eye, numpy array
    Y: y axis position of eye, numpy array
    
    Output
    ------
    T: trace matrix, columns: pos. X movement , neg. X movement , pos. Y movement , neg. Y movement
       dimensions: batch x 4 x trial_size
       
    '''
    # assuming rows contain trials, columns contain time
    batchsize = X.shape[0]
    if smooth==False:
        # X position
        X_diff = np.diff(X,axis=-1)
        X_diff = np.concatenate((np.zeros((batchsize,1)),X_diff),1)
        X_diff[np.isnan(X_diff)] = 0
        
        # Y position
        Y_diff = np.diff(Y,axis=-1)
        Y_diff[np.isnan(Y_diff)] = 0
        Y_diff = np.concatenate((np.zeros((batchsize,1)),Y_diff),1)
    else:
        # apply smooth diff using savitzky golay filter
        X_diff = savgol_filter(X, window_length=21, polyorder=1, deriv=1, delta=1.0) 
        Y_diff = savgol_filter(X, window_length=21, polyorder=1, deriv=1, delta=1.0) 
        

    X_diff[X_diff>1.5] = 1.5 
    X_diff[X_diff<-1.5] = -1.5 
    Y_diff[Y_diff>1.5] = 1.5 
    Y_diff[Y_diff<-1.5] = -1.5 
    
    #T = np.tile((X_pos,X_neg,Y_pos,Y_neg),1)
    T = np.tile((X_diff,Y_diff),1)
    T = np.swapaxes(np.swapaxes(T,0,1),1,2) # swap axes so that first dimension:trials, second: time, third: x+y

  
    return T



def labels(sacc):
    
    
    S = sacc.astype(float)
    
    return S


def predict(Output,p=0.5,smooth=True,min_sacc_dur=10): 

    # thresholding probability output of model:
    S_predic = (Output>p).astype(int) 
    
    if smooth==True:
        if len(Output.shape)<2:
            l = label(S_predic)
            for j in range(1,np.max(l)+1):
                s = np.sum(l==j)
                if s<min_sacc_dur:
                    S_predic[l==j] = 0
        else:
            # assume: first dimension: number of samples
            for n in range(Output.shape[0]):
                l = label(S_predic[n,:])
                for j in range(1,np.max(l)+1):
                    s = np.sum(l==j)
                    if s<min_sacc_dur:
                        S_predic[n,l==j] = 0
                            
    return S_predic     


def accuracy(predicted,truth):
    '''
    computes accuracy of prediction compared to groundtruth
    
    predicted,truth: numpy arrays, batchsize x time
    '''
    truth[truth<1] = 0
    truth = truth.astype(int)
    
    batchsize = predicted.shape[0]
    
    Diff_p = np.diff(predicted,axis=1)
    Diff_t = np.diff(truth,axis=1)
    
    # count variables:
    true_pos = 0
    false_neg = 0
    false_pos = 0
    on_distance = []
    off_distance = []
    
    # loop trough trials:
    for i in range(batchsize):
        trace_t = Diff_t[i,:]
        trace_p = Diff_p[i,:]

        
        start_p = np.argwhere(trace_p==1)
        end_p = np.argwhere(trace_p==-1)
        start_t = np.argwhere(trace_t==1)
        end_t = np.argwhere(trace_t==-1)
        
        
        # exclude border saccades:
        if (truth[i,0]==1):
            end_t = end_t[1:]
        #if (predicted[i,0]==1):
            #end_p = end_p[1:]
        if (truth[i,-1]==1):
            start_t = start_t[:-1]
        #if (predicted[i,-1]==1):
            #start_p = start_p[:-1] 
            
        if len(start_t)!=0: # if there at least one saccade
            
            # if all conditions met, loop through each true saccade
            for j in range(len(start_t)):
                
                content_of_pred = predicted[i,int(start_t[j]):int(end_t[j])] #content of prediction during true saccade duration 
                
                #if np.isnan(np.mean(content_of_pred))==False:
                    #true_sacc += 1
                if np.mean(content_of_pred)==0:
                    # checks if no saccade has been detected by network
                    # add false negative detection
                    false_neg += 1
                else:
                    true_pos += 1
                    # compare on- and offset of saccades
                    # find correct onset
                    # test whether onset of predicted saccade is before or after true onset:
                    pred_at_true_onset = predicted[i,int(start_t[j])]

                    if pred_at_true_onset==0:
                        try:
                            ind = np.argwhere(start_p >= start_t[j])[:,0][0]
                        except:
                            print(np.argwhere(start_p >= start_t[j]))
                            plt.plot(predicted[i,:])
                            plt.plot(truth[i,:])
                            plt.show()

                    elif pred_at_true_onset==1:

                        ind = np.argwhere(start_p <= start_t[j])[:,0]

                        # it can happen that the start of the predicted saccade lies outside of trace. 
                        # In this case, set ind to None and start index to 0
                        if len(ind)!=0:
                            ind = ind[-1]
                        else:
                            ind = None

                    if ind is not None:

                        start = start_p[ind] 
                        # find next end:
                        ends = np.argwhere(end_p>start)[:,0]
                        if len(ends)==0:
                            end = predicted.shape[1]
                        else:
                            end = end_p[ends[0]]
                        
                    else:
                        start = 0
                        # as in this case the predicted saccade is a border saccade, use the first end index of the predicted saccade:
                        end = np.argwhere(trace_p==-1)
                        if len(end)==0:
                            end = predicted.shape[1]
                        else:
                            end = end[0]

                    on_distance.append(int(start)-int(start_t[j]))
                    off_distance.append(int(end)-int(end_t[j]))

        # remove border sacades from predicted trace
        if (predicted[i,0]==1):
            end_p = end_p[1:]
        if (predicted[i,-1]==1):
            start_p = start_p[:-1] 
            
        # count false positives 
        if len(start_p)!=0:
            for j in range(len(start_p)):
                content_of_true = truth[i,int(start_p[j]):int(end_p[j])] #content of groundtruth during predicted saccade duration
                if np.mean(content_of_true)==0:
                    # checks if no true saccade has been detected
                    # add false positive detection
                    false_pos += 1

    
    return true_pos,false_pos,false_neg,on_distance,off_distance