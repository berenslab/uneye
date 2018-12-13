

![alt text](https://raw.githubusercontent.com/berenslab/uneye/master/logo.jpeg?token=AcbomYi_PxlSK_8ua5zR3m60F5DL5UQJks5bPkWrwA%3D%3D)

# U'n'Eye: Deep neural network for the detection of saccades and other eye movements
Bellet et al. 2018, **Human-level saccade and microsaccade detection with deep neural networks**
********
## Latest Updates:
- [web service](http://uneye.berenslab.org) available
- kernel size of convolution and max pooling operations now definable by the user (thus longer or shorter time windows will be seen by U'n'Eye)

------------------

### <i>Train your own eye movement detection network in 10 minutes and label your data. [Get started now](#installation)</i>
### <i>Try out our [web service](http://uneye.berenslab.org).

## <a name="content">Content</a> 
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
	- [jupyter notebook](#jupyter)
	- [command line](#cmd)
- [Module description](#module)
- [Further use](#further)

## <a name="overview">Overview</a> 
U'n'Eye is a Python 3 package, that uses [PyTorch](http://pytorch.org) for the neural network implementation.

For a description of the algorithm, see [our preprint](https://www.biorxiv.org/content/early/2018/06/29/359018).
For any questions regarding this repository please contact [marie-estelle.bellet@student.uni-tuebingen.de](mailto:marie-estelle.bellet@student.uni-tuebingen.de) or [philipp.berens@uni-tuebingen.de](philipp.berens@uni-tuebingen.de).

We provide network weights that were learned on different datasets, described in [the paper](https://www.biorxiv.org/content/early/2018/06/29/359018). The weights can be found in the folder **training** and the corresponding datasets will be available in the folder **data** after publication. For instructions on how to use pretrained networks, please see below.

**Users can train their own network to obtain optimal performance.** Please see the module description below and the example jupyter notebook **UnEye.ipynb** for instructions.

We provide a [docker](http://docker.com) container for platform-independent use that is recommended for Windows users.


[back to start](#content)

## <a name="installation">Installation:</a> [local](#local) or with [docker](#docker)

In the following, all commands written in boxes need to be entered into your terminal. 
 
### <a name="local">Local:</a> install the python package (Mac & Linux)

**1)** Via the terminal, check if you have python3 installed

	python3 --version
If not found, download and install python3 [here](https://www.python.org/downloads/release/python-364/). 

**2)** Clone the GitHub repository into your local directory and make sure you are using pip3.

	git clone https://github.com/berenslab/uneye
	alias pip=pip3
**3)** Now the last step: install the package. Use one of the following commands, depending on your platform:

For Mac:

	pip install ./ -r ./requirements_mac.txt
	
For Linux:

	pip install ./ -r ./requirements_lin.txt
	
Note: If the git command does not work under Mac OS, first run

	xcode-select --install


### <a name="docker">Docker:</a> the platform independent solution

**1)** Install docker:

for [Windows](https://docs.docker.com/docker-for-windows/install/#download-docker-for-windows), [Mac](https://store.docker.com/editions/community/docker-ce-desktop-mac) or [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository)

**2)** Download the docker image that contains U'n'Eye, pytorch and all other python packages you need. This step will take some time.

	docker pull mebellet/uneye:v-0.2	


[back to start](#content)

## <a name="usage">Usage</a> 
General:

	model = uneye.DNN(sampfreq= sampling_frequency )
	model.train(X,Y,Labels)
	model.test(X,Y,Labels)
	model.predict(X,Y)
	model.crossvalidate
	
Generally, one first calls the network and can then apply different methods, as described in the module description below.

### <a name="jupyter">Jupyter Notebook</a> 

An example jupyter notebook is provided in this repository (**UnEye.ipynb**).
Clone the repository as described above (or alternatively download the files). Put the file **UnEye.ipynb** into the folder where you want to work. Depending on whether you use U'n'Eye with the docker container or locally, enter the following into your command window to use the jupyter notebook (replace YourWorkingDirectory)

#### A) Docker


    cd /YourWorkingDirectory
    docker run -it --rm -p 6688:8888 --name uneye -v $(pwd)/.:/home/jovyan mebellet/uneye:v-0.2

<b>!! Important:</b> Now, don't just copy the output of the into your browser, but do the following:

Open your web browser. Enter "localhost:6688" as URL. Then you will be asked for a token. This token appeared when you entered the command above. It looks something like this: http://6688:8888/?token=775c758f58cdc82bf6ddf51a112228f4dd4229c5b3847bb1 . Copy the token (in this case 775c758f58cdc82bf6ddf51a112228f4dd4229c5b3847bb1) and log in. Now you should see the content of your folder in the web browser.

#### B) Local

	alias python=python3
	cd /YourWorkingDirectory
	jupyter notebook


To stop jupyter notebook, press **Ctrl + C** .


[back to start](#content)

### <a name="cmd">Command line</a> 
With the .py file **UnEye.py** you can use the package from the command line. So far it takes the following input arguments:

***

Input arguments (*=necessary):
**mode***: train or predict

**x***: filename of the horizontal eye position (.csv or .mat file)

**y***: filename of the vertical eye position (.csv or .mat file)

**labels**(*for training): filename of the eye movement ground truth labels (.csv or .mat file)

**sampfreq***: sampling frequency of the data (Hz)

**weights**: ouput/input filename of trained weights

***

first run the following, depending on whether you use the Docker container or work locally. Note: /YourWorkingDirectory **must contain the .py file UnEye.py** from this repo.
#### A) Docker

	cd /YourWorkingDirectory
	docker run -it -p 6688:8888 --name uneye -v $(PWD)/:/home/jovyan mebellet/uneye:v-0.2 /bin/bash
#### B) Local

	cd /YourWorkingDirectory
	alias python=python3



Now, stay in the command window. You can from here either **train** a new network or **predict** eye movements from new data:

#### Training

	python UnEye.py -m train -x data/x_name -y data/y_name -l data/labels_name -f sampfreq
Note: In this example the files are located in the directory _/YourWorkingDirectory/data_

The trained weights will be saves to _training/weights_ or to _training/weightsname_ if the argument _-w weightsname_ is given.


#### Prediction

	python UnEye.py -m predict -x data/x_name -y data/y_name -f sampfreq

Note: This will automatically use the weights saved under _training/weights_ unless you specify your weightsname by giving the input argument _-w training/weightsname_ .

The predicted saccade probability and the binary prediction are saved to _data/Sacc_prob_ and _data/Sacc_pred_ respectively.

***
If you use docker, exit after usage with:

	exit


[back to start](#content)

## <a name="module">Module description</a> 

The uneye module contains the **DNN** class 

	model = uneye.DNN(max_iter=500, sampfreq=1000,
                 lr=0.001, weights_name='weights',
                min_sacc_dist=1,min_sacc_dur=6,augmentation=True,
                 ks=5,mp=5,inf_correction=1.5,val_samples=30)
                
   
   -----
   
 Arguments:
	
max_iter: maximum number of epochs during training

sampfreq: sampling frequency of the eye tracker (Hz)

lr: learning rate of the network training 

weights_name: input/output filename for trained network weights

min*_*sacc_dist: minimum distance between two saccades in ms for merging of saccades

min*_*sacc_dur: minimum saccade duration in ms for removal of small events

augmentation: whether or not to use data augmentation for training, default: True

ks: kernel size of convolution operations, default=5

mp: size of max pooling operation, default=5

inf_correction: float, value to replace Infs occuring after differential of input signal

val_samples: int, number of validation samples (for early stopping criterion)
    


*** 
	
### Methods

### train: 

Train the network weights with your own training data. It is recommended to label at least 500 seconds of data. Arrange the data into a matrix of _samples*timebins_ or input them as a vector of length _timebins_ . In case you arrange the data into a matrix, the time length of each sample should be at least 1000 ms.

During training, the current iteration and maximum number of iterations will be shown. Note that training stops earlier most of the times.

	model.train(X,Y,Labels)

X,Y : horizontal and vertical eye positions, array-like, shape: {(nsamples, nbins) or (nbins)}

Labels: eye movement ground truth labels with k different values for k different classes (e.g. fixation=0, saccade=1, saccades should always be labelled with 1), array-like, shape: {(nsamples, nbins) or (nbins)}



Output: the trained weights are saved into the folder **training**. 

***

### test: 

Test performance of the network after training on your own test data. Arrange the data into a matrix of _samples*timebins_ or input them as a vector of length _timebins_ .

	Prediction, Probability, Performance = model.test(X,Y,Labels)

Input parameters:

X,Y : horizontal and vertical eye positions, {array-like, shape: {(nsamples, nbins) or (nbins)}

Labels: eye movement ground truth labels, array-like, shape: {(nsamples, nbins) or (nbins)}


Output:

Prediction: eye movement class prediction for each time bin, same shape as input

Probability: softmax probability output of network, shape: {n_samples,classes,time) or (classes,time)}

Performance: numpy dictionary containing different performance measures: **Cohen's Kappa**, **F1**, **onset distance**, **offset distance**. To call these measures, use

	cohens_kappa = Performance['kappa']
	f1 = Performance['f1']
	on_dist = Performance['on']
	off_dist = Performance['off']



***
### predict: 

Predict saccades in recordings with pretrained weights. Arrange the data into a matrix of _samples*timebins_ or input them as a vector of length _timebins_ .

	Prediction, Probability = model.predict(X,Y)

Input parameters:

X,Y : horizontal and vertical eye positions, {array-like, shape: {(nsamples, nbins) or (nbins)}


Output:

Prediction: eye movement class prediction for each time bin, same shape as input

Probability: softmax probability output of network, shape: {n_samples,classes,time) or (classes,time)}

***
### crossvalidate: 

Implementation of K-fold cross-validation.

	crossvalidate(self,X,Y,Labels,X_val,Y_val,Labels_val,Labels_test=None,K=10)

Input parameters:

X,Y : horizontal and vertical eye positions, {array-like, shape: {(nsamples, nbins) or (nbins)}

Labels: array-like, shape: {(n_timepoints),(n_samples, n_timepoints)}, class labels in range [0 classes-1], fixation=0, saccades=1
        
X_val,Y_val: array-like, shape: {(n_timepoints),(n_samples, n_timepoints)}, additional horizontal and vertical eye positions in degree for early stopping criterion. Can be small.
        
Labels_val: array-like, shape: {(n_timepoints),(n_samples, n_timepoints)}, additional class labels in range [0 classes-1], fixation=0, saccades=1 for early stopping criterion. Can be small.
        
Labels_test: array-like, shape: {(n_timepoints),(n_samples, n_timepoints)}, if test Labels different from training labels (for training with missing labels only), optional
        
K: float, number of folds of cross validation

[back to start](#content)

## <a name="further">Further use</a> 

**Adding other python modules to the docker container**

You can add other python modules to the docker container once you pulled the image (as described above). For this, run:

    docker run -it -p 6688:8888 --name uneye mebellet/uneye:v-0.2 /bin/bash
    pip install module_name
    
    
This installs the python module 'module_name'.
Exit the container by pressing Control+p and Control+q, then commit the changes to the image:

	docker commit uneye mebellet/uneye:v-0.2
	
[back to start](#content)