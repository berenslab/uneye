

![alt text](https://raw.githubusercontent.com/berenslab/uneye/master/logo.jpeg?token=AcbomYi_PxlSK_8ua5zR3m60F5DL5UQJks5bPkWrwA%3D%3D)

# U'n'Eye: Deep neural network for the detection of saccades and other eye movements
Bellet et al. 2018, **Human-level saccade and microsaccade detection with deep neural networks**
********
## Latest Updates:
- network trained on three combined datasets (generalization to different types of data)
- kernel size of convolution and max pooling operations now definable by the user (thus longer or shorter time windows will be seen by U'n'Eye)

------------------
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

We provide a [docker](http://docker.com) container for platform-independent use. Under Mac OS and Ubuntu, you can alternatively install the package on your local computer. As the network is based on PyTorch, using it locally on Windows is not straightforward. We will add a description in the future. For now, please use the docker solution if you are working on Windows.

[back to start](#content)

## <a name="installation">Installation</a> 
### A): Docker
**for Mac OS / Ubuntu / Windows**
 
**1)** Pull repo into your local directory:

	git pull https://github.com/berenslab/uneye
	
**2)** Download and install Docker:

[Windows](https://docs.docker.com/docker-for-windows/install/#download-docker-for-windows) 
[Mac OS](https://docs.docker.com/docker-for-windows/install/#download-docker-for-windows) 
[Ubuntu](https://docs.docker.com/docker-for-windows/install/#download-docker-for-windows) 
 
**2)** Download the docker image that contains U'n'Eye, pytorch and all other python packages you need. This step will take some time because the docker image has a size of 1 GB.

    docker pull mebellet/uneye:v-0.2



### B): Local
**for Mac OS & Ubuntu**
	
**1)** Check if you have python3

	python3 --version
If not found, download and install python3 [here](https://www.python.org/downloads/release/python-364/). 

**2)** Pull repo into your local directory and install package:

	git pull https://github.com/berenslab/uneye
	alias pip=pip3
	pip install ./ -r ./requirements.txt
If the git pull command does not work under Mac OS, first run

	xcode-select --install

[back to start](#content)

## <a name="usage">Usage</a> 
Genral:

	model = uneye.DNN(sampfreq= sampling_frequency )
	model.train(X,Y,Labels)
	model.test(X,Y,Labels)
	model.predict(X,Y)
	
Generally, one first calls the network and can then apply different methods, as described in the module description below.

### <a name="jupyter">Jupyter Notebook</a> 

An example jupyter notebook is provided in this repository (**UnEye.ipynb**).

Depending on whether you use U'n'Eye with the docker container or locally, do the following to use the jupyter notebook:

#### A) Docker

    cd /YourWorkingDirectory
    docker run -it --rm -p 6688:8888 --name uneye -v $(pwd)/.:/home/jovyan mebellet/uneye:v-0.2

Copy the output " http://localhost:6688... " into your browser. Then you will see the working directory and the jupyter notebook **UnEye.ipynb**.

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

**x***: filename of the horizontal eye position (.csv or .mat file)

**y***: filename of the vertical eye position (.csv or .mat file)

**labels**(*for training): filename of the eye movement ground truth labels (.csv or .mat file)

**sampfreq***: sampling frequency of the data (Hz)

**classes**: number of target classes to predict, default: 2

**weightsname**: ouput/input filename of trained weights

***

first run the following, depending on whether you use the Docker container or work locally. Note: /YourWorkingDirectory **must contain the .py file UnEye.py** from this repo.
#### A) Docker

	cd /YourWorkingDirectory
	docker run -it -p 6688:8888 --name uneye -v $(PWD)/:/home/jovyan mebellet/uneye:v-0.2 /bin/bash
#### B) Local

	cd /YourWorkingDirectory
	alias python=python3



Now you can either **train** a new network or **predict** eye movements from new data:

#### Training

	python UnEye.py -m train -x data/x_name -y data/y_name -l data/labels_name -f sampfreq
Note: In this example the files are located in the directory _/YourWorkingDirectory/data_

The trained weights will be saves to _training/weights_ or to _training/weightsname_ if the argument _-w weightsname_ is given.


#### Prediction

	python UnEye.py -m train -x data/x_name -y data/y_name -f sampfreq

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
                classes=2,min_sacc_dist=1,
                 min_sacc_dur=6,augmentation=True,
                 ks=5,mp=5)
                
   
   -----
   
 Arguments:
	
max_iter: maximum number of epochs during training

sampfreq: sampling frequency of the eye tracker (Hz)

lr: learning rate of the network training 

weights_name: input/output filename for trained network weights

classes: number of target classes to predict

min*_*sacc_dist: minimum distance between two saccades in ms for merging of saccades

min*_*sacc_dur: minimum saccade duration in ms for removal of small events

augmentation: whether or not to use data augmentation for training, default: True

ks: kernel size of convolution operations, default=5

mp: size of max pooling operation, default=5

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