

![alt text](https://raw.githubusercontent.com/berenslab/uneye/master/logo.jpeg?token=AcbomYi_PxlSK_8ua5zR3m60F5DL5UQJks5bPkWrwA%3D%3D)

## Deep neural network for the detection of saccades and other eye movements
Bellet et al. 2018, **Human-level saccade and microsaccade detection with deep neural networks**
********

uneye is a Python 3 package, that uses [pytorch](http://pytorch.org) for the neural network implementation.

For a description of the algorithm, see (link to bioarxiv).
For any questions regarding this repository please contact [marie-estelle.bellet@student.uni-tuebingen.de](mailto:marie-estelle.bellet@student.uni-tuebingen.de) or [philipp.berens@uni-tuebingen.de](philipp.berens@uni-tuebingen.de).

We provide network weights that were learned on different datasets. The weights can be found in the folder **training** and the corresponding datasets will be available in the folder **data** after publication. 

Users can train their own network to obtain optimal performance. Please see the module description below and the example jupyter notebook **UnEye.ipynb** for instructions.

We provide a [docker](http://docker.com) container for platform-independent use. Under Mac OS and Ubuntu, you can alternatively install the package on your local computer. As the network is based on PyTorch, using it locally on Windows is not straightforward. We will add a description in the future. For now, please use the docker solution if you are working on Windows.



## Installation A): Docker
**for Mac OS / Ubuntu / Windows**
 
**1)** Pull repo into your local directory:

	git pull https://github.com/berenslab/uneye
	
**2)** Download and install Docker:

[Windows](https://docs.docker.com/docker-for-windows/install/#download-docker-for-windows) 
[Mac OS](https://docs.docker.com/docker-for-windows/install/#download-docker-for-windows) 
[Ubuntu](https://docs.docker.com/docker-for-windows/install/#download-docker-for-windows) 
 
**2)** Download the docker image that contains U'n'Eye, pytorch and all other python packages you need. This step will take some time because the docker image has a size of 1.2 GB.

    docker pull marieestelle/bellet_uneye:v-0.1



## Installation B): Local
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


## Module description

The uneye module contains the **DNN** class 

	model = uneye.DNN(max_iter=500, sampfreq=1000,
                 lr=0.001, weights_name='weights',
                classes=2,min_sacc_dist=1,
                 min_sacc_dur=1,threshold=0.5)
                
   
   -----
   
 Arguments:
	
max_iter: maximum number of epochs during training

sampfreq: sampling frequency of the eye tracker (Hz)

lr: learning rate of the network training 

weights_name: input/output filename for trained network weights

classes: number of target classes to predict

min*_*sacc_dist: minimum distance between two saccades in ms for merging of saccades

min*_*sacc_dur: minimum saccade duration in ms for removal of small events

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
***
## Usage: with Jupyter Notebook

An example jupyter notebook is provided in this repository (**UnEye.ipynb**).

Depending on whether you use U'n'Eye with the docker container or locally, do the following to use the jupyter notebook:

### Docker

    cd /YourWorkingDirectory
    docker run -it --rm -p 8888:8888 -v $(pwd)/.:/home/jovyan marieestelle/bellet_uneye

Copy the output " http://localhost:8888... " into your browser. Then you will see the working directory and the jupyter notebook **UnEye.ipynb**.

### Local


	alias python=python3
	cd /YourWorkingDirectory
	jupyter notebook


To stop jupyter notebook, press **Ctrl + C** .


## Usage: from command line
With the .py file **UnEye.py** you can use the package from the command line. 

***

Input arguments (*=necessary):

**x***: filename of the horizontal eye position (.csv or .mat file)

**y***: filename of the vertical eye position (.csv or .mat file)

**labels**(*for training): filename of the eye movement ground truth labels (.csv or .mat file)

**sampfreq***: sampling frequency of the data (Hz)

**classes**: number of target classes to predict, default: 2

**weightsname**: ouput/input filename of trained weights

***

first run the following, depending on whether you use the Docker container or work locally:
### Docker

	cd /YourWorkingDirectory
	docker run -it -p 8888:8888 -v $(PWD)/:/home/jovyan marieestelle/bellet_uneye:v-0.1 /bin/bash
### Local

	cd /YourWorkingDirectory
	alias python=python3
Note: /YourWorkingDirectory **must contain the .py file UnEye.py** from this repo.
***	

Now you can either **train** a new network or **predict** eye movements from new data:

**Training:** 

	python UnEye.py -m train -x data/x_name -y data/y_name -l data/labels_name -f sampfreq
Note: In this example the files are located in the directory _/YourWorkingDirectory/data_

The trained weights will be saves to _training/weights_ or to _training/weightsname_ if the argument _-w weightsname_ is given.


**Prediction:**

	python UnEye.py -m train -x data/x_name -y data/y_name -f sampfreq

Note: This will automatically use the weights saved under _training/weights_ unless you specify your weightsname by giving the input argument _-w training/weightsname_ .

The predicted saccade probability and the binary prediction are saved to _data/Sacc_prob_ and _data/Sacc_pred_ respectively.

***
If you use docker, exit after usage with:

	exit



    
## Further use

**Adding other python modules to the docker container**

You can add other python modules to the docker container once you pulled the image (as described above). For this, run:

    docker run -it -p 8888:8888 marieestelle/bellet_uneye /bin/bash
    pip install module_name
    
This installs the python module 'module_name'.

You are now in your docker environment and can execute jupyter notebook with the following command:

    jupyter notebook
    
Or run the package from the command line as previously described.


