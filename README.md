# artGAN
Keras-based implementation of a Deep Convolutional Generative Adversarial Network, based on code from [Robbie Barrat](https://github.com/robbiebarrat/art-DCGAN), [Soumith Chintala](https://github.com/soumith/dcgan.torch), and [Felix Mohr](https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/DCGAN-MNIST.ipynb). 

## Getting Started

Each python script can be run from the command line. Python 3 is recommended. 

### Dependencies


```
tensorflow
keras
numpy
matplotlib
scipy
```

### Installing

Use install_dependencies.sh to prepare.

```
./install_dependencies.sh
```

### Usage

```
python3 kerasGANv8.py
```
* kerasGANv8.py has a hard-coded variable for the directory to search for training images. The variable is named newdir, and defined on line 38. Change this variable to the location of your training images.

* Change the options for x_dim and y_dim to match your training images.

* Adjust the batch size and noise size to the speed at which you want to your model to learn. This is very dependent on how powerful your computer is. 

* These files are intended for CPU training only.

### Future Work

* add CLI argument parser (for image directories and paramter options)
* adapt for GPU calculations

## Authors

* **Lucas Lyon** - [lalyon](https://github.com/lalyon)


## Acknowledgments
Shoutout to the following people, whose code was invaluable while developing these neural networks.

* [Robbie Barrat](https://github.com/robbiebarrat/art-DCGAN)
* [Soumith Chintala](https://github.com/soumith/dcgan.torch)
* [Felix Mohr](https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/DCGAN-MNIST.ipynb). 



