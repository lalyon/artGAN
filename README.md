# artGAN
Keras-based implementation of a Deep Convolutional Generative Adversarial Network, based on code from [Robbie Barrat](https://github.com/robbiebarrat/art-DCGAN), [Soumith Chintala](https://github.com/soumith/dcgan.torch), and [Felix Mohr](https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/DCGAN-MNIST.ipynb). 

## Getting Started

kerasGANv8.py supports command line arguments.

```
usage: kerasGANv8.py [-h] --batchSize BATCHSIZE [--noiseSize NOISESIZE]
                     [--yDim YDIM] [--xDim XDIM] [--outputDir OUTPUTDIR]
                     [--trainingDir TRAININGDIR]

optional arguments:
  -h, --help            show this help message and exit

required arguments:
  --batchSize BATCHSIZE
                        batch size
  --noiseSize NOISESIZE
                        size of noise input
  --yDim YDIM           input y dimension
  --xDim XDIM           input x dimension
  --outputDir OUTPUTDIR
                        where to save generated imgs
  --trainingDir TRAININGDIR
                        training imgs directory
```
If training on a CPU, I've found the following options productive:
```
python3 kerasGANv8.py --xDim=64 --yDim=64 --batchSize=4 --noiseSize=4 --outputDir=[your/desired/output/directory] --trainingDir=[directory/with/training/imgs]
```


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

* Change the options for x_dim, y_dim, trainingDir, and outputDir to match your training images and desired output location.

* Adjust the batch size and noise size to the speed at which you want to your model to learn. This is very dependent on how powerful your computer is. 

* These files are intended for CPU training only.

### Tips
* For images with dimensions 256x256, having ngf=100 and ndf=15 has been optimal.
* For images with dimensions 128x128 or 64x64, having ngf=160 and ndf=20 to 40 has been optimal.

### Examples of output images and their training periods
#### MNIST Training

![MNIST Digits Training](readmeImages/gifs/MNIST.gif)
#### MNIST Final Output - 2900 training epochs
![MNIST Final Output](readmeImages/stills/MNISTEpoch2900.png)

#### 128x128px Paintings Training
![gan128Paintings Training](readmeImages/gifs/gan128Paintings.gif)
#### 128x128px Paintings Final Output - 9300 training epochs
![gan128Paintings Output](readmeImages/stills/gan128PaintingsEpoch9300.png)

#### 256x256px Impressionist Training
![gan256Impres Training](readmeImages/gifs/gan256Impres.gif)
#### 256x256px Impressionist Final Output - 14740 training epochs
![gan256Impres Final Output](readmeImages/stills/gan256ImpresEpoch14740.png)

#### 512x512px Paintings Training
![gan512Paintings Training](readmeImages/gifs/gan512.gif)
#### 512x512px Paintings Final Output - 510 training epochs
![gan512Paintings Final Output](readmeImages/stills/gan512Epoch510.png)

#### 512x512px Paintings Training
![gan512Paintings Training](readmeImages/gifs/gan512.gif)
#### 512x512px Paintings Final Output - 510 training epochs
![gan512Paintings Final Output](readmeImages/stills/gan512Epoch510.png)

#### 256x256px Chuck Close Artwork Training
![ganChuck256 Training](readmeImages/gifs/ganChuck256.gif)
#### 256x256px Chuck Close Final Output - 1200 training epochs
![ganChuck256 Final Output](readmeImages/stills/ganChuck256Epoch1200.png)

#### kerasGANv1 Training
![kerasGanv1 Training](readmeImages/gifs/kerasGANv1-2.gif)
#### kerasGANv1 Final Output - 4900 training epochs
![kerasGANv1 Final Output](readmeImages/stills/kerasGANv1-2Epoch4900.png)


### Future Work

* adapt for GPU calculations

## Authors

* **Lucas Lyon** - [lalyon](https://github.com/lalyon)


## Acknowledgments
Shoutout to the following people, whose code was invaluable while developing these neural networks.

* [Robbie Barrat](https://github.com/robbiebarrat/art-DCGAN)
* [Soumith Chintala](https://github.com/soumith/dcgan.torch)
* [Felix Mohr](https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/DCGAN-MNIST.ipynb). 



