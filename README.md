# Computer Vision Lab1

## This repository content six labs describe as following:

> Lab1

>> In the first lab, we are going to train a logistic regressor on the MNIST dataset of handwritten digits. Next, we will turn this logistic regressor into a non-linear convolutional network.

> Lab2

>> To have the logistic regressor output probabilities, we need a softmax layer. In this lab we are going to implement a softmax layer. We will discover what numerical issues may arise in this layer? How can we solve them?

> Lab3

>> The spatial dimensions of the ouput image (width and height) depend on the spatial dimensions of the input image, kernel_size, padding, and striding. In order to build efficient convolutional networks, it's important to understand what the sizes are after after each convolutional layer.

In this exersise we will derive the dependency between input and output image sizes. For the sake of simplicity we assume that the input tensor is **_square_**, i.e., width = height = image_size. We will use the nn.Conv2d layer here.

> Lab4

>> In the lab, we implemented a naive convolution. In this section you will implement your own version of forward pass of nn.Conv2d without using any of PyTorch's (or numpy's) pre-defined convolutional functions.

> Lab5

>> We will now replace the logistic regressor from the lab4 by a small convolutional network with two convolutional layers and a linear layer, and ReLU activations in between the layers. Implement the model and use the same functions as before to train and test the convolutional network.

>> Batch normalization is tenchique that allows to make training more stable fast [1]. In this lab we define a convolutional network with 3 layers. Before each ReLU layer we insert a BatchNorm2d layer if `use_batch_norm` is `True`. This improves the convergence as guarantees as values have the same variance asn zero-means. As a result on average exactly half of the values will be nulled by ReLU.

[1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015).

> Lab6

>> In this lab, we are going train a deep neural network using residual connections [2]. The network below has a list of `conv_blocks`. Each convolutional block is convolution layer followed by a ReLU with optional pooling.

[2] He, Kaiming, et al. "Identity mappings in deep residual networks." European conference on computer vision. Springer, Cham, 2016.

>> Imagenet is the most famous dataset for image classification that is still in use. Real ImageNet dataset is very big (~150Gb). So we will use a smaller version that contains only two classes: bees and ants. First, download the required files and construct the dataset.