# ResNet-50 implementation
## Dataset
https://www.kaggle.com/gpiosenka/100-bird-species
## Paper 
https://arxiv.org/abs/1512.03385
### Paper Summary
#### Introduction
Deep CNNs have led to numerous breakthroughs of image classification.  
Deep net naturally integrate different-level features with classifiers in an end to end multi-layer trend, and level of layer can be enriched depending on the depth.  
Dive to the significant of depth, there is a problem of vanishing/exploding gradient. However, this one can be handled by normalized intialization and intermediate normalization layers.  
When the deeper networks are able to start converging, degradation probem emerged : the accuracy becomes saturated and degrades rapidly. However, this is not caused by overfitting and adding more layers leads to higher training error.  
They considered a shallower architecture and its deeper couterpart added more layers onto it. There is a solution for this problem : the added layers are identity mapping and other layers are copied from the learned shallower model and they can't find any way that better than this one (or unable to do so in feasible time).  
In this paper, they addressed this degradation problem by introducing a deep residual learning network. They explicitly let these layers fit a residual mapping instead of hoping each few stacked layers directly fit a desired underlying mapping. Denoting the underlying map is H(x), they let the stacked non-linear layers fit another mapping of F(x) = H(x) - x. The original mapping is recast to F(x) + x. They hypothesized that it's easier to optimize the residual mapping than the original, unreferenced mapping. To be extreme, if an identity mapping is already optimzed, it will be easy to push the residual to zero than fit and identity mapping by a stack of nonlinear layer.  
The formulation of F(x) + x can be realized by feedforward neural networks with "shortcut connections". In their case, the shortcut connection simply perform identity mapping and their outputs are added to the output of the stacked layers. Identity short-cut connections add neither extra parameter nor computational complexity.   
Their formulation always learns residual function and their identity shortcuts are never closed - all information is always passed through with additional residual function to be learned.  
### Deep Residual Learning
#### Residual Learning
Denoting that H(x) is an underlying mapping to be fit by a few stacked layers (x is the inputs of these stack). If one hypothesizes that multiple nonlinear layer can asymtotically approximate complex function, then it equivalent to hypothesize that they can asymtotically approximate the residual function, i.e. F(x) = H(x) - x (Equation 1). The original function becomes F(x) + x. Both form do the same task but the ease of learning might be different. 
#### Identity Mapping by Shortcuts
They adopt residual learning to every few stacked layers. We consider a building block defined as y = F(x,{W_i}) + x. The operation F + x is performed by a shortcut connection and element-wise addition and then they adopt the activation. As you can see, Equation 1 introduce neither extra parameters nor computation complexity. Another thing worth mentioning is the equality in dimensions of x and F. If it is not, we can use linear projection W_s by the shortcut connections to mach the dimensions (in practice they prefer to use convolution to handle this): y = F(x, {W_i}) + W_s * x (Equation 2).   
#### Network architectures
##### Plain network
Inspired by VGG nets, the convolutional layers mostly have 3 x 3 filters and follow two simple design reules : for the same output feature map size, the numbers of filters is kept; and if the feature map size is halved, the quantity of filters is doubled in order to preserve the time complexity per layer. They performed downsampling directly by stride-equal-2 convolutions. The network end with a global average pooling layer and a 1000-way FC layer with softmax. 
##### Residual network
Base on the plain network, they insert connections, which turn it into its counterpart residual verion. When the dimesions is increased, there are 2 options for this : 
* The shortcut still performs identity mapping, with extra zero padding for increase dimensions, which introduce no extra parameter.  
* The projection shortcut in Equation 2 is used to match dimensions (done by 1x1 convolutions).
For these options, when the shortcut go across feature maps of 2 different sizes, they are performed with a stride 2. 
#### Implementation   


## Implementation Details