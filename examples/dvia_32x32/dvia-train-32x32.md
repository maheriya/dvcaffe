
# cifar-100 conv net with Caffe for DVIA

Experimental layers. Includes NiN, BN and Dropout.

## Download and convert the cifar-100 dataset to LMDB


```python
##--%%time
##--!python download-cifar-100.py
##--!ipython convert-cifar-100-32x32.ipy
```

    
    Downloading...
    Dataset already downloaded. Did not download twice.
    
    Extracting...
    Dataset already extracted. Did not extract twice.
    
    Converting...
    Conversion was already done. Did not convert twice.
    
    CPU times: user 8 ms, sys: 16 ms, total: 24 ms
    Wall time: 1.17 s


## Build the model with Caffe. 


```python
import numpy as np
import os, sys
import re

scriptpath    = os.path.dirname(os.path.realpath( "xxxx" ))
caffe_root    = os.path.sep.join(scriptpath.split(os.path.sep)[:-2])
#caffe_root  = os.path.join(os.environ['HOME'], 'Projects', 'dvcaffe')
cifar_db_root = os.path.join(os.environ['HOME'], 'Projects', 'IMAGES', 'dvia', 'cifar_png.32x32')
dvia_db_root  = os.path.join(os.environ['HOME'], 'Projects', 'IMAGES', 'dvia', 'png.32x32')

import caffe
from caffe import layers as L
from caffe import params as P

print "scriptpath = {}".format(scriptpath)
print "caffe_root = {}".format(caffe_root)
print "cifar_db_root = {}".format(cifar_db_root)
print "dvia_db_root = {}".format(dvia_db_root)
```

    scriptpath = /home/maheriya/Projects/dvcaffe/examples/dvia_32x32
    caffe_root = /home/maheriya/Projects/dvcaffe
    cifar_db_root = /home/maheriya/Projects/IMAGES/dvia/cifar_png.32x32
    dvia_db_root = /home/maheriya/Projects/IMAGES/dvia/png.32x32


## Load and visualise the untrained network's internal structure and shape
The network's structure (graph) visualisation tool of caffe is broken in the current release. We will simply print here the data shapes. 


```python
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
## Use for training from scratch
learned_param = [weight_param, bias_param]

frozen_weight_param = dict(lr_mult=0.2, decay_mult=0.2)  # *0.2
frozen_bias_param   = dict(lr_mult=0.4, decay_mult=0)    # *0.2
## Use for training from a pretrained model
frozen_param = [frozen_weight_param, frozen_bias_param]

wgt_filler = {'type': 'xavier'}
bn_param = '''param {
    lr_mult: 0
  }
  param {
    lr_mult: 0
  }
  param {
    lr_mult: 0
  }'''

low_dropout = {'dropout_ratio': 0.3}
mid_dropout = {'dropout_ratio': 0.5}


def cnn_inner_layers(n, param=learned_param):
    '''
    n: caffe.NetSpec instance
    It is assumed that n.data is already created.
    '''
    # First main conv layer
    n.conv1  = L.Convolution(n.data,    kernel_size=5, stride=1, num_output=64, weight_filler=wgt_filler, param=param)
    n.pool1  = L.Pooling(n.conv1,       kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.relu1b = L.ReLU(n.pool1, in_place=True)

    # Second main conv layer
    n.conv2  = L.Convolution(n.relu1b,   kernel_size=3, stride=1, num_output=100, weight_filler=wgt_filler, param=param)
    n.pool2  = L.Pooling(n.conv2,        kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.relu2  = L.ReLU(n.pool2, in_place=True)

    # Third and last main convolution layer.
    n.conv3  = L.Convolution(n.relu2,    kernel_size=3, stride=1, num_output=200, weight_filler=wgt_filler, param=learned_param)
    n.pool3  = L.Pooling(n.conv3,        kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.relu3  = L.ReLU(n.pool3, in_place=True)

    # Last fc converted to convolution.
    n.conv_last = L.Convolution(n.relu3, kernel_size=1, stride=1, num_output=384, weight_filler=wgt_filler, param=learned_param)
    n.relu_last = L.ReLU(n.conv_last, in_place=True)
    #--n.fc_last   = L.InnerProduct(n.relu3, num_output=768, weight_filler=wgt_filler, param=learned_param)
    #--n.relu_last = L.ReLU(n.fc_last, in_place=True)
    return n


# For pre-training
def cnn_cifar(imgdb, mean_file, batch_size, mirror=True):
    n = caffe.NetSpec()
    #n.data, n.label_coarse, n.label_fine = L.HDF5Data(batch_size=batch_size, source=imgdb, ntop=3)
    n.data, n.label_coarse = L.Data(batch_size=batch_size, source=imgdb, backend=P.Data.LMDB, 
                             transform_param=dict(scale=1./256, mirror=mirror, mean_file=mean_file), ntop=2)

    # Create inner layers
    n = cnn_inner_layers(n)

    # Output: 20-class and 100-class classifiers
    n.fc_coarse       = L.InnerProduct(n.relu_last, num_output=20, weight_filler=wgt_filler, param=learned_param)
    n.accuracy_coarse = L.Accuracy(n.fc_coarse, n.label_coarse)
    n.loss_coarse     = L.SoftmaxWithLoss(n.fc_coarse, n.label_coarse)
#     n.fc_coarse       = L.Convolution(n.drop_last, kernel_size=1, stride=1, num_output=20, weight_filler=wgt_filler, param=learned_param)
#     n.fc_avpool_coarse= L.Pooling(n.fc_coarse, kernel_size=2, stride=2, pool=P.Pooling.AVE)
#     n.accuracy_coarse = L.Accuracy(n.fc_avpool_coarse, n.label_coarse)
#     n.loss_coarse     = L.SoftmaxWithLoss(n.fc_avpool_coarse, n.label_coarse, loss_weight=0.65)

#     n.fc_fine         = L.InnerProduct(n.drop_last, num_output=100, weight_filler=wgt_filler, param=learned_param)
#     n.accuracy_fine   = L.Accuracy(n.fc_fine, n.label_fine)
#     n.loss_fine       = L.SoftmaxWithLoss(n.fc_fine, n.label_fine, loss_weight=0.35)
##     n.fc_avpool_fine  = L.Pooling(n.fc_fine, kernel_size=2, stride=2, pool=P.Pooling.AVE)
##     n.accuracy_fine   = L.Accuracy(n.fc_avpool_fine, n.label_fine)
##     n.loss_fine       = L.SoftmaxWithLoss(n.fc_avpool_fine, n.label_fine, loss_weight=0.35)

    return n.to_proto()

with open('dvia_pretrain.prototxt', 'w') as f:
    lmdb      = os.path.join(cifar_db_root, 'data/cifar_32x32/trn_lmdb')
    mean_file = os.path.join(cifar_db_root, 'data/cifar_32x32/trn_mean.binaryproto')
    prto = str(cnn_cifar(lmdb, mean_file, 100, mirror=True))
    prto = re.sub(r'top: "(bn[0-3])"(\s+)param {[^}]+}', 'top: "\\1"\\2{}'.format(bn_param), prto)
    f.write(prto)
    
with open('dvia_pretest.prototxt', 'w') as f:
    lmdb      = os.path.join(cifar_db_root, 'data/cifar_32x32/val_lmdb')
    mean_file = os.path.join(cifar_db_root, 'data/cifar_32x32/val_mean.binaryproto')
    prto = str(cnn_cifar(lmdb, mean_file, 120, mirror=False))
    prto = re.sub(r'top: "(bn[0-3])"(\s+)param {[^}]+}', 'top: "\\1"\\2{}'.format(bn_param), prto)
    f.write(prto)

!python /usr/local/caffe/python/draw_net.py dvia_pretrain.prototxt cifar_net.png

# For training
def cnn(lmdb, mean_file, batch_size, mirror=True):
    n = caffe.NetSpec()
    ## Input LMDB data layer
    n.data, n.label = L.Data(batch_size=batch_size, source=lmdb, backend=P.Data.LMDB, 
                             transform_param=dict(scale=1./256, mirror=True, mean_file=mean_file), ntop=2)

    # Create inner layers
    n = cnn_inner_layers(n, frozen_param)

    # Output 4-class classifier
    n.fc_class         = L.InnerProduct(n.relu_last, num_output=4, weight_filler=wgt_filler, param=learned_param)
    n.accuracy_class   = L.Accuracy(n.fc_class, n.label)
    n.loss_class       = L.SoftmaxWithLoss(n.fc_class, n.label)

##     n.fc_class         = L.Convolution(n.drop_last, kernel_size=1, stride=1, num_output=4, weight_filler=wgt_filler, param=learned_param)
##     n.fc_avpool_class  = L.Pooling(n.fc_class, kernel_size=2, stride=2, pool=P.Pooling.AVE)
##     n.accuracy_class   = L.Accuracy(n.fc_avpool_class, n.label)
##     n.loss_class       = L.SoftmaxWithLoss(n.fc_avpool_class, n.label)

    return n.to_proto()

with open('dvia_train.prototxt', 'w') as f:
    lmdb      = os.path.join(dvia_db_root, 'data/dvia_32x32/trn_lmdb')
    mean_file = os.path.join(dvia_db_root, 'data/dvia_32x32/trn_mean.binaryproto')
    prto = str(cnn(lmdb, mean_file, 100, mirror=True))
    prto = re.sub(r'top: "(bn[0-3])"(\s+)param {[^}]+}', 'top: "\\1"\\2{}'.format(bn_param), prto)
    f.write(prto)
    
with open('dvia_test.prototxt', 'w') as f:
    lmdb      = os.path.join(dvia_db_root, 'data/dvia_32x32/val_lmdb')
    mean_file = os.path.join(dvia_db_root, 'data/dvia_32x32/val_mean.binaryproto')
    prto = str(cnn(lmdb, mean_file, 120, mirror=True))
    prto = re.sub(r'top: "(bn[0-3])"(\s+)param {[^}]+}', 'top: "\\1"\\2{}'.format(bn_param), prto)
    f.write(prto)

!python /usr/local/caffe/python/draw_net.py dvia_train.prototxt dvia_net.png
```

    Drawing net to cifar_net.png
    Drawing net to dvia_net.png



```python
caffe.set_mode_gpu()
solver = None
solver = caffe.get_solver('dvia_solver.prototxt')
```


```python
print("Layers' features:")
[(k, v.data.shape) for k, v in solver.net.blobs.items()]
```

    Layers' features:





    [('data', (100, 3, 32, 32)),
     ('label', (100,)),
     ('label_data_1_split_0', (100,)),
     ('label_data_1_split_1', (100,)),
     ('conv1', (100, 64, 28, 28)),
     ('pool1', (100, 64, 14, 14)),
     ('conv2', (100, 100, 12, 12)),
     ('pool2', (100, 100, 6, 6)),
     ('conv3', (100, 200, 4, 4)),
     ('pool3', (100, 200, 2, 2)),
     ('conv_last', (100, 384, 2, 2)),
     ('fc_class', (100, 4)),
     ('fc_class_fc_class_0_split_0', (100, 4)),
     ('fc_class_fc_class_0_split_1', (100, 4)),
     ('accuracy_class', ()),
     ('loss_class', ())]




```python
print("Parameters and shape:")
[(k, v[0].data.shape) for k, v in solver.net.params.items()]
```

    Parameters and shape:





    [('conv1', (64, 3, 5, 5)),
     ('conv2', (100, 64, 3, 3)),
     ('conv3', (200, 100, 3, 3)),
     ('conv_last', (384, 200, 1, 1)),
     ('fc_class', (4, 1536))]



## Pre-Train Using Cifar-100 DB (32x32 original images)
The purpose of this pre-training part is to take advantage of the Cifar-100 database to get better feature extractor as a initial condition for later training with our own image database. 


```python
solver = None
```


```python
%%time
!caffe train -solver dvia_presolver.prototxt
```


```python
!ls -rt cifar_pretrain_iter*.caffemodel | tail -n1 | xargs -i cp {} cifar_pretrained.caffemodel
```

## Solver's params

The solver's params for the created net are defined in a `.prototxt` file. 

Notice that because `max_iter: 100000`, the training will loop 2 times on the 50000 training data. Because we train data by minibatches of 100 as defined above when creating the net, there will be a total of `100000*100/50000 = 200` epochs on some of those pre-shuffled 100 images minibatches.

We will test the net on `test_iter: 100` different test images at each `test_interval: 1000` images trained. 
____

Here, **RMSProp** is used, it is SDG-based, it converges faster than a pure SGD and it is robust.
____


```python
!cat dvia_solver.prototxt
```

    train_net: "dvia_train.prototxt"
    test_net: "dvia_test.prototxt"
    
    test_iter: 100
    test_interval: 1000
    
    base_lr: 0.001
    momentum: 0.0
    weight_decay: 0.001
    
    lr_policy: "inv"
    gamma: 0.0001
    power: 0.75
    
    display: 100
    
    max_iter: 100000
    
    snapshot: 25000
    snapshot_prefix: "dvia_train"
    solver_mode: GPU
    
    type: "RMSProp"
    rms_decay: 0.98


## Alternative way to train directly in Python
Since a recent update, there is no output in python by default, which is bad for debugging. 
Skip this cell and train with the second method shown below if needed. It is commented out in case you just chain some `shift+enter` ipython shortcuts. 


```python
# %%time
# solver.solve()
solver = None
```

## Train by calling caffe in command line
Just set the parameters correctly. Be sure that the notebook is at the root of the ipython notebook server. 
You can run this in an external terminal if you open it in the notebook's directory. 

It is also possible to finetune an existing net with a different solver or different data. Here I do it, because I feel the net could better fit the data. 


```python
%%time
!caffe train -solver dvia_solver.prototxt -weights cifar_pretrained.caffemodel
```

    CPU times: user 0 ns, sys: 0 ns, total: 0 ns
    Wall time: 3.1 µs


Caffe brewed. 
## Test the model completely on test data
Let's test directly in command-line:


```python
%%time
!ls -rt dvia_train_iter*.caffemodel | tail -n1 | xargs -i cp {} dvia_trained.caffemodel
!caffe test -model dvia_test.prototxt -weights dvia_trained.caffemodel -iterations 100
```

    I1112 15:22:32.683954 16332 caffe.cpp:279] Use CPU.
    I1112 15:22:32.881461 16332 net.cpp:58] Initializing net from parameters: 
    state {
      phase: TEST
      level: 0
      stage: ""
    }
    layer {
      name: "data"
      type: "Data"
      top: "data"
      top: "label"
      transform_param {
        scale: 0.00390625
        mirror: true
        mean_file: "/home/maheriya/Projects/IMAGES/dvia/png.32x32/data/dvia_32x32/val_mean.binaryproto"
      }
      data_param {
        source: "/home/maheriya/Projects/IMAGES/dvia/png.32x32/data/dvia_32x32/val_lmdb"
        batch_size: 120
        backend: LMDB
      }
    }
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      param {
        lr_mult: 0.2
        decay_mult: 0.2
      }
      param {
        lr_mult: 0.4
        decay_mult: 0
      }
      convolution_param {
        num_output: 64
        kernel_size: 5
        stride: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool1"
      type: "Pooling"
      bottom: "conv1"
      top: "pool1"
      pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
      }
    }
    layer {
      name: "relu1b"
      type: "ReLU"
      bottom: "pool1"
      top: "pool1"
    }
    layer {
      name: "conv2"
      type: "Convolution"
      bottom: "pool1"
      top: "conv2"
      param {
        lr_mult: 0.2
        decay_mult: 0.2
      }
      param {
        lr_mult: 0.4
        decay_mult: 0
      }
      convolution_param {
        num_output: 100
        kernel_size: 3
        stride: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool2"
      type: "Pooling"
      bottom: "conv2"
      top: "pool2"
      pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
      }
    }
    layer {
      name: "relu2"
      type: "ReLU"
      bottom: "pool2"
      top: "pool2"
    }
    layer {
      name: "conv3"
      type: "Convolution"
      bottom: "pool2"
      top: "conv3"
      param {
        lr_mult: 1
        decay_mult: 1
      }
      param {
        lr_mult: 2
        decay_mult: 0
      }
      convolution_param {
        num_output: 200
        kernel_size: 3
        stride: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool3"
      type: "Pooling"
      bottom: "conv3"
      top: "pool3"
      pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
      }
    }
    layer {
      name: "relu3"
      type: "ReLU"
      bottom: "pool3"
      top: "pool3"
    }
    layer {
      name: "conv_last"
      type: "Convolution"
      bottom: "pool3"
      top: "conv_last"
      param {
        lr_mult: 1
        decay_mult: 1
      }
      param {
        lr_mult: 2
        decay_mult: 0
      }
      convolution_param {
        num_output: 384
        kernel_size: 1
        stride: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu_last"
      type: "ReLU"
      bottom: "conv_last"
      top: "conv_last"
    }
    layer {
      name: "fc_class"
      type: "InnerProduct"
      bottom: "conv_last"
      top: "fc_class"
      param {
        lr_mult: 1
        decay_mult: 1
      }
      param {
        lr_mult: 2
        decay_mult: 0
      }
      inner_product_param {
        num_output: 4
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "accuracy_class"
      type: "Accuracy"
      bottom: "fc_class"
      bottom: "label"
      top: "accuracy_class"
    }
    layer {
      name: "loss_class"
      type: "SoftmaxWithLoss"
      bottom: "fc_class"
      bottom: "label"
      top: "loss_class"
    }
    I1112 15:22:32.881676 16332 layer_factory.hpp:77] Creating layer data
    I1112 15:22:32.882624 16332 net.cpp:100] Creating Layer data
    I1112 15:22:32.882637 16332 net.cpp:408] data -> data
    I1112 15:22:32.882655 16332 net.cpp:408] data -> label
    I1112 15:22:32.882678 16332 data_transformer.cpp:25] Loading mean file from: /home/maheriya/Projects/IMAGES/dvia/png.32x32/data/dvia_32x32/val_mean.binaryproto
    I1112 15:22:32.883309 16337 db_lmdb.cpp:35] Opened lmdb /home/maheriya/Projects/IMAGES/dvia/png.32x32/data/dvia_32x32/val_lmdb
    I1112 15:22:32.883611 16332 data_layer.cpp:41] output data size: 120,3,32,32
    I1112 15:22:32.885067 16332 net.cpp:150] Setting up data
    I1112 15:22:32.885093 16332 net.cpp:157] Top shape: 120 3 32 32 (368640)
    I1112 15:22:32.885102 16332 net.cpp:157] Top shape: 120 (120)
    I1112 15:22:32.885103 16332 net.cpp:165] Memory required for data: 1475040
    I1112 15:22:32.885115 16332 layer_factory.hpp:77] Creating layer label_data_1_split
    I1112 15:22:32.885128 16332 net.cpp:100] Creating Layer label_data_1_split
    I1112 15:22:32.885133 16332 net.cpp:434] label_data_1_split <- label
    I1112 15:22:32.885148 16332 net.cpp:408] label_data_1_split -> label_data_1_split_0
    I1112 15:22:32.885179 16332 net.cpp:408] label_data_1_split -> label_data_1_split_1
    I1112 15:22:32.885188 16332 net.cpp:150] Setting up label_data_1_split
    I1112 15:22:32.885192 16332 net.cpp:157] Top shape: 120 (120)
    I1112 15:22:32.885195 16332 net.cpp:157] Top shape: 120 (120)
    I1112 15:22:32.885198 16332 net.cpp:165] Memory required for data: 1476000
    I1112 15:22:32.885201 16332 layer_factory.hpp:77] Creating layer conv1
    I1112 15:22:32.885216 16332 net.cpp:100] Creating Layer conv1
    I1112 15:22:32.885220 16332 net.cpp:434] conv1 <- data
    I1112 15:22:32.885226 16332 net.cpp:408] conv1 -> conv1
    I1112 15:22:32.885290 16332 net.cpp:150] Setting up conv1
    I1112 15:22:32.885294 16332 net.cpp:157] Top shape: 120 64 28 28 (6021120)
    I1112 15:22:32.885296 16332 net.cpp:165] Memory required for data: 25560480
    I1112 15:22:32.885310 16332 layer_factory.hpp:77] Creating layer pool1
    I1112 15:22:32.885316 16332 net.cpp:100] Creating Layer pool1
    I1112 15:22:32.885318 16332 net.cpp:434] pool1 <- conv1
    I1112 15:22:32.885321 16332 net.cpp:408] pool1 -> pool1
    I1112 15:22:32.885334 16332 net.cpp:150] Setting up pool1
    I1112 15:22:32.885339 16332 net.cpp:157] Top shape: 120 64 14 14 (1505280)
    I1112 15:22:32.885341 16332 net.cpp:165] Memory required for data: 31581600
    I1112 15:22:32.885344 16332 layer_factory.hpp:77] Creating layer relu1b
    I1112 15:22:32.885349 16332 net.cpp:100] Creating Layer relu1b
    I1112 15:22:32.885352 16332 net.cpp:434] relu1b <- pool1
    I1112 15:22:32.885355 16332 net.cpp:395] relu1b -> pool1 (in-place)
    I1112 15:22:32.885360 16332 net.cpp:150] Setting up relu1b
    I1112 15:22:32.885363 16332 net.cpp:157] Top shape: 120 64 14 14 (1505280)
    I1112 15:22:32.885366 16332 net.cpp:165] Memory required for data: 37602720
    I1112 15:22:32.885370 16332 layer_factory.hpp:77] Creating layer conv2
    I1112 15:22:32.885375 16332 net.cpp:100] Creating Layer conv2
    I1112 15:22:32.885376 16332 net.cpp:434] conv2 <- pool1
    I1112 15:22:32.885382 16332 net.cpp:408] conv2 -> conv2
    I1112 15:22:32.885792 16332 net.cpp:150] Setting up conv2
    I1112 15:22:32.885797 16332 net.cpp:157] Top shape: 120 100 12 12 (1728000)
    I1112 15:22:32.885802 16332 net.cpp:165] Memory required for data: 44514720
    I1112 15:22:32.885807 16332 layer_factory.hpp:77] Creating layer pool2
    I1112 15:22:32.885812 16332 net.cpp:100] Creating Layer pool2
    I1112 15:22:32.885815 16332 net.cpp:434] pool2 <- conv2
    I1112 15:22:32.885818 16332 net.cpp:408] pool2 -> pool2
    I1112 15:22:32.885825 16332 net.cpp:150] Setting up pool2
    I1112 15:22:32.885828 16332 net.cpp:157] Top shape: 120 100 6 6 (432000)
    I1112 15:22:32.885831 16332 net.cpp:165] Memory required for data: 46242720
    I1112 15:22:32.885833 16332 layer_factory.hpp:77] Creating layer relu2
    I1112 15:22:32.885838 16332 net.cpp:100] Creating Layer relu2
    I1112 15:22:32.885840 16332 net.cpp:434] relu2 <- pool2
    I1112 15:22:32.885844 16332 net.cpp:395] relu2 -> pool2 (in-place)
    I1112 15:22:32.885848 16332 net.cpp:150] Setting up relu2
    I1112 15:22:32.885851 16332 net.cpp:157] Top shape: 120 100 6 6 (432000)
    I1112 15:22:32.885854 16332 net.cpp:165] Memory required for data: 47970720
    I1112 15:22:32.885856 16332 layer_factory.hpp:77] Creating layer conv3
    I1112 15:22:32.885862 16332 net.cpp:100] Creating Layer conv3
    I1112 15:22:32.885865 16332 net.cpp:434] conv3 <- pool2
    I1112 15:22:32.885869 16332 net.cpp:408] conv3 -> conv3
    I1112 15:22:32.887289 16332 net.cpp:150] Setting up conv3
    I1112 15:22:32.887296 16332 net.cpp:157] Top shape: 120 200 4 4 (384000)
    I1112 15:22:32.887300 16332 net.cpp:165] Memory required for data: 49506720
    I1112 15:22:32.887305 16332 layer_factory.hpp:77] Creating layer pool3
    I1112 15:22:32.887311 16332 net.cpp:100] Creating Layer pool3
    I1112 15:22:32.887317 16332 net.cpp:434] pool3 <- conv3
    I1112 15:22:32.887322 16332 net.cpp:408] pool3 -> pool3
    I1112 15:22:32.887328 16332 net.cpp:150] Setting up pool3
    I1112 15:22:32.887332 16332 net.cpp:157] Top shape: 120 200 2 2 (96000)
    I1112 15:22:32.887336 16332 net.cpp:165] Memory required for data: 49890720
    I1112 15:22:32.887338 16332 layer_factory.hpp:77] Creating layer relu3
    I1112 15:22:32.887359 16332 net.cpp:100] Creating Layer relu3
    I1112 15:22:32.887362 16332 net.cpp:434] relu3 <- pool3
    I1112 15:22:32.887367 16332 net.cpp:395] relu3 -> pool3 (in-place)
    I1112 15:22:32.887372 16332 net.cpp:150] Setting up relu3
    I1112 15:22:32.887375 16332 net.cpp:157] Top shape: 120 200 2 2 (96000)
    I1112 15:22:32.887378 16332 net.cpp:165] Memory required for data: 50274720
    I1112 15:22:32.887382 16332 layer_factory.hpp:77] Creating layer conv_last
    I1112 15:22:32.887387 16332 net.cpp:100] Creating Layer conv_last
    I1112 15:22:32.887390 16332 net.cpp:434] conv_last <- pool3
    I1112 15:22:32.887394 16332 net.cpp:408] conv_last -> conv_last
    I1112 15:22:32.887974 16332 net.cpp:150] Setting up conv_last
    I1112 15:22:32.887979 16332 net.cpp:157] Top shape: 120 384 2 2 (184320)
    I1112 15:22:32.887981 16332 net.cpp:165] Memory required for data: 51012000
    I1112 15:22:32.887985 16332 layer_factory.hpp:77] Creating layer relu_last
    I1112 15:22:32.887990 16332 net.cpp:100] Creating Layer relu_last
    I1112 15:22:32.887994 16332 net.cpp:434] relu_last <- conv_last
    I1112 15:22:32.887997 16332 net.cpp:395] relu_last -> conv_last (in-place)
    I1112 15:22:32.888001 16332 net.cpp:150] Setting up relu_last
    I1112 15:22:32.888005 16332 net.cpp:157] Top shape: 120 384 2 2 (184320)
    I1112 15:22:32.888007 16332 net.cpp:165] Memory required for data: 51749280
    I1112 15:22:32.888010 16332 layer_factory.hpp:77] Creating layer fc_class
    I1112 15:22:32.888015 16332 net.cpp:100] Creating Layer fc_class
    I1112 15:22:32.888017 16332 net.cpp:434] fc_class <- conv_last
    I1112 15:22:32.888022 16332 net.cpp:408] fc_class -> fc_class
    I1112 15:22:32.888074 16332 net.cpp:150] Setting up fc_class
    I1112 15:22:32.888078 16332 net.cpp:157] Top shape: 120 4 (480)
    I1112 15:22:32.888082 16332 net.cpp:165] Memory required for data: 51751200
    I1112 15:22:32.888087 16332 layer_factory.hpp:77] Creating layer fc_class_fc_class_0_split
    I1112 15:22:32.888092 16332 net.cpp:100] Creating Layer fc_class_fc_class_0_split
    I1112 15:22:32.888093 16332 net.cpp:434] fc_class_fc_class_0_split <- fc_class
    I1112 15:22:32.888098 16332 net.cpp:408] fc_class_fc_class_0_split -> fc_class_fc_class_0_split_0
    I1112 15:22:32.888103 16332 net.cpp:408] fc_class_fc_class_0_split -> fc_class_fc_class_0_split_1
    I1112 15:22:32.888108 16332 net.cpp:150] Setting up fc_class_fc_class_0_split
    I1112 15:22:32.888111 16332 net.cpp:157] Top shape: 120 4 (480)
    I1112 15:22:32.888114 16332 net.cpp:157] Top shape: 120 4 (480)
    I1112 15:22:32.888116 16332 net.cpp:165] Memory required for data: 51755040
    I1112 15:22:32.888119 16332 layer_factory.hpp:77] Creating layer accuracy_class
    I1112 15:22:32.888123 16332 net.cpp:100] Creating Layer accuracy_class
    I1112 15:22:32.888139 16332 net.cpp:434] accuracy_class <- fc_class_fc_class_0_split_0
    I1112 15:22:32.888144 16332 net.cpp:434] accuracy_class <- label_data_1_split_0
    I1112 15:22:32.888149 16332 net.cpp:408] accuracy_class -> accuracy_class
    I1112 15:22:32.888154 16332 net.cpp:150] Setting up accuracy_class
    I1112 15:22:32.888156 16332 net.cpp:157] Top shape: (1)
    I1112 15:22:32.888159 16332 net.cpp:165] Memory required for data: 51755044
    I1112 15:22:32.888160 16332 layer_factory.hpp:77] Creating layer loss_class
    I1112 15:22:32.888166 16332 net.cpp:100] Creating Layer loss_class
    I1112 15:22:32.888170 16332 net.cpp:434] loss_class <- fc_class_fc_class_0_split_1
    I1112 15:22:32.888172 16332 net.cpp:434] loss_class <- label_data_1_split_1
    I1112 15:22:32.888176 16332 net.cpp:408] loss_class -> loss_class
    I1112 15:22:32.888182 16332 layer_factory.hpp:77] Creating layer loss_class
    I1112 15:22:32.888195 16332 net.cpp:150] Setting up loss_class
    I1112 15:22:32.888197 16332 net.cpp:157] Top shape: (1)
    I1112 15:22:32.888200 16332 net.cpp:160]     with loss weight 1
    I1112 15:22:32.888227 16332 net.cpp:165] Memory required for data: 51755048
    I1112 15:22:32.888231 16332 net.cpp:226] loss_class needs backward computation.
    I1112 15:22:32.888239 16332 net.cpp:228] accuracy_class does not need backward computation.
    I1112 15:22:32.888242 16332 net.cpp:226] fc_class_fc_class_0_split needs backward computation.
    I1112 15:22:32.888254 16332 net.cpp:226] fc_class needs backward computation.
    I1112 15:22:32.888257 16332 net.cpp:226] relu_last needs backward computation.
    I1112 15:22:32.888262 16332 net.cpp:226] conv_last needs backward computation.
    I1112 15:22:32.888264 16332 net.cpp:226] relu3 needs backward computation.
    I1112 15:22:32.888267 16332 net.cpp:226] pool3 needs backward computation.
    I1112 15:22:32.888270 16332 net.cpp:226] conv3 needs backward computation.
    I1112 15:22:32.888273 16332 net.cpp:226] relu2 needs backward computation.
    I1112 15:22:32.888276 16332 net.cpp:226] pool2 needs backward computation.
    I1112 15:22:32.888283 16332 net.cpp:226] conv2 needs backward computation.
    I1112 15:22:32.888285 16332 net.cpp:226] relu1b needs backward computation.
    I1112 15:22:32.888289 16332 net.cpp:226] pool1 needs backward computation.
    I1112 15:22:32.888293 16332 net.cpp:226] conv1 needs backward computation.
    I1112 15:22:32.888295 16332 net.cpp:228] label_data_1_split does not need backward computation.
    I1112 15:22:32.888298 16332 net.cpp:228] data does not need backward computation.
    I1112 15:22:32.888301 16332 net.cpp:270] This network produces output accuracy_class
    I1112 15:22:32.888304 16332 net.cpp:270] This network produces output loss_class
    I1112 15:22:32.888317 16332 net.cpp:283] Network initialization done.
    I1112 15:22:32.889116 16332 caffe.cpp:285] Running for 100 iterations.
    I1112 15:22:32.889137 16332 blocking_queue.cpp:50] Data layer prefetch queue empty
    I1112 15:22:33.678433 16332 caffe.cpp:308] Batch 0, accuracy_class = 0.758333
    I1112 15:22:33.678465 16332 caffe.cpp:308] Batch 0, loss_class = 1.36296
    I1112 15:22:33.976104 16332 caffe.cpp:308] Batch 1, accuracy_class = 0.775
    I1112 15:22:33.976125 16332 caffe.cpp:308] Batch 1, loss_class = 1.4246
    I1112 15:22:34.259084 16332 caffe.cpp:308] Batch 2, accuracy_class = 0.841667
    I1112 15:22:34.259102 16332 caffe.cpp:308] Batch 2, loss_class = 1.08871
    I1112 15:22:34.555891 16332 caffe.cpp:308] Batch 3, accuracy_class = 0.775
    I1112 15:22:34.555912 16332 caffe.cpp:308] Batch 3, loss_class = 1.56853
    I1112 15:22:34.857353 16332 caffe.cpp:308] Batch 4, accuracy_class = 0.8
    I1112 15:22:34.857370 16332 caffe.cpp:308] Batch 4, loss_class = 0.948312
    I1112 15:22:35.157063 16332 caffe.cpp:308] Batch 5, accuracy_class = 0.766667
    I1112 15:22:35.157083 16332 caffe.cpp:308] Batch 5, loss_class = 1.37621
    I1112 15:22:35.454571 16332 caffe.cpp:308] Batch 6, accuracy_class = 0.766667
    I1112 15:22:35.454589 16332 caffe.cpp:308] Batch 6, loss_class = 0.899253
    I1112 15:22:35.764498 16332 caffe.cpp:308] Batch 7, accuracy_class = 0.766667
    I1112 15:22:35.764531 16332 caffe.cpp:308] Batch 7, loss_class = 1.04577
    I1112 15:22:36.063776 16332 caffe.cpp:308] Batch 8, accuracy_class = 0.85
    I1112 15:22:36.063798 16332 caffe.cpp:308] Batch 8, loss_class = 0.9094
    I1112 15:22:36.343297 16332 caffe.cpp:308] Batch 9, accuracy_class = 0.783333
    I1112 15:22:36.343314 16332 caffe.cpp:308] Batch 9, loss_class = 1.5325
    I1112 15:22:36.624696 16332 caffe.cpp:308] Batch 10, accuracy_class = 0.816667
    I1112 15:22:36.624712 16332 caffe.cpp:308] Batch 10, loss_class = 1.30508
    I1112 15:22:36.906775 16332 caffe.cpp:308] Batch 11, accuracy_class = 0.766667
    I1112 15:22:36.906792 16332 caffe.cpp:308] Batch 11, loss_class = 1.19724
    I1112 15:22:37.185494 16332 caffe.cpp:308] Batch 12, accuracy_class = 0.866667
    I1112 15:22:37.185510 16332 caffe.cpp:308] Batch 12, loss_class = 0.652219
    I1112 15:22:37.464215 16332 caffe.cpp:308] Batch 13, accuracy_class = 0.766667
    I1112 15:22:37.464231 16332 caffe.cpp:308] Batch 13, loss_class = 1.38818
    I1112 15:22:37.748214 16332 caffe.cpp:308] Batch 14, accuracy_class = 0.8
    I1112 15:22:37.748229 16332 caffe.cpp:308] Batch 14, loss_class = 1.25893
    I1112 15:22:38.029340 16332 caffe.cpp:308] Batch 15, accuracy_class = 0.791667
    I1112 15:22:38.029359 16332 caffe.cpp:308] Batch 15, loss_class = 1.04652
    I1112 15:22:38.310088 16332 caffe.cpp:308] Batch 16, accuracy_class = 0.833333
    I1112 15:22:38.310104 16332 caffe.cpp:308] Batch 16, loss_class = 0.99371
    I1112 15:22:38.590132 16332 caffe.cpp:308] Batch 17, accuracy_class = 0.841667
    I1112 15:22:38.590152 16332 caffe.cpp:308] Batch 17, loss_class = 0.89693
    I1112 15:22:38.873924 16332 caffe.cpp:308] Batch 18, accuracy_class = 0.808333
    I1112 15:22:38.873942 16332 caffe.cpp:308] Batch 18, loss_class = 0.819305
    I1112 15:22:39.155167 16332 caffe.cpp:308] Batch 19, accuracy_class = 0.816667
    I1112 15:22:39.155184 16332 caffe.cpp:308] Batch 19, loss_class = 1.09079
    I1112 15:22:39.433418 16332 caffe.cpp:308] Batch 20, accuracy_class = 0.791667
    I1112 15:22:39.433435 16332 caffe.cpp:308] Batch 20, loss_class = 1.03289
    I1112 15:22:39.717499 16332 caffe.cpp:308] Batch 21, accuracy_class = 0.85
    I1112 15:22:39.717516 16332 caffe.cpp:308] Batch 21, loss_class = 1.0961
    I1112 15:22:39.998940 16332 caffe.cpp:308] Batch 22, accuracy_class = 0.775
    I1112 15:22:39.998957 16332 caffe.cpp:308] Batch 22, loss_class = 1.36935
    I1112 15:22:40.276515 16332 caffe.cpp:308] Batch 23, accuracy_class = 0.825
    I1112 15:22:40.276532 16332 caffe.cpp:308] Batch 23, loss_class = 0.728407
    I1112 15:22:40.554708 16332 caffe.cpp:308] Batch 24, accuracy_class = 0.825
    I1112 15:22:40.554741 16332 caffe.cpp:308] Batch 24, loss_class = 1.03997
    I1112 15:22:40.837554 16332 caffe.cpp:308] Batch 25, accuracy_class = 0.816667
    I1112 15:22:40.837571 16332 caffe.cpp:308] Batch 25, loss_class = 0.959863
    I1112 15:22:41.117734 16332 caffe.cpp:308] Batch 26, accuracy_class = 0.891667
    I1112 15:22:41.117750 16332 caffe.cpp:308] Batch 26, loss_class = 0.582547
    I1112 15:22:41.399121 16332 caffe.cpp:308] Batch 27, accuracy_class = 0.825
    I1112 15:22:41.399137 16332 caffe.cpp:308] Batch 27, loss_class = 1.21953
    I1112 15:22:41.681704 16332 caffe.cpp:308] Batch 28, accuracy_class = 0.85
    I1112 15:22:41.681723 16332 caffe.cpp:308] Batch 28, loss_class = 0.869855
    I1112 15:22:41.962853 16332 caffe.cpp:308] Batch 29, accuracy_class = 0.85
    I1112 15:22:41.962872 16332 caffe.cpp:308] Batch 29, loss_class = 0.913842
    I1112 15:22:42.241092 16332 caffe.cpp:308] Batch 30, accuracy_class = 0.791667
    I1112 15:22:42.241109 16332 caffe.cpp:308] Batch 30, loss_class = 0.959414
    I1112 15:22:42.522271 16332 caffe.cpp:308] Batch 31, accuracy_class = 0.8
    I1112 15:22:42.522289 16332 caffe.cpp:308] Batch 31, loss_class = 1.00081
    I1112 15:22:42.805626 16332 caffe.cpp:308] Batch 32, accuracy_class = 0.791667
    I1112 15:22:42.805642 16332 caffe.cpp:308] Batch 32, loss_class = 1.75229
    I1112 15:22:43.088102 16332 caffe.cpp:308] Batch 33, accuracy_class = 0.825
    I1112 15:22:43.088119 16332 caffe.cpp:308] Batch 33, loss_class = 1.26022
    I1112 15:22:43.365797 16332 caffe.cpp:308] Batch 34, accuracy_class = 0.85
    I1112 15:22:43.365813 16332 caffe.cpp:308] Batch 34, loss_class = 0.771386
    I1112 15:22:43.646075 16332 caffe.cpp:308] Batch 35, accuracy_class = 0.841667
    I1112 15:22:43.646090 16332 caffe.cpp:308] Batch 35, loss_class = 1.31771
    I1112 15:22:43.928948 16332 caffe.cpp:308] Batch 36, accuracy_class = 0.866667
    I1112 15:22:43.928966 16332 caffe.cpp:308] Batch 36, loss_class = 0.793337
    I1112 15:22:44.208842 16332 caffe.cpp:308] Batch 37, accuracy_class = 0.9
    I1112 15:22:44.208859 16332 caffe.cpp:308] Batch 37, loss_class = 0.65698
    I1112 15:22:44.487496 16332 caffe.cpp:308] Batch 38, accuracy_class = 0.758333
    I1112 15:22:44.487514 16332 caffe.cpp:308] Batch 38, loss_class = 1.93279
    I1112 15:22:44.768703 16332 caffe.cpp:308] Batch 39, accuracy_class = 0.866667
    I1112 15:22:44.768733 16332 caffe.cpp:308] Batch 39, loss_class = 1.08267
    I1112 15:22:45.048882 16332 caffe.cpp:308] Batch 40, accuracy_class = 0.775
    I1112 15:22:45.048898 16332 caffe.cpp:308] Batch 40, loss_class = 1.32169
    I1112 15:22:45.326495 16332 caffe.cpp:308] Batch 41, accuracy_class = 0.858333
    I1112 15:22:45.326510 16332 caffe.cpp:308] Batch 41, loss_class = 0.900295
    I1112 15:22:45.607226 16332 caffe.cpp:308] Batch 42, accuracy_class = 0.8
    I1112 15:22:45.607244 16332 caffe.cpp:308] Batch 42, loss_class = 1.164
    I1112 15:22:45.888716 16332 caffe.cpp:308] Batch 43, accuracy_class = 0.766667
    I1112 15:22:45.888732 16332 caffe.cpp:308] Batch 43, loss_class = 1.30043
    I1112 15:22:46.167954 16332 caffe.cpp:308] Batch 44, accuracy_class = 0.766667
    I1112 15:22:46.167989 16332 caffe.cpp:308] Batch 44, loss_class = 1.50058
    I1112 15:22:46.445875 16332 caffe.cpp:308] Batch 45, accuracy_class = 0.85
    I1112 15:22:46.445893 16332 caffe.cpp:308] Batch 45, loss_class = 0.943513
    I1112 15:22:46.727983 16332 caffe.cpp:308] Batch 46, accuracy_class = 0.775
    I1112 15:22:46.728003 16332 caffe.cpp:308] Batch 46, loss_class = 1.56576
    I1112 15:22:47.008716 16332 caffe.cpp:308] Batch 47, accuracy_class = 0.783333
    I1112 15:22:47.008733 16332 caffe.cpp:308] Batch 47, loss_class = 0.983287
    I1112 15:22:47.288532 16332 caffe.cpp:308] Batch 48, accuracy_class = 0.766667
    I1112 15:22:47.288547 16332 caffe.cpp:308] Batch 48, loss_class = 1.38496
    I1112 15:22:47.568851 16332 caffe.cpp:308] Batch 49, accuracy_class = 0.758333
    I1112 15:22:47.568884 16332 caffe.cpp:308] Batch 49, loss_class = 1.02143
    I1112 15:22:47.850657 16332 caffe.cpp:308] Batch 50, accuracy_class = 0.8
    I1112 15:22:47.850673 16332 caffe.cpp:308] Batch 50, loss_class = 0.888617
    I1112 15:22:48.131410 16332 caffe.cpp:308] Batch 51, accuracy_class = 0.85
    I1112 15:22:48.131428 16332 caffe.cpp:308] Batch 51, loss_class = 0.91793
    I1112 15:22:48.410207 16332 caffe.cpp:308] Batch 52, accuracy_class = 0.775
    I1112 15:22:48.410223 16332 caffe.cpp:308] Batch 52, loss_class = 1.58806
    I1112 15:22:48.694483 16332 caffe.cpp:308] Batch 53, accuracy_class = 0.8
    I1112 15:22:48.694504 16332 caffe.cpp:308] Batch 53, loss_class = 1.33932
    I1112 15:22:48.977674 16332 caffe.cpp:308] Batch 54, accuracy_class = 0.758333
    I1112 15:22:48.977690 16332 caffe.cpp:308] Batch 54, loss_class = 1.16486
    I1112 15:22:49.255931 16332 caffe.cpp:308] Batch 55, accuracy_class = 0.85
    I1112 15:22:49.255946 16332 caffe.cpp:308] Batch 55, loss_class = 0.724734
    I1112 15:22:49.533507 16332 caffe.cpp:308] Batch 56, accuracy_class = 0.775
    I1112 15:22:49.533524 16332 caffe.cpp:308] Batch 56, loss_class = 1.33469
    I1112 15:22:49.816983 16332 caffe.cpp:308] Batch 57, accuracy_class = 0.8
    I1112 15:22:49.816999 16332 caffe.cpp:308] Batch 57, loss_class = 1.22494
    I1112 15:22:50.097530 16332 caffe.cpp:308] Batch 58, accuracy_class = 0.8
    I1112 15:22:50.097550 16332 caffe.cpp:308] Batch 58, loss_class = 0.941001
    I1112 15:22:50.377605 16332 caffe.cpp:308] Batch 59, accuracy_class = 0.816667
    I1112 15:22:50.377624 16332 caffe.cpp:308] Batch 59, loss_class = 1.15388
    I1112 15:22:51.095083 16332 caffe.cpp:308] Batch 60, accuracy_class = 0.858333
    I1112 15:22:51.095100 16332 caffe.cpp:308] Batch 60, loss_class = 0.76152
    I1112 15:22:51.373679 16332 caffe.cpp:308] Batch 61, accuracy_class = 0.833333
    I1112 15:22:51.373695 16332 caffe.cpp:308] Batch 61, loss_class = 0.826445
    I1112 15:22:51.654161 16332 caffe.cpp:308] Batch 62, accuracy_class = 0.8
    I1112 15:22:51.654177 16332 caffe.cpp:308] Batch 62, loss_class = 1.03875
    I1112 15:22:51.936936 16332 caffe.cpp:308] Batch 63, accuracy_class = 0.816667
    I1112 15:22:51.936954 16332 caffe.cpp:308] Batch 63, loss_class = 1.04272
    I1112 15:22:52.216290 16332 caffe.cpp:308] Batch 64, accuracy_class = 0.841667
    I1112 15:22:52.216307 16332 caffe.cpp:308] Batch 64, loss_class = 1.14925
    I1112 15:22:52.495467 16332 caffe.cpp:308] Batch 65, accuracy_class = 0.775
    I1112 15:22:52.495484 16332 caffe.cpp:308] Batch 65, loss_class = 1.32489
    I1112 15:22:52.781129 16332 caffe.cpp:308] Batch 66, accuracy_class = 0.816667
    I1112 15:22:52.781147 16332 caffe.cpp:308] Batch 66, loss_class = 0.937469
    I1112 15:22:53.062258 16332 caffe.cpp:308] Batch 67, accuracy_class = 0.841667
    I1112 15:22:53.062275 16332 caffe.cpp:308] Batch 67, loss_class = 0.913312
    I1112 15:22:53.343931 16332 caffe.cpp:308] Batch 68, accuracy_class = 0.808333
    I1112 15:22:53.343948 16332 caffe.cpp:308] Batch 68, loss_class = 0.938669
    I1112 15:22:53.623946 16332 caffe.cpp:308] Batch 69, accuracy_class = 0.883333
    I1112 15:22:53.623965 16332 caffe.cpp:308] Batch 69, loss_class = 0.658566


## The model achieved near 87.91% accuracy
The above is purely test/validation database that is not used for training.


```python
!jupyter nbconvert --to markdown dvia-train-32x32.ipynb
```

    [NbConvertApp] Converting notebook dvia-train-32x32.ipynb to markdown
    [NbConvertApp] Writing 1531721 bytes to dvia-train-32x32.md



```python

```
