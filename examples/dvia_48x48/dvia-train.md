
# cifar-100 conv net with Caffe for DVIA

Based on guillaume-chevalier's implementation of NiN for Cifar-100 imageset. This is a variation of the same database scaled _up_ to 48x48 (DVIA images at 48x48 full resolution).

It is based on the NIN (Network In Network) architecture detailed in this paper: http://arxiv.org/pdf/1312.4400v3.pdf. 

https://github.com/guillaume-chevalier/python-caffe-custom-cifar-100-conv-net for original implementation.

## Convert the cifar-100 dataset to Caffe's HDF5 format
This step converts previously downloaded Cifar-100 database to 48x48 HDF5 DB.


```python
%%time

!ipython convert-cifar-100.ipy
```

    Converting...
    Conversion was already done. Did not convert twice.
    
    CPU times: user 32 ms, sys: 20 ms, total: 52 ms
    Wall time: 2.93 s


## Build the model with Caffe. 


```python
import numpy as np
import os, sys

scriptpath = os.path.dirname(os.path.realpath( "__file__" ))
caffe_root  = os.path.sep.join(scriptpath.split(os.path.sep)[:-2])
#caffe_root = os.path.join(os.environ['HOME'], 'Projects', 'dvcaffe')
db_root     = os.path.join(os.environ['HOME'], 'Projects', 'IMAGES', 'dvia', 'png.48x48')

import caffe
from caffe import layers as L
from caffe import params as P


print "caffe_root = {}".format(caffe_root)
print "db_root = {}".format(db_root)
```

    /usr/local/caffe/python/caffe/pycaffe.py:13: RuntimeWarning: to-Python converter for boost::shared_ptr<caffe::Net<float> > already registered; second conversion method ignored.
      from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \
    /usr/local/caffe/python/caffe/pycaffe.py:13: RuntimeWarning: to-Python converter for boost::shared_ptr<caffe::Blob<float> > already registered; second conversion method ignored.
      from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \
    /usr/local/caffe/python/caffe/pycaffe.py:13: RuntimeWarning: to-Python converter for boost::shared_ptr<caffe::Solver<float> > already registered; second conversion method ignored.
      from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \


    caffe_root = /home/maheriya/Projects/dvcaffe
    db_root = /home/maheriya/Projects/IMAGES/dvia/png.48x48



```python
def cnn(lmdb, batch_size):
    n = caffe.NetSpec()
    ## Input LMDB data layer
    n.data, n.label = L.Data(batch_size=batch_size, source=lmdb, backend=P.Data.LMDB, 
                             transform_param=dict(scale=1./256), ntop=2)
    
    n.conv1 = L.Convolution(n.data, kernel_size=5, stride=2, num_output=64, weight_filler=dict(type='xavier'))
    n.cccp1a = L.Convolution(n.conv1, kernel_size=1, num_output=42, weight_filler=dict(type='xavier'))
    n.relu1a = L.ReLU(n.cccp1a, in_place=True)
    n.cccp1b = L.Convolution(n.relu1a, kernel_size=1, num_output=32, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.cccp1b, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.drop1 = L.Dropout(n.pool1, in_place=True)
    n.relu1b = L.ReLU(n.drop1, in_place=True)
    
    n.conv2 = L.Convolution(n.relu1b, kernel_size=3, num_output=64, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.drop2 = L.Dropout(n.pool2, in_place=True)
    n.relu2 = L.ReLU(n.drop2, in_place=True)
    
    n.conv3 = L.Convolution(n.relu2, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.AVE)
    n.relu3 = L.ReLU(n.pool3, in_place=True)
    
    n.fc1 = L.InnerProduct(n.relu3, num_output=500, weight_filler=dict(type='xavier'))
    n.relu4 = L.ReLU(n.fc1, in_place=True)
    
    # Output 4-class classifier
    n.fc_class = L.InnerProduct(n.relu4, num_output=4, weight_filler=dict(type='xavier'))
    n.accuracy_class = L.Accuracy(n.fc_class, n.label)
    n.loss_c = L.SoftmaxWithLoss(n.fc_class, n.label)
    
    return n.to_proto()
    
with open('dvia_train.prototxt', 'w') as f:
    trn_lmdb = os.path.join(db_root, 'data/dvia_48x48/dvia_trn_lmdb')
    f.write(str(cnn(trn_lmdb, 100)))
    
with open('dvia_test.prototxt', 'w') as f:
    val_lmdb = os.path.join(db_root, 'data/dvia_48x48/dvia_val_lmdb')
    f.write(str(cnn(val_lmdb, 120)))

!python /usr/local/caffe/python/draw_net.py dvia_train.prototxt dvia_nin.png
```

    /usr/local/caffe/python/caffe/pycaffe.py:13: RuntimeWarning: to-Python converter for boost::shared_ptr<caffe::Net<float> > already registered; second conversion method ignored.
      from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \
    /usr/local/caffe/python/caffe/pycaffe.py:13: RuntimeWarning: to-Python converter for boost::shared_ptr<caffe::Blob<float> > already registered; second conversion method ignored.
      from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \
    /usr/local/caffe/python/caffe/pycaffe.py:13: RuntimeWarning: to-Python converter for boost::shared_ptr<caffe::Solver<float> > already registered; second conversion method ignored.
      from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \
    Drawing net to dvia_nin.png


## Load and visualise the untrained network's internal structure and shape
The network's structure (graph) visualisation tool of caffe is broken in the current release. We will simply print here the data shapes. 


```python
caffe.set_mode_gpu()
solver = caffe.get_solver('dvia_solver.prototxt')
```


```python
print("Layers' features:")
[(k, v.data.shape) for k, v in solver.net.blobs.items()]
```

    Layers' features:





    [('data', (100, 3, 48, 48)),
     ('label', (100,)),
     ('label_data_1_split_0', (100,)),
     ('label_data_1_split_1', (100,)),
     ('conv1', (100, 64, 22, 22)),
     ('cccp1a', (100, 42, 22, 22)),
     ('cccp1b', (100, 32, 22, 22)),
     ('pool1', (100, 32, 11, 11)),
     ('conv2', (100, 64, 9, 9)),
     ('pool2', (100, 64, 4, 4)),
     ('conv3', (100, 96, 2, 2)),
     ('pool3', (100, 96, 1, 1)),
     ('fc1', (100, 500)),
     ('fc_class', (100, 4)),
     ('fc_class_fc_class_0_split_0', (100, 4)),
     ('fc_class_fc_class_0_split_1', (100, 4)),
     ('accuracy_class', ()),
     ('loss_c', ())]




```python
print("Parameters and shape:")
[(k, v[0].data.shape) for k, v in solver.net.params.items()]
```

    Parameters and shape:





    [('conv1', (64, 3, 5, 5)),
     ('cccp1a', (42, 64, 1, 1)),
     ('cccp1b', (32, 42, 1, 1)),
     ('conv2', (64, 32, 3, 3)),
     ('conv3', (96, 64, 3, 3)),
     ('fc1', (500, 96)),
     ('fc_class', (4, 500))]



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
    
    base_lr: 0.0002
    momentum: 0.0
    weight_decay: 0.001
    
    lr_policy: "inv"
    gamma: 0.0001
    power: 0.75
    
    display: 100
    
    max_iter: 150000
    
    snapshot: 50000
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

    I0515 00:20:44.174131 23147 caffe.cpp:185] Using GPUs 0
    I0515 00:20:44.192370 23147 caffe.cpp:190] GPU 0: GeForce GTX 470
    I0515 00:20:44.342478 23147 solver.cpp:48] Initializing solver from parameters: 
    train_net: "dvia_train.prototxt"
    test_net: "dvia_test.prototxt"
    test_iter: 100
    test_interval: 1000
    base_lr: 0.0002
    display: 100
    max_iter: 150000
    lr_policy: "inv"
    gamma: 0.0001
    power: 0.75
    momentum: 0
    weight_decay: 0.001
    snapshot: 50000
    snapshot_prefix: "dvia_train"
    solver_mode: GPU
    device_id: 0
    rms_decay: 0.98
    type: "RMSProp"
    I0515 00:20:44.342741 23147 solver.cpp:81] Creating training net from train_net file: dvia_train.prototxt
    I0515 00:20:44.343603 23147 net.cpp:49] Initializing net from parameters: 
    state {
      phase: TRAIN
    }
    layer {
      name: "data"
      type: "Data"
      top: "data"
      top: "label"
      transform_param {
        scale: 0.00390625
      }
      data_param {
        source: "/home/maheriya/Projects/IMAGES/dvia/png.48x48/data/dvia_48x48/dvia_trn_lmdb"
        batch_size: 100
        backend: LMDB
      }
    }
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 64
        kernel_size: 5
        stride: 2
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "cccp1a"
      type: "Convolution"
      bottom: "conv1"
      top: "cccp1a"
      convolution_param {
        num_output: 42
        kernel_size: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu1a"
      type: "ReLU"
      bottom: "cccp1a"
      top: "cccp1a"
    }
    layer {
      name: "cccp1b"
      type: "Convolution"
      bottom: "cccp1a"
      top: "cccp1b"
      convolution_param {
        num_output: 32
        kernel_size: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool1"
      type: "Pooling"
      bottom: "cccp1b"
      top: "pool1"
      pooling_param {
        pool: MAX
        kernel_size: 3
        stride: 2
      }
    }
    layer {
      name: "drop1"
      type: "Dropout"
      bottom: "pool1"
      top: "pool1"
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
      convolution_param {
        num_output: 64
        kernel_size: 3
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
        kernel_size: 3
        stride: 2
      }
    }
    layer {
      name: "drop2"
      type: "Dropout"
      bottom: "pool2"
      top: "pool2"
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
      convolution_param {
        num_output: 96
        kernel_size: 3
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
        pool: AVE
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
      name: "fc1"
      type: "InnerProduct"
      bottom: "pool3"
      top: "fc1"
      inner_product_param {
        num_output: 500
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu4"
      type: "ReLU"
      bottom: "fc1"
      top: "fc1"
    }
    layer {
      name: "fc_class"
      type: "InnerProduct"
      bottom: "fc1"
      top: "fc_class"
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
      name: "loss_c"
      type: "SoftmaxWithLoss"
      bottom: "fc_class"
      bottom: "label"
      top: "loss_c"
    }
    I0515 00:20:44.344199 23147 layer_factory.hpp:77] Creating layer data
    I0515 00:20:44.344691 23147 net.cpp:91] Creating Layer data
    I0515 00:20:44.344712 23147 net.cpp:399] data -> data
    I0515 00:20:44.344749 23147 net.cpp:399] data -> label
    I0515 00:20:44.345757 23150 db_lmdb.cpp:38] Opened lmdb /home/maheriya/Projects/IMAGES/dvia/png.48x48/data/dvia_48x48/dvia_trn_lmdb
    I0515 00:20:44.365716 23147 data_layer.cpp:41] output data size: 100,3,48,48
    I0515 00:20:44.374152 23147 net.cpp:141] Setting up data
    I0515 00:20:44.374243 23147 net.cpp:148] Top shape: 100 3 48 48 (691200)
    I0515 00:20:44.374258 23147 net.cpp:148] Top shape: 100 (100)
    I0515 00:20:44.374310 23147 net.cpp:156] Memory required for data: 2765200
    I0515 00:20:44.374335 23147 layer_factory.hpp:77] Creating layer label_data_1_split
    I0515 00:20:44.374362 23147 net.cpp:91] Creating Layer label_data_1_split
    I0515 00:20:44.374377 23147 net.cpp:425] label_data_1_split <- label
    I0515 00:20:44.374405 23147 net.cpp:399] label_data_1_split -> label_data_1_split_0
    I0515 00:20:44.374426 23147 net.cpp:399] label_data_1_split -> label_data_1_split_1
    I0515 00:20:44.374488 23147 net.cpp:141] Setting up label_data_1_split
    I0515 00:20:44.374502 23147 net.cpp:148] Top shape: 100 (100)
    I0515 00:20:44.374516 23147 net.cpp:148] Top shape: 100 (100)
    I0515 00:20:44.374526 23147 net.cpp:156] Memory required for data: 2766000
    I0515 00:20:44.374536 23147 layer_factory.hpp:77] Creating layer conv1
    I0515 00:20:44.374563 23147 net.cpp:91] Creating Layer conv1
    I0515 00:20:44.374575 23147 net.cpp:425] conv1 <- data
    I0515 00:20:44.374589 23147 net.cpp:399] conv1 -> conv1
    I0515 00:20:44.376025 23147 net.cpp:141] Setting up conv1
    I0515 00:20:44.376051 23147 net.cpp:148] Top shape: 100 64 22 22 (3097600)
    I0515 00:20:44.376062 23147 net.cpp:156] Memory required for data: 15156400
    I0515 00:20:44.376085 23147 layer_factory.hpp:77] Creating layer cccp1a
    I0515 00:20:44.376107 23147 net.cpp:91] Creating Layer cccp1a
    I0515 00:20:44.376118 23147 net.cpp:425] cccp1a <- conv1
    I0515 00:20:44.376135 23147 net.cpp:399] cccp1a -> cccp1a
    I0515 00:20:44.377779 23147 net.cpp:141] Setting up cccp1a
    I0515 00:20:44.377800 23147 net.cpp:148] Top shape: 100 42 22 22 (2032800)
    I0515 00:20:44.377812 23147 net.cpp:156] Memory required for data: 23287600
    I0515 00:20:44.377831 23147 layer_factory.hpp:77] Creating layer relu1a
    I0515 00:20:44.377849 23147 net.cpp:91] Creating Layer relu1a
    I0515 00:20:44.377861 23147 net.cpp:425] relu1a <- cccp1a
    I0515 00:20:44.377874 23147 net.cpp:386] relu1a -> cccp1a (in-place)
    I0515 00:20:44.377890 23147 net.cpp:141] Setting up relu1a
    I0515 00:20:44.377902 23147 net.cpp:148] Top shape: 100 42 22 22 (2032800)
    I0515 00:20:44.377913 23147 net.cpp:156] Memory required for data: 31418800
    I0515 00:20:44.377923 23147 layer_factory.hpp:77] Creating layer cccp1b
    I0515 00:20:44.377941 23147 net.cpp:91] Creating Layer cccp1b
    I0515 00:20:44.377953 23147 net.cpp:425] cccp1b <- cccp1a
    I0515 00:20:44.377965 23147 net.cpp:399] cccp1b -> cccp1b
    I0515 00:20:44.378250 23147 net.cpp:141] Setting up cccp1b
    I0515 00:20:44.378265 23147 net.cpp:148] Top shape: 100 32 22 22 (1548800)
    I0515 00:20:44.378275 23147 net.cpp:156] Memory required for data: 37614000
    I0515 00:20:44.378293 23147 layer_factory.hpp:77] Creating layer pool1
    I0515 00:20:44.378311 23147 net.cpp:91] Creating Layer pool1
    I0515 00:20:44.378322 23147 net.cpp:425] pool1 <- cccp1b
    I0515 00:20:44.378336 23147 net.cpp:399] pool1 -> pool1
    I0515 00:20:44.378399 23147 net.cpp:141] Setting up pool1
    I0515 00:20:44.378413 23147 net.cpp:148] Top shape: 100 32 11 11 (387200)
    I0515 00:20:44.378423 23147 net.cpp:156] Memory required for data: 39162800
    I0515 00:20:44.378434 23147 layer_factory.hpp:77] Creating layer drop1
    I0515 00:20:44.378448 23147 net.cpp:91] Creating Layer drop1
    I0515 00:20:44.378458 23147 net.cpp:425] drop1 <- pool1
    I0515 00:20:44.378473 23147 net.cpp:386] drop1 -> pool1 (in-place)
    I0515 00:20:44.378505 23147 net.cpp:141] Setting up drop1
    I0515 00:20:44.378520 23147 net.cpp:148] Top shape: 100 32 11 11 (387200)
    I0515 00:20:44.378530 23147 net.cpp:156] Memory required for data: 40711600
    I0515 00:20:44.378540 23147 layer_factory.hpp:77] Creating layer relu1b
    I0515 00:20:44.378557 23147 net.cpp:91] Creating Layer relu1b
    I0515 00:20:44.378568 23147 net.cpp:425] relu1b <- pool1
    I0515 00:20:44.378582 23147 net.cpp:386] relu1b -> pool1 (in-place)
    I0515 00:20:44.378597 23147 net.cpp:141] Setting up relu1b
    I0515 00:20:44.378610 23147 net.cpp:148] Top shape: 100 32 11 11 (387200)
    I0515 00:20:44.378620 23147 net.cpp:156] Memory required for data: 42260400
    I0515 00:20:44.378631 23147 layer_factory.hpp:77] Creating layer conv2
    I0515 00:20:44.378646 23147 net.cpp:91] Creating Layer conv2
    I0515 00:20:44.378684 23147 net.cpp:425] conv2 <- pool1
    I0515 00:20:44.378700 23147 net.cpp:399] conv2 -> conv2
    I0515 00:20:44.379663 23147 net.cpp:141] Setting up conv2
    I0515 00:20:44.379681 23147 net.cpp:148] Top shape: 100 64 9 9 (518400)
    I0515 00:20:44.379693 23147 net.cpp:156] Memory required for data: 44334000
    I0515 00:20:44.379706 23147 layer_factory.hpp:77] Creating layer pool2
    I0515 00:20:44.379724 23147 net.cpp:91] Creating Layer pool2
    I0515 00:20:44.379735 23147 net.cpp:425] pool2 <- conv2
    I0515 00:20:44.379750 23147 net.cpp:399] pool2 -> pool2
    I0515 00:20:44.379802 23147 net.cpp:141] Setting up pool2
    I0515 00:20:44.379817 23147 net.cpp:148] Top shape: 100 64 4 4 (102400)
    I0515 00:20:44.379827 23147 net.cpp:156] Memory required for data: 44743600
    I0515 00:20:44.379837 23147 layer_factory.hpp:77] Creating layer drop2
    I0515 00:20:44.379853 23147 net.cpp:91] Creating Layer drop2
    I0515 00:20:44.379863 23147 net.cpp:425] drop2 <- pool2
    I0515 00:20:44.379876 23147 net.cpp:386] drop2 -> pool2 (in-place)
    I0515 00:20:44.379905 23147 net.cpp:141] Setting up drop2
    I0515 00:20:44.379921 23147 net.cpp:148] Top shape: 100 64 4 4 (102400)
    I0515 00:20:44.379932 23147 net.cpp:156] Memory required for data: 45153200
    I0515 00:20:44.379945 23147 layer_factory.hpp:77] Creating layer relu2
    I0515 00:20:44.379958 23147 net.cpp:91] Creating Layer relu2
    I0515 00:20:44.379969 23147 net.cpp:425] relu2 <- pool2
    I0515 00:20:44.379983 23147 net.cpp:386] relu2 -> pool2 (in-place)
    I0515 00:20:44.379998 23147 net.cpp:141] Setting up relu2
    I0515 00:20:44.380010 23147 net.cpp:148] Top shape: 100 64 4 4 (102400)
    I0515 00:20:44.380023 23147 net.cpp:156] Memory required for data: 45562800
    I0515 00:20:44.380033 23147 layer_factory.hpp:77] Creating layer conv3
    I0515 00:20:44.380048 23147 net.cpp:91] Creating Layer conv3
    I0515 00:20:44.380059 23147 net.cpp:425] conv3 <- pool2
    I0515 00:20:44.380074 23147 net.cpp:399] conv3 -> conv3
    I0515 00:20:44.380769 23147 net.cpp:141] Setting up conv3
    I0515 00:20:44.380784 23147 net.cpp:148] Top shape: 100 96 2 2 (38400)
    I0515 00:20:44.380795 23147 net.cpp:156] Memory required for data: 45716400
    I0515 00:20:44.380811 23147 layer_factory.hpp:77] Creating layer pool3
    I0515 00:20:44.380825 23147 net.cpp:91] Creating Layer pool3
    I0515 00:20:44.380836 23147 net.cpp:425] pool3 <- conv3
    I0515 00:20:44.380851 23147 net.cpp:399] pool3 -> pool3
    I0515 00:20:44.380885 23147 net.cpp:141] Setting up pool3
    I0515 00:20:44.380900 23147 net.cpp:148] Top shape: 100 96 1 1 (9600)
    I0515 00:20:44.380911 23147 net.cpp:156] Memory required for data: 45754800
    I0515 00:20:44.380921 23147 layer_factory.hpp:77] Creating layer relu3
    I0515 00:20:44.380935 23147 net.cpp:91] Creating Layer relu3
    I0515 00:20:44.380946 23147 net.cpp:425] relu3 <- pool3
    I0515 00:20:44.380959 23147 net.cpp:386] relu3 -> pool3 (in-place)
    I0515 00:20:44.380973 23147 net.cpp:141] Setting up relu3
    I0515 00:20:44.380986 23147 net.cpp:148] Top shape: 100 96 1 1 (9600)
    I0515 00:20:44.380996 23147 net.cpp:156] Memory required for data: 45793200
    I0515 00:20:44.381007 23147 layer_factory.hpp:77] Creating layer fc1
    I0515 00:20:44.381027 23147 net.cpp:91] Creating Layer fc1
    I0515 00:20:44.381038 23147 net.cpp:425] fc1 <- pool3
    I0515 00:20:44.381052 23147 net.cpp:399] fc1 -> fc1
    I0515 00:20:44.381546 23147 net.cpp:141] Setting up fc1
    I0515 00:20:44.381561 23147 net.cpp:148] Top shape: 100 500 (50000)
    I0515 00:20:44.381572 23147 net.cpp:156] Memory required for data: 45993200
    I0515 00:20:44.381587 23147 layer_factory.hpp:77] Creating layer relu4
    I0515 00:20:44.381603 23147 net.cpp:91] Creating Layer relu4
    I0515 00:20:44.381613 23147 net.cpp:425] relu4 <- fc1
    I0515 00:20:44.381628 23147 net.cpp:386] relu4 -> fc1 (in-place)
    I0515 00:20:44.381642 23147 net.cpp:141] Setting up relu4
    I0515 00:20:44.381655 23147 net.cpp:148] Top shape: 100 500 (50000)
    I0515 00:20:44.381665 23147 net.cpp:156] Memory required for data: 46193200
    I0515 00:20:44.381676 23147 layer_factory.hpp:77] Creating layer fc_class
    I0515 00:20:44.381690 23147 net.cpp:91] Creating Layer fc_class
    I0515 00:20:44.381700 23147 net.cpp:425] fc_class <- fc1
    I0515 00:20:44.381731 23147 net.cpp:399] fc_class -> fc_class
    I0515 00:20:44.381870 23147 net.cpp:141] Setting up fc_class
    I0515 00:20:44.381885 23147 net.cpp:148] Top shape: 100 4 (400)
    I0515 00:20:44.381896 23147 net.cpp:156] Memory required for data: 46194800
    I0515 00:20:44.381909 23147 layer_factory.hpp:77] Creating layer fc_class_fc_class_0_split
    I0515 00:20:44.381925 23147 net.cpp:91] Creating Layer fc_class_fc_class_0_split
    I0515 00:20:44.381937 23147 net.cpp:425] fc_class_fc_class_0_split <- fc_class
    I0515 00:20:44.381950 23147 net.cpp:399] fc_class_fc_class_0_split -> fc_class_fc_class_0_split_0
    I0515 00:20:44.381965 23147 net.cpp:399] fc_class_fc_class_0_split -> fc_class_fc_class_0_split_1
    I0515 00:20:44.382014 23147 net.cpp:141] Setting up fc_class_fc_class_0_split
    I0515 00:20:44.382028 23147 net.cpp:148] Top shape: 100 4 (400)
    I0515 00:20:44.382040 23147 net.cpp:148] Top shape: 100 4 (400)
    I0515 00:20:44.382052 23147 net.cpp:156] Memory required for data: 46198000
    I0515 00:20:44.382064 23147 layer_factory.hpp:77] Creating layer accuracy_class
    I0515 00:20:44.382081 23147 net.cpp:91] Creating Layer accuracy_class
    I0515 00:20:44.382091 23147 net.cpp:425] accuracy_class <- fc_class_fc_class_0_split_0
    I0515 00:20:44.382103 23147 net.cpp:425] accuracy_class <- label_data_1_split_0
    I0515 00:20:44.382117 23147 net.cpp:399] accuracy_class -> accuracy_class
    I0515 00:20:44.382134 23147 net.cpp:141] Setting up accuracy_class
    I0515 00:20:44.382148 23147 net.cpp:148] Top shape: (1)
    I0515 00:20:44.382158 23147 net.cpp:156] Memory required for data: 46198004
    I0515 00:20:44.382169 23147 layer_factory.hpp:77] Creating layer loss_c
    I0515 00:20:44.382185 23147 net.cpp:91] Creating Layer loss_c
    I0515 00:20:44.382197 23147 net.cpp:425] loss_c <- fc_class_fc_class_0_split_1
    I0515 00:20:44.382210 23147 net.cpp:425] loss_c <- label_data_1_split_1
    I0515 00:20:44.382222 23147 net.cpp:399] loss_c -> loss_c
    I0515 00:20:44.382241 23147 layer_factory.hpp:77] Creating layer loss_c
    I0515 00:20:44.382365 23147 net.cpp:141] Setting up loss_c
    I0515 00:20:44.382380 23147 net.cpp:148] Top shape: (1)
    I0515 00:20:44.382390 23147 net.cpp:151]     with loss weight 1
    I0515 00:20:44.382433 23147 net.cpp:156] Memory required for data: 46198008
    I0515 00:20:44.382444 23147 net.cpp:217] loss_c needs backward computation.
    I0515 00:20:44.382455 23147 net.cpp:219] accuracy_class does not need backward computation.
    I0515 00:20:44.382467 23147 net.cpp:217] fc_class_fc_class_0_split needs backward computation.
    I0515 00:20:44.382478 23147 net.cpp:217] fc_class needs backward computation.
    I0515 00:20:44.382488 23147 net.cpp:217] relu4 needs backward computation.
    I0515 00:20:44.382499 23147 net.cpp:217] fc1 needs backward computation.
    I0515 00:20:44.382509 23147 net.cpp:217] relu3 needs backward computation.
    I0515 00:20:44.382520 23147 net.cpp:217] pool3 needs backward computation.
    I0515 00:20:44.382531 23147 net.cpp:217] conv3 needs backward computation.
    I0515 00:20:44.382546 23147 net.cpp:217] relu2 needs backward computation.
    I0515 00:20:44.382557 23147 net.cpp:217] drop2 needs backward computation.
    I0515 00:20:44.382567 23147 net.cpp:217] pool2 needs backward computation.
    I0515 00:20:44.382578 23147 net.cpp:217] conv2 needs backward computation.
    I0515 00:20:44.382589 23147 net.cpp:217] relu1b needs backward computation.
    I0515 00:20:44.382599 23147 net.cpp:217] drop1 needs backward computation.
    I0515 00:20:44.382611 23147 net.cpp:217] pool1 needs backward computation.
    I0515 00:20:44.382621 23147 net.cpp:217] cccp1b needs backward computation.
    I0515 00:20:44.382632 23147 net.cpp:217] relu1a needs backward computation.
    I0515 00:20:44.382642 23147 net.cpp:217] cccp1a needs backward computation.
    I0515 00:20:44.382653 23147 net.cpp:217] conv1 needs backward computation.
    I0515 00:20:44.382664 23147 net.cpp:219] label_data_1_split does not need backward computation.
    I0515 00:20:44.382676 23147 net.cpp:219] data does not need backward computation.
    I0515 00:20:44.382686 23147 net.cpp:261] This network produces output accuracy_class
    I0515 00:20:44.382710 23147 net.cpp:261] This network produces output loss_c
    I0515 00:20:44.382738 23147 net.cpp:274] Network initialization done.
    I0515 00:20:44.383363 23147 solver.cpp:181] Creating test net (#0) specified by test_net file: dvia_test.prototxt
    I0515 00:20:44.383667 23147 net.cpp:49] Initializing net from parameters: 
    state {
      phase: TEST
    }
    layer {
      name: "data"
      type: "Data"
      top: "data"
      top: "label"
      transform_param {
        scale: 0.00390625
      }
      data_param {
        source: "/home/maheriya/Projects/IMAGES/dvia/png.48x48/data/dvia_48x48/dvia_val_lmdb"
        batch_size: 120
        backend: LMDB
      }
    }
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 64
        kernel_size: 5
        stride: 2
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "cccp1a"
      type: "Convolution"
      bottom: "conv1"
      top: "cccp1a"
      convolution_param {
        num_output: 42
        kernel_size: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu1a"
      type: "ReLU"
      bottom: "cccp1a"
      top: "cccp1a"
    }
    layer {
      name: "cccp1b"
      type: "Convolution"
      bottom: "cccp1a"
      top: "cccp1b"
      convolution_param {
        num_output: 32
        kernel_size: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool1"
      type: "Pooling"
      bottom: "cccp1b"
      top: "pool1"
      pooling_param {
        pool: MAX
        kernel_size: 3
        stride: 2
      }
    }
    layer {
      name: "drop1"
      type: "Dropout"
      bottom: "pool1"
      top: "pool1"
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
      convolution_param {
        num_output: 64
        kernel_size: 3
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
        kernel_size: 3
        stride: 2
      }
    }
    layer {
      name: "drop2"
      type: "Dropout"
      bottom: "pool2"
      top: "pool2"
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
      convolution_param {
        num_output: 96
        kernel_size: 3
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
        pool: AVE
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
      name: "fc1"
      type: "InnerProduct"
      bottom: "pool3"
      top: "fc1"
      inner_product_param {
        num_output: 500
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu4"
      type: "ReLU"
      bottom: "fc1"
      top: "fc1"
    }
    layer {
      name: "fc_class"
      type: "InnerProduct"
      bottom: "fc1"
      top: "fc_class"
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
      name: "loss_c"
      type: "SoftmaxWithLoss"
      bottom: "fc_class"
      bottom: "label"
      top: "loss_c"
    }
    I0515 00:20:44.384235 23147 layer_factory.hpp:77] Creating layer data
    I0515 00:20:44.384434 23147 net.cpp:91] Creating Layer data
    I0515 00:20:44.384451 23147 net.cpp:399] data -> data
    I0515 00:20:44.384471 23147 net.cpp:399] data -> label
    I0515 00:20:44.385191 23152 db_lmdb.cpp:38] Opened lmdb /home/maheriya/Projects/IMAGES/dvia/png.48x48/data/dvia_48x48/dvia_val_lmdb
    I0515 00:20:44.385555 23147 data_layer.cpp:41] output data size: 120,3,48,48
    I0515 00:20:44.399152 23147 net.cpp:141] Setting up data
    I0515 00:20:44.399214 23147 net.cpp:148] Top shape: 120 3 48 48 (829440)
    I0515 00:20:44.399227 23147 net.cpp:148] Top shape: 120 (120)
    I0515 00:20:44.399238 23147 net.cpp:156] Memory required for data: 3318240
    I0515 00:20:44.399253 23147 layer_factory.hpp:77] Creating layer label_data_1_split
    I0515 00:20:44.399276 23147 net.cpp:91] Creating Layer label_data_1_split
    I0515 00:20:44.399288 23147 net.cpp:425] label_data_1_split <- label
    I0515 00:20:44.399304 23147 net.cpp:399] label_data_1_split -> label_data_1_split_0
    I0515 00:20:44.399361 23147 net.cpp:399] label_data_1_split -> label_data_1_split_1
    I0515 00:20:44.399423 23147 net.cpp:141] Setting up label_data_1_split
    I0515 00:20:44.399438 23147 net.cpp:148] Top shape: 120 (120)
    I0515 00:20:44.399451 23147 net.cpp:148] Top shape: 120 (120)
    I0515 00:20:44.399461 23147 net.cpp:156] Memory required for data: 3319200
    I0515 00:20:44.399471 23147 layer_factory.hpp:77] Creating layer conv1
    I0515 00:20:44.399494 23147 net.cpp:91] Creating Layer conv1
    I0515 00:20:44.399514 23147 net.cpp:425] conv1 <- data
    I0515 00:20:44.399531 23147 net.cpp:399] conv1 -> conv1
    I0515 00:20:44.399878 23147 net.cpp:141] Setting up conv1
    I0515 00:20:44.399896 23147 net.cpp:148] Top shape: 120 64 22 22 (3717120)
    I0515 00:20:44.399906 23147 net.cpp:156] Memory required for data: 18187680
    I0515 00:20:44.399925 23147 layer_factory.hpp:77] Creating layer cccp1a
    I0515 00:20:44.399943 23147 net.cpp:91] Creating Layer cccp1a
    I0515 00:20:44.399955 23147 net.cpp:425] cccp1a <- conv1
    I0515 00:20:44.399969 23147 net.cpp:399] cccp1a -> cccp1a
    I0515 00:20:44.401347 23147 net.cpp:141] Setting up cccp1a
    I0515 00:20:44.401363 23147 net.cpp:148] Top shape: 120 42 22 22 (2439360)
    I0515 00:20:44.401376 23147 net.cpp:156] Memory required for data: 27945120
    I0515 00:20:44.401392 23147 layer_factory.hpp:77] Creating layer relu1a
    I0515 00:20:44.401408 23147 net.cpp:91] Creating Layer relu1a
    I0515 00:20:44.401419 23147 net.cpp:425] relu1a <- cccp1a
    I0515 00:20:44.401432 23147 net.cpp:386] relu1a -> cccp1a (in-place)
    I0515 00:20:44.401448 23147 net.cpp:141] Setting up relu1a
    I0515 00:20:44.401459 23147 net.cpp:148] Top shape: 120 42 22 22 (2439360)
    I0515 00:20:44.401470 23147 net.cpp:156] Memory required for data: 37702560
    I0515 00:20:44.401480 23147 layer_factory.hpp:77] Creating layer cccp1b
    I0515 00:20:44.401499 23147 net.cpp:91] Creating Layer cccp1b
    I0515 00:20:44.401510 23147 net.cpp:425] cccp1b <- cccp1a
    I0515 00:20:44.401526 23147 net.cpp:399] cccp1b -> cccp1b
    I0515 00:20:44.401821 23147 net.cpp:141] Setting up cccp1b
    I0515 00:20:44.401837 23147 net.cpp:148] Top shape: 120 32 22 22 (1858560)
    I0515 00:20:44.401849 23147 net.cpp:156] Memory required for data: 45136800
    I0515 00:20:44.401867 23147 layer_factory.hpp:77] Creating layer pool1
    I0515 00:20:44.401883 23147 net.cpp:91] Creating Layer pool1
    I0515 00:20:44.401895 23147 net.cpp:425] pool1 <- cccp1b
    I0515 00:20:44.401911 23147 net.cpp:399] pool1 -> pool1
    I0515 00:20:44.401962 23147 net.cpp:141] Setting up pool1
    I0515 00:20:44.401975 23147 net.cpp:148] Top shape: 120 32 11 11 (464640)
    I0515 00:20:44.401986 23147 net.cpp:156] Memory required for data: 46995360
    I0515 00:20:44.401998 23147 layer_factory.hpp:77] Creating layer drop1
    I0515 00:20:44.402014 23147 net.cpp:91] Creating Layer drop1
    I0515 00:20:44.402024 23147 net.cpp:425] drop1 <- pool1
    I0515 00:20:44.402036 23147 net.cpp:386] drop1 -> pool1 (in-place)
    I0515 00:20:44.402070 23147 net.cpp:141] Setting up drop1
    I0515 00:20:44.402083 23147 net.cpp:148] Top shape: 120 32 11 11 (464640)
    I0515 00:20:44.402094 23147 net.cpp:156] Memory required for data: 48853920
    I0515 00:20:44.402106 23147 layer_factory.hpp:77] Creating layer relu1b
    I0515 00:20:44.402119 23147 net.cpp:91] Creating Layer relu1b
    I0515 00:20:44.402130 23147 net.cpp:425] relu1b <- pool1
    I0515 00:20:44.402145 23147 net.cpp:386] relu1b -> pool1 (in-place)
    I0515 00:20:44.402159 23147 net.cpp:141] Setting up relu1b
    I0515 00:20:44.402173 23147 net.cpp:148] Top shape: 120 32 11 11 (464640)
    I0515 00:20:44.402184 23147 net.cpp:156] Memory required for data: 50712480
    I0515 00:20:44.402194 23147 layer_factory.hpp:77] Creating layer conv2
    I0515 00:20:44.402209 23147 net.cpp:91] Creating Layer conv2
    I0515 00:20:44.402220 23147 net.cpp:425] conv2 <- pool1
    I0515 00:20:44.402236 23147 net.cpp:399] conv2 -> conv2
    I0515 00:20:44.402709 23147 net.cpp:141] Setting up conv2
    I0515 00:20:44.402726 23147 net.cpp:148] Top shape: 120 64 9 9 (622080)
    I0515 00:20:44.402740 23147 net.cpp:156] Memory required for data: 53200800
    I0515 00:20:44.402770 23147 layer_factory.hpp:77] Creating layer pool2
    I0515 00:20:44.402786 23147 net.cpp:91] Creating Layer pool2
    I0515 00:20:44.402796 23147 net.cpp:425] pool2 <- conv2
    I0515 00:20:44.402812 23147 net.cpp:399] pool2 -> pool2
    I0515 00:20:44.403581 23147 net.cpp:141] Setting up pool2
    I0515 00:20:44.403599 23147 net.cpp:148] Top shape: 120 64 4 4 (122880)
    I0515 00:20:44.403609 23147 net.cpp:156] Memory required for data: 53692320
    I0515 00:20:44.403620 23147 layer_factory.hpp:77] Creating layer drop2
    I0515 00:20:44.403635 23147 net.cpp:91] Creating Layer drop2
    I0515 00:20:44.403646 23147 net.cpp:425] drop2 <- pool2
    I0515 00:20:44.403659 23147 net.cpp:386] drop2 -> pool2 (in-place)
    I0515 00:20:44.403700 23147 net.cpp:141] Setting up drop2
    I0515 00:20:44.403714 23147 net.cpp:148] Top shape: 120 64 4 4 (122880)
    I0515 00:20:44.403725 23147 net.cpp:156] Memory required for data: 54183840
    I0515 00:20:44.403735 23147 layer_factory.hpp:77] Creating layer relu2
    I0515 00:20:44.403748 23147 net.cpp:91] Creating Layer relu2
    I0515 00:20:44.403760 23147 net.cpp:425] relu2 <- pool2
    I0515 00:20:44.403775 23147 net.cpp:386] relu2 -> pool2 (in-place)
    I0515 00:20:44.403791 23147 net.cpp:141] Setting up relu2
    I0515 00:20:44.403805 23147 net.cpp:148] Top shape: 120 64 4 4 (122880)
    I0515 00:20:44.403828 23147 net.cpp:156] Memory required for data: 54675360
    I0515 00:20:44.403839 23147 layer_factory.hpp:77] Creating layer conv3
    I0515 00:20:44.403856 23147 net.cpp:91] Creating Layer conv3
    I0515 00:20:44.403868 23147 net.cpp:425] conv3 <- pool2
    I0515 00:20:44.403882 23147 net.cpp:399] conv3 -> conv3
    I0515 00:20:44.404659 23147 net.cpp:141] Setting up conv3
    I0515 00:20:44.404686 23147 net.cpp:148] Top shape: 120 96 2 2 (46080)
    I0515 00:20:44.404698 23147 net.cpp:156] Memory required for data: 54859680
    I0515 00:20:44.404716 23147 layer_factory.hpp:77] Creating layer pool3
    I0515 00:20:44.404732 23147 net.cpp:91] Creating Layer pool3
    I0515 00:20:44.404743 23147 net.cpp:425] pool3 <- conv3
    I0515 00:20:44.404757 23147 net.cpp:399] pool3 -> pool3
    I0515 00:20:44.404790 23147 net.cpp:141] Setting up pool3
    I0515 00:20:44.404804 23147 net.cpp:148] Top shape: 120 96 1 1 (11520)
    I0515 00:20:44.404824 23147 net.cpp:156] Memory required for data: 54905760
    I0515 00:20:44.404834 23147 layer_factory.hpp:77] Creating layer relu3
    I0515 00:20:44.404846 23147 net.cpp:91] Creating Layer relu3
    I0515 00:20:44.404858 23147 net.cpp:425] relu3 <- pool3
    I0515 00:20:44.404873 23147 net.cpp:386] relu3 -> pool3 (in-place)
    I0515 00:20:44.404887 23147 net.cpp:141] Setting up relu3
    I0515 00:20:44.404899 23147 net.cpp:148] Top shape: 120 96 1 1 (11520)
    I0515 00:20:44.404911 23147 net.cpp:156] Memory required for data: 54951840
    I0515 00:20:44.404920 23147 layer_factory.hpp:77] Creating layer fc1
    I0515 00:20:44.404937 23147 net.cpp:91] Creating Layer fc1
    I0515 00:20:44.404947 23147 net.cpp:425] fc1 <- pool3
    I0515 00:20:44.404975 23147 net.cpp:399] fc1 -> fc1
    I0515 00:20:44.405517 23147 net.cpp:141] Setting up fc1
    I0515 00:20:44.405535 23147 net.cpp:148] Top shape: 120 500 (60000)
    I0515 00:20:44.405563 23147 net.cpp:156] Memory required for data: 55191840
    I0515 00:20:44.405578 23147 layer_factory.hpp:77] Creating layer relu4
    I0515 00:20:44.405594 23147 net.cpp:91] Creating Layer relu4
    I0515 00:20:44.405606 23147 net.cpp:425] relu4 <- fc1
    I0515 00:20:44.405618 23147 net.cpp:386] relu4 -> fc1 (in-place)
    I0515 00:20:44.405632 23147 net.cpp:141] Setting up relu4
    I0515 00:20:44.405644 23147 net.cpp:148] Top shape: 120 500 (60000)
    I0515 00:20:44.405654 23147 net.cpp:156] Memory required for data: 55431840
    I0515 00:20:44.405664 23147 layer_factory.hpp:77] Creating layer fc_class
    I0515 00:20:44.405680 23147 net.cpp:91] Creating Layer fc_class
    I0515 00:20:44.405700 23147 net.cpp:425] fc_class <- fc1
    I0515 00:20:44.405714 23147 net.cpp:399] fc_class -> fc_class
    I0515 00:20:44.405866 23147 net.cpp:141] Setting up fc_class
    I0515 00:20:44.405880 23147 net.cpp:148] Top shape: 120 4 (480)
    I0515 00:20:44.405891 23147 net.cpp:156] Memory required for data: 55433760
    I0515 00:20:44.405905 23147 layer_factory.hpp:77] Creating layer fc_class_fc_class_0_split
    I0515 00:20:44.405936 23147 net.cpp:91] Creating Layer fc_class_fc_class_0_split
    I0515 00:20:44.405946 23147 net.cpp:425] fc_class_fc_class_0_split <- fc_class
    I0515 00:20:44.405962 23147 net.cpp:399] fc_class_fc_class_0_split -> fc_class_fc_class_0_split_0
    I0515 00:20:44.405987 23147 net.cpp:399] fc_class_fc_class_0_split -> fc_class_fc_class_0_split_1
    I0515 00:20:44.406033 23147 net.cpp:141] Setting up fc_class_fc_class_0_split
    I0515 00:20:44.406046 23147 net.cpp:148] Top shape: 120 4 (480)
    I0515 00:20:44.406059 23147 net.cpp:148] Top shape: 120 4 (480)
    I0515 00:20:44.406069 23147 net.cpp:156] Memory required for data: 55437600
    I0515 00:20:44.406080 23147 layer_factory.hpp:77] Creating layer accuracy_class
    I0515 00:20:44.406096 23147 net.cpp:91] Creating Layer accuracy_class
    I0515 00:20:44.406107 23147 net.cpp:425] accuracy_class <- fc_class_fc_class_0_split_0
    I0515 00:20:44.406119 23147 net.cpp:425] accuracy_class <- label_data_1_split_0
    I0515 00:20:44.406146 23147 net.cpp:399] accuracy_class -> accuracy_class
    I0515 00:20:44.406167 23147 net.cpp:141] Setting up accuracy_class
    I0515 00:20:44.406180 23147 net.cpp:148] Top shape: (1)
    I0515 00:20:44.406190 23147 net.cpp:156] Memory required for data: 55437604
    I0515 00:20:44.406201 23147 layer_factory.hpp:77] Creating layer loss_c
    I0515 00:20:44.406214 23147 net.cpp:91] Creating Layer loss_c
    I0515 00:20:44.406225 23147 net.cpp:425] loss_c <- fc_class_fc_class_0_split_1
    I0515 00:20:44.406237 23147 net.cpp:425] loss_c <- label_data_1_split_1
    I0515 00:20:44.406253 23147 net.cpp:399] loss_c -> loss_c
    I0515 00:20:44.406281 23147 layer_factory.hpp:77] Creating layer loss_c
    I0515 00:20:44.406396 23147 net.cpp:141] Setting up loss_c
    I0515 00:20:44.406411 23147 net.cpp:148] Top shape: (1)
    I0515 00:20:44.406431 23147 net.cpp:151]     with loss weight 1
    I0515 00:20:44.406451 23147 net.cpp:156] Memory required for data: 55437608
    I0515 00:20:44.406461 23147 net.cpp:217] loss_c needs backward computation.
    I0515 00:20:44.406472 23147 net.cpp:219] accuracy_class does not need backward computation.
    I0515 00:20:44.406484 23147 net.cpp:217] fc_class_fc_class_0_split needs backward computation.
    I0515 00:20:44.406494 23147 net.cpp:217] fc_class needs backward computation.
    I0515 00:20:44.406505 23147 net.cpp:217] relu4 needs backward computation.
    I0515 00:20:44.406515 23147 net.cpp:217] fc1 needs backward computation.
    I0515 00:20:44.406525 23147 net.cpp:217] relu3 needs backward computation.
    I0515 00:20:44.406535 23147 net.cpp:217] pool3 needs backward computation.
    I0515 00:20:44.406546 23147 net.cpp:217] conv3 needs backward computation.
    I0515 00:20:44.406558 23147 net.cpp:217] relu2 needs backward computation.
    I0515 00:20:44.406575 23147 net.cpp:217] drop2 needs backward computation.
    I0515 00:20:44.406585 23147 net.cpp:217] pool2 needs backward computation.
    I0515 00:20:44.406599 23147 net.cpp:217] conv2 needs backward computation.
    I0515 00:20:44.406610 23147 net.cpp:217] relu1b needs backward computation.
    I0515 00:20:44.406620 23147 net.cpp:217] drop1 needs backward computation.
    I0515 00:20:44.406630 23147 net.cpp:217] pool1 needs backward computation.
    I0515 00:20:44.406641 23147 net.cpp:217] cccp1b needs backward computation.
    I0515 00:20:44.406651 23147 net.cpp:217] relu1a needs backward computation.
    I0515 00:20:44.406661 23147 net.cpp:217] cccp1a needs backward computation.
    I0515 00:20:44.406672 23147 net.cpp:217] conv1 needs backward computation.
    I0515 00:20:44.406683 23147 net.cpp:219] label_data_1_split does not need backward computation.
    I0515 00:20:44.406694 23147 net.cpp:219] data does not need backward computation.
    I0515 00:20:44.406704 23147 net.cpp:261] This network produces output accuracy_class
    I0515 00:20:44.406728 23147 net.cpp:261] This network produces output loss_c
    I0515 00:20:44.406754 23147 net.cpp:274] Network initialization done.
    I0515 00:20:44.406867 23147 solver.cpp:60] Solver scaffolding done.
    I0515 00:20:44.407552 23147 caffe.cpp:129] Finetuning from cifar_pretrained.caffemodel
    I0515 00:20:44.408570 23147 net.cpp:753] Ignoring source layer label_coarse_data_1_split
    I0515 00:20:44.408603 23147 net.cpp:753] Ignoring source layer label_fine_data_2_split
    I0515 00:20:44.408793 23147 net.cpp:753] Ignoring source layer fc1_relu4_0_split
    I0515 00:20:44.408805 23147 net.cpp:753] Ignoring source layer fc_coarse
    I0515 00:20:44.408815 23147 net.cpp:753] Ignoring source layer fc_coarse_fc_coarse_0_split
    I0515 00:20:44.408825 23147 net.cpp:753] Ignoring source layer accuracy_coarse
    I0515 00:20:44.408835 23147 net.cpp:753] Ignoring source layer loss_coarse
    I0515 00:20:44.408844 23147 net.cpp:753] Ignoring source layer fc_fine
    I0515 00:20:44.408854 23147 net.cpp:753] Ignoring source layer fc_fine_fc_fine_0_split
    I0515 00:20:44.408864 23147 net.cpp:753] Ignoring source layer accuracy_fine
    I0515 00:20:44.408874 23147 net.cpp:753] Ignoring source layer loss_f
    I0515 00:20:44.409334 23147 net.cpp:753] Ignoring source layer label_coarse_data_1_split
    I0515 00:20:44.409348 23147 net.cpp:753] Ignoring source layer label_fine_data_2_split
    I0515 00:20:44.409482 23147 net.cpp:753] Ignoring source layer fc1_relu4_0_split
    I0515 00:20:44.409492 23147 net.cpp:753] Ignoring source layer fc_coarse
    I0515 00:20:44.409502 23147 net.cpp:753] Ignoring source layer fc_coarse_fc_coarse_0_split
    I0515 00:20:44.409512 23147 net.cpp:753] Ignoring source layer accuracy_coarse
    I0515 00:20:44.409533 23147 net.cpp:753] Ignoring source layer loss_coarse
    I0515 00:20:44.409543 23147 net.cpp:753] Ignoring source layer fc_fine
    I0515 00:20:44.409554 23147 net.cpp:753] Ignoring source layer fc_fine_fc_fine_0_split
    I0515 00:20:44.409564 23147 net.cpp:753] Ignoring source layer accuracy_fine
    I0515 00:20:44.409574 23147 net.cpp:753] Ignoring source layer loss_f
    I0515 00:20:44.409632 23147 caffe.cpp:219] Starting Optimization
    I0515 00:20:44.409663 23147 solver.cpp:279] Solving 
    I0515 00:20:44.409674 23147 solver.cpp:280] Learning Rate Policy: inv
    I0515 00:20:44.410472 23147 solver.cpp:337] Iteration 0, Testing net (#0)
    I0515 00:20:44.410718 23147 blocking_queue.cpp:50] Data layer prefetch queue empty
    I0515 00:20:48.768152 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.1705
    I0515 00:20:48.768211 23147 solver.cpp:404]     Test net output #1: loss_c = 2.36089 (* 1 = 2.36089 loss)
    I0515 00:20:48.832216 23147 solver.cpp:228] Iteration 0, loss = 3.12112
    I0515 00:20:48.832273 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.18
    I0515 00:20:48.832293 23147 solver.cpp:244]     Train net output #1: loss_c = 3.12112 (* 1 = 3.12112 loss)
    I0515 00:20:48.832329 23147 sgd_solver.cpp:106] Iteration 0, lr = 0.0002
    I0515 00:20:56.989027 23147 solver.cpp:228] Iteration 100, loss = 0.509834
    I0515 00:20:56.989073 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.79
    I0515 00:20:56.989096 23147 solver.cpp:244]     Train net output #1: loss_c = 0.509834 (* 1 = 0.509834 loss)
    I0515 00:20:56.989111 23147 sgd_solver.cpp:106] Iteration 100, lr = 0.000198513
    I0515 00:21:05.114989 23147 solver.cpp:228] Iteration 200, loss = 0.345128
    I0515 00:21:05.115047 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.88
    I0515 00:21:05.115077 23147 solver.cpp:244]     Train net output #1: loss_c = 0.345128 (* 1 = 0.345128 loss)
    I0515 00:21:05.115098 23147 sgd_solver.cpp:106] Iteration 200, lr = 0.000197052
    I0515 00:21:13.229331 23147 solver.cpp:228] Iteration 300, loss = 0.325418
    I0515 00:21:13.229388 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.9
    I0515 00:21:13.229418 23147 solver.cpp:244]     Train net output #1: loss_c = 0.325418 (* 1 = 0.325418 loss)
    I0515 00:21:13.229439 23147 sgd_solver.cpp:106] Iteration 300, lr = 0.000195615
    I0515 00:21:21.324443 23147 solver.cpp:228] Iteration 400, loss = 0.354361
    I0515 00:21:21.324574 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.88
    I0515 00:21:21.324605 23147 solver.cpp:244]     Train net output #1: loss_c = 0.354361 (* 1 = 0.354361 loss)
    I0515 00:21:21.324627 23147 sgd_solver.cpp:106] Iteration 400, lr = 0.000194203
    I0515 00:21:29.418239 23147 solver.cpp:228] Iteration 500, loss = 0.31128
    I0515 00:21:29.418298 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.87
    I0515 00:21:29.418326 23147 solver.cpp:244]     Train net output #1: loss_c = 0.31128 (* 1 = 0.31128 loss)
    I0515 00:21:29.418347 23147 sgd_solver.cpp:106] Iteration 500, lr = 0.000192814
    I0515 00:21:37.507972 23147 solver.cpp:228] Iteration 600, loss = 0.252716
    I0515 00:21:37.508019 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.89
    I0515 00:21:37.508044 23147 solver.cpp:244]     Train net output #1: loss_c = 0.252716 (* 1 = 0.252716 loss)
    I0515 00:21:37.508059 23147 sgd_solver.cpp:106] Iteration 600, lr = 0.000191448
    I0515 00:21:45.649096 23147 solver.cpp:228] Iteration 700, loss = 0.207527
    I0515 00:21:45.649145 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.92
    I0515 00:21:45.649165 23147 solver.cpp:244]     Train net output #1: loss_c = 0.207527 (* 1 = 0.207527 loss)
    I0515 00:21:45.649181 23147 sgd_solver.cpp:106] Iteration 700, lr = 0.000190104
    I0515 00:21:53.792326 23147 solver.cpp:228] Iteration 800, loss = 0.214889
    I0515 00:21:53.792520 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.91
    I0515 00:21:53.792541 23147 solver.cpp:244]     Train net output #1: loss_c = 0.214889 (* 1 = 0.214889 loss)
    I0515 00:21:53.792574 23147 sgd_solver.cpp:106] Iteration 800, lr = 0.000188783
    I0515 00:22:01.935613 23147 solver.cpp:228] Iteration 900, loss = 0.153903
    I0515 00:22:01.935664 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.94
    I0515 00:22:01.935683 23147 solver.cpp:244]     Train net output #1: loss_c = 0.153903 (* 1 = 0.153903 loss)
    I0515 00:22:01.935698 23147 sgd_solver.cpp:106] Iteration 900, lr = 0.000187482
    I0515 00:22:09.998468 23147 solver.cpp:337] Iteration 1000, Testing net (#0)
    I0515 00:22:14.422369 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.781917
    I0515 00:22:14.422423 23147 solver.cpp:404]     Test net output #1: loss_c = 0.689119 (* 1 = 0.689119 loss)
    I0515 00:22:14.476177 23147 solver.cpp:228] Iteration 1000, loss = 0.157666
    I0515 00:22:14.476203 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.96
    I0515 00:22:14.476222 23147 solver.cpp:244]     Train net output #1: loss_c = 0.157666 (* 1 = 0.157666 loss)
    I0515 00:22:14.476241 23147 sgd_solver.cpp:106] Iteration 1000, lr = 0.000186202
    I0515 00:22:22.617501 23147 solver.cpp:228] Iteration 1100, loss = 0.188451
    I0515 00:22:22.617559 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 00:22:22.617579 23147 solver.cpp:244]     Train net output #1: loss_c = 0.188451 (* 1 = 0.188451 loss)
    I0515 00:22:22.617599 23147 sgd_solver.cpp:106] Iteration 1100, lr = 0.000184943
    I0515 00:22:30.763491 23147 solver.cpp:228] Iteration 1200, loss = 0.205193
    I0515 00:22:30.763639 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.94
    I0515 00:22:30.763684 23147 solver.cpp:244]     Train net output #1: loss_c = 0.205193 (* 1 = 0.205193 loss)
    I0515 00:22:30.763710 23147 sgd_solver.cpp:106] Iteration 1200, lr = 0.000183703
    I0515 00:22:38.903888 23147 solver.cpp:228] Iteration 1300, loss = 0.13401
    I0515 00:22:38.903934 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.94
    I0515 00:22:38.903954 23147 solver.cpp:244]     Train net output #1: loss_c = 0.13401 (* 1 = 0.13401 loss)
    I0515 00:22:38.903969 23147 sgd_solver.cpp:106] Iteration 1300, lr = 0.000182482
    I0515 00:22:47.044226 23147 solver.cpp:228] Iteration 1400, loss = 0.149625
    I0515 00:22:47.044273 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.96
    I0515 00:22:47.044293 23147 solver.cpp:244]     Train net output #1: loss_c = 0.149625 (* 1 = 0.149625 loss)
    I0515 00:22:47.044307 23147 sgd_solver.cpp:106] Iteration 1400, lr = 0.000181281
    I0515 00:22:55.185609 23147 solver.cpp:228] Iteration 1500, loss = 0.186672
    I0515 00:22:55.185648 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.95
    I0515 00:22:55.185667 23147 solver.cpp:244]     Train net output #1: loss_c = 0.186672 (* 1 = 0.186672 loss)
    I0515 00:22:55.185683 23147 sgd_solver.cpp:106] Iteration 1500, lr = 0.000180097
    I0515 00:23:03.322026 23147 solver.cpp:228] Iteration 1600, loss = 0.149855
    I0515 00:23:03.322216 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.95
    I0515 00:23:03.322238 23147 solver.cpp:244]     Train net output #1: loss_c = 0.149855 (* 1 = 0.149855 loss)
    I0515 00:23:03.322253 23147 sgd_solver.cpp:106] Iteration 1600, lr = 0.000178931
    I0515 00:23:11.451588 23147 solver.cpp:228] Iteration 1700, loss = 0.142251
    I0515 00:23:11.451632 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.96
    I0515 00:23:11.451652 23147 solver.cpp:244]     Train net output #1: loss_c = 0.142251 (* 1 = 0.142251 loss)
    I0515 00:23:11.451666 23147 sgd_solver.cpp:106] Iteration 1700, lr = 0.000177783
    I0515 00:23:19.593376 23147 solver.cpp:228] Iteration 1800, loss = 0.105358
    I0515 00:23:19.593420 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.96
    I0515 00:23:19.593441 23147 solver.cpp:244]     Train net output #1: loss_c = 0.105358 (* 1 = 0.105358 loss)
    I0515 00:23:19.593454 23147 sgd_solver.cpp:106] Iteration 1800, lr = 0.000176652
    I0515 00:23:27.734899 23147 solver.cpp:228] Iteration 1900, loss = 0.0937696
    I0515 00:23:27.734941 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.96
    I0515 00:23:27.734961 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0937695 (* 1 = 0.0937695 loss)
    I0515 00:23:27.734975 23147 sgd_solver.cpp:106] Iteration 1900, lr = 0.000175537
    I0515 00:23:35.795773 23147 solver.cpp:337] Iteration 2000, Testing net (#0)
    I0515 00:23:40.196486 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.77825
    I0515 00:23:40.196544 23147 solver.cpp:404]     Test net output #1: loss_c = 0.676467 (* 1 = 0.676467 loss)
    I0515 00:23:40.247779 23147 solver.cpp:228] Iteration 2000, loss = 0.0652314
    I0515 00:23:40.247819 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:23:40.247839 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0652314 (* 1 = 0.0652314 loss)
    I0515 00:23:40.247862 23147 sgd_solver.cpp:106] Iteration 2000, lr = 0.000174439
    I0515 00:23:48.389807 23147 solver.cpp:228] Iteration 2100, loss = 0.201123
    I0515 00:23:48.389855 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.94
    I0515 00:23:48.389879 23147 solver.cpp:244]     Train net output #1: loss_c = 0.201123 (* 1 = 0.201123 loss)
    I0515 00:23:48.389894 23147 sgd_solver.cpp:106] Iteration 2100, lr = 0.000173357
    I0515 00:23:56.531635 23147 solver.cpp:228] Iteration 2200, loss = 0.088149
    I0515 00:23:56.531678 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:23:56.531700 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0881489 (* 1 = 0.0881489 loss)
    I0515 00:23:56.531715 23147 sgd_solver.cpp:106] Iteration 2200, lr = 0.00017229
    I0515 00:24:04.673355 23147 solver.cpp:228] Iteration 2300, loss = 0.141884
    I0515 00:24:04.673401 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.95
    I0515 00:24:04.673423 23147 solver.cpp:244]     Train net output #1: loss_c = 0.141884 (* 1 = 0.141884 loss)
    I0515 00:24:04.673437 23147 sgd_solver.cpp:106] Iteration 2300, lr = 0.000171238
    I0515 00:24:12.814910 23147 solver.cpp:228] Iteration 2400, loss = 0.122307
    I0515 00:24:12.815172 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.96
    I0515 00:24:12.815218 23147 solver.cpp:244]     Train net output #1: loss_c = 0.122307 (* 1 = 0.122307 loss)
    I0515 00:24:12.815244 23147 sgd_solver.cpp:106] Iteration 2400, lr = 0.000170202
    I0515 00:24:20.935571 23147 solver.cpp:228] Iteration 2500, loss = 0.143125
    I0515 00:24:20.935617 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.96
    I0515 00:24:20.935637 23147 solver.cpp:244]     Train net output #1: loss_c = 0.143125 (* 1 = 0.143125 loss)
    I0515 00:24:20.935652 23147 sgd_solver.cpp:106] Iteration 2500, lr = 0.000169179
    I0515 00:24:29.076360 23147 solver.cpp:228] Iteration 2600, loss = 0.0727476
    I0515 00:24:29.076406 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 00:24:29.076426 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0727476 (* 1 = 0.0727476 loss)
    I0515 00:24:29.076442 23147 sgd_solver.cpp:106] Iteration 2600, lr = 0.000168171
    I0515 00:24:37.215654 23147 solver.cpp:228] Iteration 2700, loss = 0.0738156
    I0515 00:24:37.215698 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.96
    I0515 00:24:37.215718 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0738155 (* 1 = 0.0738155 loss)
    I0515 00:24:37.215733 23147 sgd_solver.cpp:106] Iteration 2700, lr = 0.000167177
    I0515 00:24:45.355484 23147 solver.cpp:228] Iteration 2800, loss = 0.0925319
    I0515 00:24:45.355723 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.96
    I0515 00:24:45.355772 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0925318 (* 1 = 0.0925318 loss)
    I0515 00:24:45.355797 23147 sgd_solver.cpp:106] Iteration 2800, lr = 0.000166197
    I0515 00:24:53.496520 23147 solver.cpp:228] Iteration 2900, loss = 0.0705444
    I0515 00:24:53.496570 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:24:53.496590 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0705443 (* 1 = 0.0705443 loss)
    I0515 00:24:53.496605 23147 sgd_solver.cpp:106] Iteration 2900, lr = 0.00016523
    I0515 00:25:01.556342 23147 solver.cpp:337] Iteration 3000, Testing net (#0)
    I0515 00:25:05.963806 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.78125
    I0515 00:25:05.963860 23147 solver.cpp:404]     Test net output #1: loss_c = 0.661877 (* 1 = 0.661877 loss)
    I0515 00:25:06.015185 23147 solver.cpp:228] Iteration 3000, loss = 0.0706372
    I0515 00:25:06.015208 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:25:06.015226 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0706371 (* 1 = 0.0706371 loss)
    I0515 00:25:06.015249 23147 sgd_solver.cpp:106] Iteration 3000, lr = 0.000164275
    I0515 00:25:14.156615 23147 solver.cpp:228] Iteration 3100, loss = 0.0965188
    I0515 00:25:14.156664 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.96
    I0515 00:25:14.156684 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0965187 (* 1 = 0.0965187 loss)
    I0515 00:25:14.156699 23147 sgd_solver.cpp:106] Iteration 3100, lr = 0.000163334
    I0515 00:25:22.297809 23147 solver.cpp:228] Iteration 3200, loss = 0.0521626
    I0515 00:25:22.297960 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:25:22.298003 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0521625 (* 1 = 0.0521625 loss)
    I0515 00:25:22.298028 23147 sgd_solver.cpp:106] Iteration 3200, lr = 0.000162405
    I0515 00:25:30.442827 23147 solver.cpp:228] Iteration 3300, loss = 0.0467749
    I0515 00:25:30.442869 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:25:30.442889 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0467749 (* 1 = 0.0467749 loss)
    I0515 00:25:30.442903 23147 sgd_solver.cpp:106] Iteration 3300, lr = 0.000161488
    I0515 00:25:38.581008 23147 solver.cpp:228] Iteration 3400, loss = 0.117206
    I0515 00:25:38.581056 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.95
    I0515 00:25:38.581076 23147 solver.cpp:244]     Train net output #1: loss_c = 0.117206 (* 1 = 0.117206 loss)
    I0515 00:25:38.581092 23147 sgd_solver.cpp:106] Iteration 3400, lr = 0.000160584
    I0515 00:25:46.723342 23147 solver.cpp:228] Iteration 3500, loss = 0.066325
    I0515 00:25:46.723394 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:25:46.723417 23147 solver.cpp:244]     Train net output #1: loss_c = 0.066325 (* 1 = 0.066325 loss)
    I0515 00:25:46.723431 23147 sgd_solver.cpp:106] Iteration 3500, lr = 0.000159691
    I0515 00:25:54.865510 23147 solver.cpp:228] Iteration 3600, loss = 0.0601808
    I0515 00:25:54.865667 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:25:54.865713 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0601808 (* 1 = 0.0601808 loss)
    I0515 00:25:54.865738 23147 sgd_solver.cpp:106] Iteration 3600, lr = 0.000158809
    I0515 00:26:03.005157 23147 solver.cpp:228] Iteration 3700, loss = 0.101852
    I0515 00:26:03.005205 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 00:26:03.005225 23147 solver.cpp:244]     Train net output #1: loss_c = 0.101852 (* 1 = 0.101852 loss)
    I0515 00:26:03.005240 23147 sgd_solver.cpp:106] Iteration 3700, lr = 0.000157939
    I0515 00:26:11.146795 23147 solver.cpp:228] Iteration 3800, loss = 0.0276547
    I0515 00:26:11.146842 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:26:11.146862 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0276547 (* 1 = 0.0276547 loss)
    I0515 00:26:11.146878 23147 sgd_solver.cpp:106] Iteration 3800, lr = 0.00015708
    I0515 00:26:19.273697 23147 solver.cpp:228] Iteration 3900, loss = 0.051455
    I0515 00:26:19.273746 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:26:19.273767 23147 solver.cpp:244]     Train net output #1: loss_c = 0.051455 (* 1 = 0.051455 loss)
    I0515 00:26:19.273782 23147 sgd_solver.cpp:106] Iteration 3900, lr = 0.000156232
    I0515 00:26:27.333797 23147 solver.cpp:337] Iteration 4000, Testing net (#0)
    I0515 00:26:31.692436 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.80125
    I0515 00:26:31.692505 23147 solver.cpp:404]     Test net output #1: loss_c = 0.695844 (* 1 = 0.695844 loss)
    I0515 00:26:31.747535 23147 solver.cpp:228] Iteration 4000, loss = 0.0300884
    I0515 00:26:31.747571 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:26:31.747597 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0300884 (* 1 = 0.0300884 loss)
    I0515 00:26:31.747623 23147 sgd_solver.cpp:106] Iteration 4000, lr = 0.000155394
    I0515 00:26:39.833719 23147 solver.cpp:228] Iteration 4100, loss = 0.0454649
    I0515 00:26:39.833781 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:26:39.833809 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0454649 (* 1 = 0.0454649 loss)
    I0515 00:26:39.833832 23147 sgd_solver.cpp:106] Iteration 4100, lr = 0.000154567
    I0515 00:26:47.921243 23147 solver.cpp:228] Iteration 4200, loss = 0.0195914
    I0515 00:26:47.921305 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:26:47.921337 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0195915 (* 1 = 0.0195915 loss)
    I0515 00:26:47.921360 23147 sgd_solver.cpp:106] Iteration 4200, lr = 0.00015375
    I0515 00:26:56.005525 23147 solver.cpp:228] Iteration 4300, loss = 0.0392389
    I0515 00:26:56.005586 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:26:56.005615 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0392389 (* 1 = 0.0392389 loss)
    I0515 00:26:56.005637 23147 sgd_solver.cpp:106] Iteration 4300, lr = 0.000152942
    I0515 00:27:04.093081 23147 solver.cpp:228] Iteration 4400, loss = 0.0460643
    I0515 00:27:04.093209 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 00:27:04.093240 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0460643 (* 1 = 0.0460643 loss)
    I0515 00:27:04.093261 23147 sgd_solver.cpp:106] Iteration 4400, lr = 0.000152145
    I0515 00:27:12.184487 23147 solver.cpp:228] Iteration 4500, loss = 0.0423088
    I0515 00:27:12.184545 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:27:12.184574 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0423088 (* 1 = 0.0423088 loss)
    I0515 00:27:12.184597 23147 sgd_solver.cpp:106] Iteration 4500, lr = 0.000151358
    I0515 00:27:20.279356 23147 solver.cpp:228] Iteration 4600, loss = 0.0363668
    I0515 00:27:20.279418 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:27:20.279448 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0363668 (* 1 = 0.0363668 loss)
    I0515 00:27:20.279469 23147 sgd_solver.cpp:106] Iteration 4600, lr = 0.000150579
    I0515 00:27:28.385536 23147 solver.cpp:228] Iteration 4700, loss = 0.05545
    I0515 00:27:28.385576 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:27:28.385594 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0554501 (* 1 = 0.0554501 loss)
    I0515 00:27:28.385609 23147 sgd_solver.cpp:106] Iteration 4700, lr = 0.00014981
    I0515 00:27:36.563176 23147 solver.cpp:228] Iteration 4800, loss = 0.02455
    I0515 00:27:36.563352 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:27:36.563372 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0245501 (* 1 = 0.0245501 loss)
    I0515 00:27:36.563388 23147 sgd_solver.cpp:106] Iteration 4800, lr = 0.000149051
    I0515 00:27:44.702306 23147 solver.cpp:228] Iteration 4900, loss = 0.0358918
    I0515 00:27:44.702358 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:27:44.702378 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0358919 (* 1 = 0.0358919 loss)
    I0515 00:27:44.702392 23147 sgd_solver.cpp:106] Iteration 4900, lr = 0.0001483
    I0515 00:27:52.761144 23147 solver.cpp:337] Iteration 5000, Testing net (#0)
    I0515 00:27:57.176450 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.7925
    I0515 00:27:57.176504 23147 solver.cpp:404]     Test net output #1: loss_c = 0.66512 (* 1 = 0.66512 loss)
    I0515 00:27:57.232569 23147 solver.cpp:228] Iteration 5000, loss = 0.0478103
    I0515 00:27:57.232630 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:27:57.232659 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0478104 (* 1 = 0.0478104 loss)
    I0515 00:27:57.232684 23147 sgd_solver.cpp:106] Iteration 5000, lr = 0.000147558
    I0515 00:28:05.325909 23147 solver.cpp:228] Iteration 5100, loss = 0.0291576
    I0515 00:28:05.325970 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:28:05.325999 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0291576 (* 1 = 0.0291576 loss)
    I0515 00:28:05.326020 23147 sgd_solver.cpp:106] Iteration 5100, lr = 0.000146824
    I0515 00:28:13.410617 23147 solver.cpp:228] Iteration 5200, loss = 0.052846
    I0515 00:28:13.410763 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:28:13.410807 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0528461 (* 1 = 0.0528461 loss)
    I0515 00:28:13.410832 23147 sgd_solver.cpp:106] Iteration 5200, lr = 0.000146099
    I0515 00:28:21.551537 23147 solver.cpp:228] Iteration 5300, loss = 0.0282989
    I0515 00:28:21.551581 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:28:21.551602 23147 solver.cpp:244]     Train net output #1: loss_c = 0.028299 (* 1 = 0.028299 loss)
    I0515 00:28:21.551616 23147 sgd_solver.cpp:106] Iteration 5300, lr = 0.000145382
    I0515 00:28:29.693116 23147 solver.cpp:228] Iteration 5400, loss = 0.092634
    I0515 00:28:29.693168 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:28:29.693188 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0926341 (* 1 = 0.0926341 loss)
    I0515 00:28:29.693203 23147 sgd_solver.cpp:106] Iteration 5400, lr = 0.000144674
    I0515 00:28:37.833829 23147 solver.cpp:228] Iteration 5500, loss = 0.0486075
    I0515 00:28:37.833880 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:28:37.833904 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0486075 (* 1 = 0.0486075 loss)
    I0515 00:28:37.833920 23147 sgd_solver.cpp:106] Iteration 5500, lr = 0.000143973
    I0515 00:28:45.975522 23147 solver.cpp:228] Iteration 5600, loss = 0.0221497
    I0515 00:28:45.975630 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:28:45.975651 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0221497 (* 1 = 0.0221497 loss)
    I0515 00:28:45.975669 23147 sgd_solver.cpp:106] Iteration 5600, lr = 0.00014328
    I0515 00:28:54.115576 23147 solver.cpp:228] Iteration 5700, loss = 0.0236901
    I0515 00:28:54.115623 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:28:54.115645 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0236901 (* 1 = 0.0236901 loss)
    I0515 00:28:54.115660 23147 sgd_solver.cpp:106] Iteration 5700, lr = 0.000142595
    I0515 00:29:02.258816 23147 solver.cpp:228] Iteration 5800, loss = 0.0269928
    I0515 00:29:02.258867 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:29:02.258888 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0269928 (* 1 = 0.0269928 loss)
    I0515 00:29:02.258903 23147 sgd_solver.cpp:106] Iteration 5800, lr = 0.000141918
    I0515 00:29:10.400642 23147 solver.cpp:228] Iteration 5900, loss = 0.0694388
    I0515 00:29:10.400694 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 00:29:10.400715 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0694388 (* 1 = 0.0694388 loss)
    I0515 00:29:10.400730 23147 sgd_solver.cpp:106] Iteration 5900, lr = 0.000141248
    I0515 00:29:18.461310 23147 solver.cpp:337] Iteration 6000, Testing net (#0)
    I0515 00:29:22.874963 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.812666
    I0515 00:29:22.875022 23147 solver.cpp:404]     Test net output #1: loss_c = 0.644565 (* 1 = 0.644565 loss)
    I0515 00:29:22.927332 23147 solver.cpp:228] Iteration 6000, loss = 0.0547985
    I0515 00:29:22.927402 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:29:22.927423 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0547985 (* 1 = 0.0547985 loss)
    I0515 00:29:22.927444 23147 sgd_solver.cpp:106] Iteration 6000, lr = 0.000140585
    I0515 00:29:31.068859 23147 solver.cpp:228] Iteration 6100, loss = 0.0176325
    I0515 00:29:31.068903 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:29:31.068927 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0176326 (* 1 = 0.0176326 loss)
    I0515 00:29:31.068940 23147 sgd_solver.cpp:106] Iteration 6100, lr = 0.00013993
    I0515 00:29:39.207460 23147 solver.cpp:228] Iteration 6200, loss = 0.0422857
    I0515 00:29:39.207518 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:29:39.207538 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0422857 (* 1 = 0.0422857 loss)
    I0515 00:29:39.207553 23147 sgd_solver.cpp:106] Iteration 6200, lr = 0.000139282
    I0515 00:29:47.347283 23147 solver.cpp:228] Iteration 6300, loss = 0.0487895
    I0515 00:29:47.347344 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 00:29:47.347373 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0487895 (* 1 = 0.0487895 loss)
    I0515 00:29:47.347394 23147 sgd_solver.cpp:106] Iteration 6300, lr = 0.00013864
    I0515 00:29:55.433811 23147 solver.cpp:228] Iteration 6400, loss = 0.105585
    I0515 00:29:55.433917 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 00:29:55.433946 23147 solver.cpp:244]     Train net output #1: loss_c = 0.105585 (* 1 = 0.105585 loss)
    I0515 00:29:55.433969 23147 sgd_solver.cpp:106] Iteration 6400, lr = 0.000138006
    I0515 00:30:03.521307 23147 solver.cpp:228] Iteration 6500, loss = 0.0309051
    I0515 00:30:03.521365 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:30:03.521395 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0309051 (* 1 = 0.0309051 loss)
    I0515 00:30:03.521417 23147 sgd_solver.cpp:106] Iteration 6500, lr = 0.000137378
    I0515 00:30:11.610479 23147 solver.cpp:228] Iteration 6600, loss = 0.0532491
    I0515 00:30:11.610540 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:30:11.610570 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0532491 (* 1 = 0.0532491 loss)
    I0515 00:30:11.610592 23147 sgd_solver.cpp:106] Iteration 6600, lr = 0.000136757
    I0515 00:30:19.707129 23147 solver.cpp:228] Iteration 6700, loss = 0.022792
    I0515 00:30:19.707177 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:30:19.707206 23147 solver.cpp:244]     Train net output #1: loss_c = 0.022792 (* 1 = 0.022792 loss)
    I0515 00:30:19.707229 23147 sgd_solver.cpp:106] Iteration 6700, lr = 0.000136142
    I0515 00:30:27.795375 23147 solver.cpp:228] Iteration 6800, loss = 0.0209266
    I0515 00:30:27.795467 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:30:27.795496 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0209266 (* 1 = 0.0209266 loss)
    I0515 00:30:27.795523 23147 sgd_solver.cpp:106] Iteration 6800, lr = 0.000135534
    I0515 00:30:35.882522 23147 solver.cpp:228] Iteration 6900, loss = 0.0289904
    I0515 00:30:35.882576 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:30:35.882606 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0289904 (* 1 = 0.0289904 loss)
    I0515 00:30:35.882627 23147 sgd_solver.cpp:106] Iteration 6900, lr = 0.000134932
    I0515 00:30:43.892549 23147 solver.cpp:337] Iteration 7000, Testing net (#0)
    I0515 00:30:48.248493 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.805583
    I0515 00:30:48.248551 23147 solver.cpp:404]     Test net output #1: loss_c = 0.58964 (* 1 = 0.58964 loss)
    I0515 00:30:48.303742 23147 solver.cpp:228] Iteration 7000, loss = 0.045811
    I0515 00:30:48.303776 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 00:30:48.303802 23147 solver.cpp:244]     Train net output #1: loss_c = 0.045811 (* 1 = 0.045811 loss)
    I0515 00:30:48.303828 23147 sgd_solver.cpp:106] Iteration 7000, lr = 0.000134336
    I0515 00:30:56.394673 23147 solver.cpp:228] Iteration 7100, loss = 0.0589104
    I0515 00:30:56.394729 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:30:56.394758 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0589104 (* 1 = 0.0589104 loss)
    I0515 00:30:56.394781 23147 sgd_solver.cpp:106] Iteration 7100, lr = 0.000133747
    I0515 00:31:04.481503 23147 solver.cpp:228] Iteration 7200, loss = 0.0398815
    I0515 00:31:04.481710 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:31:04.481772 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0398814 (* 1 = 0.0398814 loss)
    I0515 00:31:04.481796 23147 sgd_solver.cpp:106] Iteration 7200, lr = 0.000133163
    I0515 00:31:12.568109 23147 solver.cpp:228] Iteration 7300, loss = 0.0272044
    I0515 00:31:12.568168 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:31:12.568197 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0272043 (* 1 = 0.0272043 loss)
    I0515 00:31:12.568218 23147 sgd_solver.cpp:106] Iteration 7300, lr = 0.000132585
    I0515 00:31:20.677939 23147 solver.cpp:228] Iteration 7400, loss = 0.0512787
    I0515 00:31:20.677994 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:31:20.678021 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0512786 (* 1 = 0.0512786 loss)
    I0515 00:31:20.678043 23147 sgd_solver.cpp:106] Iteration 7400, lr = 0.000132013
    I0515 00:31:28.763038 23147 solver.cpp:228] Iteration 7500, loss = 0.0389936
    I0515 00:31:28.763093 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:31:28.763123 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0389935 (* 1 = 0.0389935 loss)
    I0515 00:31:28.763144 23147 sgd_solver.cpp:106] Iteration 7500, lr = 0.000131447
    I0515 00:31:36.853240 23147 solver.cpp:228] Iteration 7600, loss = 0.0295571
    I0515 00:31:36.853387 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:31:36.853435 23147 solver.cpp:244]     Train net output #1: loss_c = 0.029557 (* 1 = 0.029557 loss)
    I0515 00:31:36.853459 23147 sgd_solver.cpp:106] Iteration 7600, lr = 0.000130887
    I0515 00:31:44.992295 23147 solver.cpp:228] Iteration 7700, loss = 0.0452869
    I0515 00:31:44.992344 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:31:44.992368 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0452868 (* 1 = 0.0452868 loss)
    I0515 00:31:44.992383 23147 sgd_solver.cpp:106] Iteration 7700, lr = 0.000130332
    I0515 00:31:53.132900 23147 solver.cpp:228] Iteration 7800, loss = 0.0351686
    I0515 00:31:53.132941 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:31:53.132963 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0351686 (* 1 = 0.0351686 loss)
    I0515 00:31:53.132977 23147 sgd_solver.cpp:106] Iteration 7800, lr = 0.000129782
    I0515 00:32:01.273990 23147 solver.cpp:228] Iteration 7900, loss = 0.0248163
    I0515 00:32:01.274041 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:32:01.274065 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0248162 (* 1 = 0.0248162 loss)
    I0515 00:32:01.274078 23147 sgd_solver.cpp:106] Iteration 7900, lr = 0.000129238
    I0515 00:32:09.334411 23147 solver.cpp:337] Iteration 8000, Testing net (#0)
    I0515 00:32:13.722434 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.814
    I0515 00:32:13.722492 23147 solver.cpp:404]     Test net output #1: loss_c = 0.60041 (* 1 = 0.60041 loss)
    I0515 00:32:13.774507 23147 solver.cpp:228] Iteration 8000, loss = 0.0346037
    I0515 00:32:13.774562 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:32:13.774582 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0346036 (* 1 = 0.0346036 loss)
    I0515 00:32:13.774600 23147 sgd_solver.cpp:106] Iteration 8000, lr = 0.000128699
    I0515 00:32:21.868083 23147 solver.cpp:228] Iteration 8100, loss = 0.0347974
    I0515 00:32:21.868144 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:32:21.868173 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0347973 (* 1 = 0.0347973 loss)
    I0515 00:32:21.868196 23147 sgd_solver.cpp:106] Iteration 8100, lr = 0.000128165
    I0515 00:32:29.954819 23147 solver.cpp:228] Iteration 8200, loss = 0.0718848
    I0515 00:32:29.954872 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:32:29.954901 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0718848 (* 1 = 0.0718848 loss)
    I0515 00:32:29.954924 23147 sgd_solver.cpp:106] Iteration 8200, lr = 0.000127637
    I0515 00:32:38.040246 23147 solver.cpp:228] Iteration 8300, loss = 0.030082
    I0515 00:32:38.040304 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:32:38.040334 23147 solver.cpp:244]     Train net output #1: loss_c = 0.030082 (* 1 = 0.030082 loss)
    I0515 00:32:38.040356 23147 sgd_solver.cpp:106] Iteration 8300, lr = 0.000127113
    I0515 00:32:46.129484 23147 solver.cpp:228] Iteration 8400, loss = 0.0459857
    I0515 00:32:46.129585 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:32:46.129614 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0459857 (* 1 = 0.0459857 loss)
    I0515 00:32:46.129637 23147 sgd_solver.cpp:106] Iteration 8400, lr = 0.000126595
    I0515 00:32:54.219535 23147 solver.cpp:228] Iteration 8500, loss = 0.0196482
    I0515 00:32:54.219588 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:32:54.219619 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0196481 (* 1 = 0.0196481 loss)
    I0515 00:32:54.219640 23147 sgd_solver.cpp:106] Iteration 8500, lr = 0.000126081
    I0515 00:33:02.306768 23147 solver.cpp:228] Iteration 8600, loss = 0.0195381
    I0515 00:33:02.306823 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:33:02.306852 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0195381 (* 1 = 0.0195381 loss)
    I0515 00:33:02.306874 23147 sgd_solver.cpp:106] Iteration 8600, lr = 0.000125573
    I0515 00:33:10.399526 23147 solver.cpp:228] Iteration 8700, loss = 0.0397299
    I0515 00:33:10.399580 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:33:10.399607 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0397299 (* 1 = 0.0397299 loss)
    I0515 00:33:10.399629 23147 sgd_solver.cpp:106] Iteration 8700, lr = 0.000125069
    I0515 00:33:18.489239 23147 solver.cpp:228] Iteration 8800, loss = 0.0468039
    I0515 00:33:18.489362 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 00:33:18.489393 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0468039 (* 1 = 0.0468039 loss)
    I0515 00:33:18.489414 23147 sgd_solver.cpp:106] Iteration 8800, lr = 0.000124569
    I0515 00:33:26.577570 23147 solver.cpp:228] Iteration 8900, loss = 0.0361443
    I0515 00:33:26.577631 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:33:26.577661 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0361443 (* 1 = 0.0361443 loss)
    I0515 00:33:26.577683 23147 sgd_solver.cpp:106] Iteration 8900, lr = 0.000124075
    I0515 00:33:34.583235 23147 solver.cpp:337] Iteration 9000, Testing net (#0)
    I0515 00:33:38.941087 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.796916
    I0515 00:33:38.941154 23147 solver.cpp:404]     Test net output #1: loss_c = 0.623339 (* 1 = 0.623339 loss)
    I0515 00:33:38.996199 23147 solver.cpp:228] Iteration 9000, loss = 0.0430436
    I0515 00:33:38.996260 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:33:38.996290 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0430436 (* 1 = 0.0430436 loss)
    I0515 00:33:38.996315 23147 sgd_solver.cpp:106] Iteration 9000, lr = 0.000123585
    I0515 00:33:47.134655 23147 solver.cpp:228] Iteration 9100, loss = 0.0251566
    I0515 00:33:47.134702 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:33:47.134723 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0251566 (* 1 = 0.0251566 loss)
    I0515 00:33:47.134738 23147 sgd_solver.cpp:106] Iteration 9100, lr = 0.000123099
    I0515 00:33:55.274615 23147 solver.cpp:228] Iteration 9200, loss = 0.0206143
    I0515 00:33:55.274802 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:33:55.274824 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0206143 (* 1 = 0.0206143 loss)
    I0515 00:33:55.274839 23147 sgd_solver.cpp:106] Iteration 9200, lr = 0.000122618
    I0515 00:34:03.414223 23147 solver.cpp:228] Iteration 9300, loss = 0.0262582
    I0515 00:34:03.414273 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:34:03.414294 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0262582 (* 1 = 0.0262582 loss)
    I0515 00:34:03.414309 23147 sgd_solver.cpp:106] Iteration 9300, lr = 0.000122141
    I0515 00:34:11.556329 23147 solver.cpp:228] Iteration 9400, loss = 0.0343376
    I0515 00:34:11.556372 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:34:11.556393 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0343375 (* 1 = 0.0343375 loss)
    I0515 00:34:11.556407 23147 sgd_solver.cpp:106] Iteration 9400, lr = 0.000121669
    I0515 00:34:19.699095 23147 solver.cpp:228] Iteration 9500, loss = 0.0240549
    I0515 00:34:19.699141 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:34:19.699162 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0240549 (* 1 = 0.0240549 loss)
    I0515 00:34:19.699175 23147 sgd_solver.cpp:106] Iteration 9500, lr = 0.0001212
    I0515 00:34:27.842169 23147 solver.cpp:228] Iteration 9600, loss = 0.0161116
    I0515 00:34:27.842319 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:34:27.842365 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0161115 (* 1 = 0.0161115 loss)
    I0515 00:34:27.842391 23147 sgd_solver.cpp:106] Iteration 9600, lr = 0.000120736
    I0515 00:34:35.981495 23147 solver.cpp:228] Iteration 9700, loss = 0.0191925
    I0515 00:34:35.981546 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:34:35.981566 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0191925 (* 1 = 0.0191925 loss)
    I0515 00:34:35.981581 23147 sgd_solver.cpp:106] Iteration 9700, lr = 0.000120276
    I0515 00:34:44.121759 23147 solver.cpp:228] Iteration 9800, loss = 0.0296545
    I0515 00:34:44.121811 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:34:44.121831 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0296545 (* 1 = 0.0296545 loss)
    I0515 00:34:44.121846 23147 sgd_solver.cpp:106] Iteration 9800, lr = 0.00011982
    I0515 00:34:52.261878 23147 solver.cpp:228] Iteration 9900, loss = 0.0141021
    I0515 00:34:52.261930 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:34:52.261952 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0141021 (* 1 = 0.0141021 loss)
    I0515 00:34:52.261967 23147 sgd_solver.cpp:106] Iteration 9900, lr = 0.000119369
    I0515 00:35:00.321667 23147 solver.cpp:337] Iteration 10000, Testing net (#0)
    I0515 00:35:04.705379 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.799417
    I0515 00:35:04.705427 23147 solver.cpp:404]     Test net output #1: loss_c = 0.590137 (* 1 = 0.590137 loss)
    I0515 00:35:04.756790 23147 solver.cpp:228] Iteration 10000, loss = 0.0888725
    I0515 00:35:04.756829 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.96
    I0515 00:35:04.756849 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0888725 (* 1 = 0.0888725 loss)
    I0515 00:35:04.756867 23147 sgd_solver.cpp:106] Iteration 10000, lr = 0.000118921
    I0515 00:35:12.897279 23147 solver.cpp:228] Iteration 10100, loss = 0.0153379
    I0515 00:35:12.897331 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:35:12.897352 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0153379 (* 1 = 0.0153379 loss)
    I0515 00:35:12.897367 23147 sgd_solver.cpp:106] Iteration 10100, lr = 0.000118477
    I0515 00:35:21.030099 23147 solver.cpp:228] Iteration 10200, loss = 0.0617474
    I0515 00:35:21.030141 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 00:35:21.030165 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0617474 (* 1 = 0.0617474 loss)
    I0515 00:35:21.030180 23147 sgd_solver.cpp:106] Iteration 10200, lr = 0.000118037
    I0515 00:35:29.168786 23147 solver.cpp:228] Iteration 10300, loss = 0.0383719
    I0515 00:35:29.168835 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:35:29.168856 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0383718 (* 1 = 0.0383718 loss)
    I0515 00:35:29.168871 23147 sgd_solver.cpp:106] Iteration 10300, lr = 0.0001176
    I0515 00:35:37.309326 23147 solver.cpp:228] Iteration 10400, loss = 0.067195
    I0515 00:35:37.309571 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 00:35:37.309617 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0671949 (* 1 = 0.0671949 loss)
    I0515 00:35:37.309643 23147 sgd_solver.cpp:106] Iteration 10400, lr = 0.000117168
    I0515 00:35:45.451326 23147 solver.cpp:228] Iteration 10500, loss = 0.0293081
    I0515 00:35:45.451375 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:35:45.451396 23147 solver.cpp:244]     Train net output #1: loss_c = 0.029308 (* 1 = 0.029308 loss)
    I0515 00:35:45.451411 23147 sgd_solver.cpp:106] Iteration 10500, lr = 0.000116739
    I0515 00:35:53.592934 23147 solver.cpp:228] Iteration 10600, loss = 0.0287992
    I0515 00:35:53.592983 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:35:53.593006 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0287992 (* 1 = 0.0287992 loss)
    I0515 00:35:53.593021 23147 sgd_solver.cpp:106] Iteration 10600, lr = 0.000116313
    I0515 00:36:01.735198 23147 solver.cpp:228] Iteration 10700, loss = 0.0117416
    I0515 00:36:01.735249 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:36:01.735268 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0117416 (* 1 = 0.0117416 loss)
    I0515 00:36:01.735283 23147 sgd_solver.cpp:106] Iteration 10700, lr = 0.000115892
    I0515 00:36:09.877153 23147 solver.cpp:228] Iteration 10800, loss = 0.0363359
    I0515 00:36:09.877257 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:36:09.877279 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0363359 (* 1 = 0.0363359 loss)
    I0515 00:36:09.877292 23147 sgd_solver.cpp:106] Iteration 10800, lr = 0.000115474
    I0515 00:36:18.017526 23147 solver.cpp:228] Iteration 10900, loss = 0.0446662
    I0515 00:36:18.017568 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:36:18.017591 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0446662 (* 1 = 0.0446662 loss)
    I0515 00:36:18.017606 23147 sgd_solver.cpp:106] Iteration 10900, lr = 0.000115059
    I0515 00:36:26.079742 23147 solver.cpp:337] Iteration 11000, Testing net (#0)
    I0515 00:36:30.450618 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.808833
    I0515 00:36:30.450678 23147 solver.cpp:404]     Test net output #1: loss_c = 0.591554 (* 1 = 0.591554 loss)
    I0515 00:36:30.505565 23147 solver.cpp:228] Iteration 11000, loss = 0.0442617
    I0515 00:36:30.505601 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:36:30.505628 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0442616 (* 1 = 0.0442616 loss)
    I0515 00:36:30.505652 23147 sgd_solver.cpp:106] Iteration 11000, lr = 0.000114648
    I0515 00:36:38.592917 23147 solver.cpp:228] Iteration 11100, loss = 0.0619198
    I0515 00:36:38.592970 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 00:36:38.592998 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0619198 (* 1 = 0.0619198 loss)
    I0515 00:36:38.593020 23147 sgd_solver.cpp:106] Iteration 11100, lr = 0.00011424
    I0515 00:36:46.683028 23147 solver.cpp:228] Iteration 11200, loss = 0.0222484
    I0515 00:36:46.683267 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:36:46.683312 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0222483 (* 1 = 0.0222483 loss)
    I0515 00:36:46.683336 23147 sgd_solver.cpp:106] Iteration 11200, lr = 0.000113836
    I0515 00:36:54.749387 23147 solver.cpp:228] Iteration 11300, loss = 0.02733
    I0515 00:36:54.749431 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:36:54.749454 23147 solver.cpp:244]     Train net output #1: loss_c = 0.02733 (* 1 = 0.02733 loss)
    I0515 00:36:54.749471 23147 sgd_solver.cpp:106] Iteration 11300, lr = 0.000113435
    I0515 00:37:02.863400 23147 solver.cpp:228] Iteration 11400, loss = 0.0398809
    I0515 00:37:02.863453 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:37:02.863473 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0398808 (* 1 = 0.0398808 loss)
    I0515 00:37:02.863489 23147 sgd_solver.cpp:106] Iteration 11400, lr = 0.000113037
    I0515 00:37:11.004550 23147 solver.cpp:228] Iteration 11500, loss = 0.0310092
    I0515 00:37:11.004600 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:37:11.004621 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0310091 (* 1 = 0.0310091 loss)
    I0515 00:37:11.004637 23147 sgd_solver.cpp:106] Iteration 11500, lr = 0.000112642
    I0515 00:37:19.149054 23147 solver.cpp:228] Iteration 11600, loss = 0.0134124
    I0515 00:37:19.149189 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:37:19.149235 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0134124 (* 1 = 0.0134124 loss)
    I0515 00:37:19.149258 23147 sgd_solver.cpp:106] Iteration 11600, lr = 0.000112251
    I0515 00:37:27.288892 23147 solver.cpp:228] Iteration 11700, loss = 0.0357101
    I0515 00:37:27.288931 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:37:27.288950 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0357101 (* 1 = 0.0357101 loss)
    I0515 00:37:27.288964 23147 sgd_solver.cpp:106] Iteration 11700, lr = 0.000111863
    I0515 00:37:35.428544 23147 solver.cpp:228] Iteration 11800, loss = 0.0288281
    I0515 00:37:35.428589 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:37:35.428613 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0288281 (* 1 = 0.0288281 loss)
    I0515 00:37:35.428627 23147 sgd_solver.cpp:106] Iteration 11800, lr = 0.000111478
    I0515 00:37:43.571161 23147 solver.cpp:228] Iteration 11900, loss = 0.0495557
    I0515 00:37:43.571203 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:37:43.571223 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0495556 (* 1 = 0.0495556 loss)
    I0515 00:37:43.571239 23147 sgd_solver.cpp:106] Iteration 11900, lr = 0.000111096
    I0515 00:37:51.630269 23147 solver.cpp:337] Iteration 12000, Testing net (#0)
    I0515 00:37:56.024319 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.798667
    I0515 00:37:56.024369 23147 solver.cpp:404]     Test net output #1: loss_c = 0.615581 (* 1 = 0.615581 loss)
    I0515 00:37:56.080318 23147 solver.cpp:228] Iteration 12000, loss = 0.0257507
    I0515 00:37:56.080381 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:37:56.080410 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0257507 (* 1 = 0.0257507 loss)
    I0515 00:37:56.080435 23147 sgd_solver.cpp:106] Iteration 12000, lr = 0.000110717
    I0515 00:38:04.216140 23147 solver.cpp:228] Iteration 12100, loss = 0.0359227
    I0515 00:38:04.216197 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:38:04.216226 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0359228 (* 1 = 0.0359228 loss)
    I0515 00:38:04.216248 23147 sgd_solver.cpp:106] Iteration 12100, lr = 0.000110341
    I0515 00:38:12.302040 23147 solver.cpp:228] Iteration 12200, loss = 0.0175199
    I0515 00:38:12.302100 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:38:12.302129 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0175199 (* 1 = 0.0175199 loss)
    I0515 00:38:12.302151 23147 sgd_solver.cpp:106] Iteration 12200, lr = 0.000109968
    I0515 00:38:20.398066 23147 solver.cpp:228] Iteration 12300, loss = 0.0259959
    I0515 00:38:20.398120 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:38:20.398149 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0259959 (* 1 = 0.0259959 loss)
    I0515 00:38:20.398170 23147 sgd_solver.cpp:106] Iteration 12300, lr = 0.000109598
    I0515 00:38:28.522342 23147 solver.cpp:228] Iteration 12400, loss = 0.0327433
    I0515 00:38:28.522579 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:38:28.522625 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0327433 (* 1 = 0.0327433 loss)
    I0515 00:38:28.522650 23147 sgd_solver.cpp:106] Iteration 12400, lr = 0.000109231
    I0515 00:38:36.650756 23147 solver.cpp:228] Iteration 12500, loss = 0.00876418
    I0515 00:38:36.650804 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:38:36.650823 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00876421 (* 1 = 0.00876421 loss)
    I0515 00:38:36.650838 23147 sgd_solver.cpp:106] Iteration 12500, lr = 0.000108866
    I0515 00:38:44.792938 23147 solver.cpp:228] Iteration 12600, loss = 0.0732577
    I0515 00:38:44.792985 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:38:44.793004 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0732577 (* 1 = 0.0732577 loss)
    I0515 00:38:44.793020 23147 sgd_solver.cpp:106] Iteration 12600, lr = 0.000108505
    I0515 00:38:52.934226 23147 solver.cpp:228] Iteration 12700, loss = 0.0235279
    I0515 00:38:52.934276 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:38:52.934298 23147 solver.cpp:244]     Train net output #1: loss_c = 0.023528 (* 1 = 0.023528 loss)
    I0515 00:38:52.934314 23147 sgd_solver.cpp:106] Iteration 12700, lr = 0.000108146
    I0515 00:39:01.066939 23147 solver.cpp:228] Iteration 12800, loss = 0.0246004
    I0515 00:39:01.067087 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:39:01.067133 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0246004 (* 1 = 0.0246004 loss)
    I0515 00:39:01.067158 23147 sgd_solver.cpp:106] Iteration 12800, lr = 0.00010779
    I0515 00:39:09.194825 23147 solver.cpp:228] Iteration 12900, loss = 0.0326073
    I0515 00:39:09.194874 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:39:09.194893 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0326074 (* 1 = 0.0326074 loss)
    I0515 00:39:09.194911 23147 sgd_solver.cpp:106] Iteration 12900, lr = 0.000107437
    I0515 00:39:17.255399 23147 solver.cpp:337] Iteration 13000, Testing net (#0)
    I0515 00:39:21.622710 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.797917
    I0515 00:39:21.622766 23147 solver.cpp:404]     Test net output #1: loss_c = 0.631085 (* 1 = 0.631085 loss)
    I0515 00:39:21.678836 23147 solver.cpp:228] Iteration 13000, loss = 0.0405524
    I0515 00:39:21.678896 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:39:21.678926 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0405524 (* 1 = 0.0405524 loss)
    I0515 00:39:21.678951 23147 sgd_solver.cpp:106] Iteration 13000, lr = 0.000107086
    I0515 00:39:29.806059 23147 solver.cpp:228] Iteration 13100, loss = 0.0180977
    I0515 00:39:29.806108 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:39:29.806129 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0180978 (* 1 = 0.0180978 loss)
    I0515 00:39:29.806143 23147 sgd_solver.cpp:106] Iteration 13100, lr = 0.000106738
    I0515 00:39:37.944172 23147 solver.cpp:228] Iteration 13200, loss = 0.0329995
    I0515 00:39:37.944368 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:39:37.944389 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0329995 (* 1 = 0.0329995 loss)
    I0515 00:39:37.944406 23147 sgd_solver.cpp:106] Iteration 13200, lr = 0.000106393
    I0515 00:39:46.085476 23147 solver.cpp:228] Iteration 13300, loss = 0.00653043
    I0515 00:39:46.085528 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:39:46.085549 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00653051 (* 1 = 0.00653051 loss)
    I0515 00:39:46.085564 23147 sgd_solver.cpp:106] Iteration 13300, lr = 0.000106051
    I0515 00:39:54.225932 23147 solver.cpp:228] Iteration 13400, loss = 0.0452398
    I0515 00:39:54.225983 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:39:54.226004 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0452399 (* 1 = 0.0452399 loss)
    I0515 00:39:54.226019 23147 sgd_solver.cpp:106] Iteration 13400, lr = 0.00010571
    I0515 00:40:02.366156 23147 solver.cpp:228] Iteration 13500, loss = 0.0223948
    I0515 00:40:02.366207 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:40:02.366227 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0223949 (* 1 = 0.0223949 loss)
    I0515 00:40:02.366242 23147 sgd_solver.cpp:106] Iteration 13500, lr = 0.000105373
    I0515 00:40:10.506902 23147 solver.cpp:228] Iteration 13600, loss = 0.0148896
    I0515 00:40:10.506978 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:40:10.506999 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0148896 (* 1 = 0.0148896 loss)
    I0515 00:40:10.507016 23147 sgd_solver.cpp:106] Iteration 13600, lr = 0.000105038
    I0515 00:40:18.644834 23147 solver.cpp:228] Iteration 13700, loss = 0.0279905
    I0515 00:40:18.644886 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:40:18.644906 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0279906 (* 1 = 0.0279906 loss)
    I0515 00:40:18.644922 23147 sgd_solver.cpp:106] Iteration 13700, lr = 0.000104705
    I0515 00:40:26.787083 23147 solver.cpp:228] Iteration 13800, loss = 0.0256817
    I0515 00:40:26.787132 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:40:26.787153 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0256818 (* 1 = 0.0256818 loss)
    I0515 00:40:26.787168 23147 sgd_solver.cpp:106] Iteration 13800, lr = 0.000104375
    I0515 00:40:34.928843 23147 solver.cpp:228] Iteration 13900, loss = 0.0555896
    I0515 00:40:34.928892 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:40:34.928913 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0555897 (* 1 = 0.0555897 loss)
    I0515 00:40:34.928927 23147 sgd_solver.cpp:106] Iteration 13900, lr = 0.000104047
    I0515 00:40:42.984267 23147 solver.cpp:337] Iteration 14000, Testing net (#0)
    I0515 00:40:47.374912 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.829583
    I0515 00:40:47.374961 23147 solver.cpp:404]     Test net output #1: loss_c = 0.580338 (* 1 = 0.580338 loss)
    I0515 00:40:47.431181 23147 solver.cpp:228] Iteration 14000, loss = 0.0137979
    I0515 00:40:47.431236 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:40:47.431264 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0137979 (* 1 = 0.0137979 loss)
    I0515 00:40:47.431288 23147 sgd_solver.cpp:106] Iteration 14000, lr = 0.000103722
    I0515 00:40:55.568553 23147 solver.cpp:228] Iteration 14100, loss = 0.010965
    I0515 00:40:55.568598 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:40:55.568619 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0109651 (* 1 = 0.0109651 loss)
    I0515 00:40:55.568634 23147 sgd_solver.cpp:106] Iteration 14100, lr = 0.000103399
    I0515 00:41:03.708667 23147 solver.cpp:228] Iteration 14200, loss = 0.00901656
    I0515 00:41:03.708710 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:41:03.708729 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00901662 (* 1 = 0.00901662 loss)
    I0515 00:41:03.708745 23147 sgd_solver.cpp:106] Iteration 14200, lr = 0.000103079
    I0515 00:41:11.849148 23147 solver.cpp:228] Iteration 14300, loss = 0.0229422
    I0515 00:41:11.849195 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:41:11.849215 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0229423 (* 1 = 0.0229423 loss)
    I0515 00:41:11.849231 23147 sgd_solver.cpp:106] Iteration 14300, lr = 0.00010276
    I0515 00:41:19.958567 23147 solver.cpp:228] Iteration 14400, loss = 0.01869
    I0515 00:41:19.958827 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:41:19.958878 23147 solver.cpp:244]     Train net output #1: loss_c = 0.01869 (* 1 = 0.01869 loss)
    I0515 00:41:19.958902 23147 sgd_solver.cpp:106] Iteration 14400, lr = 0.000102444
    I0515 00:41:28.087285 23147 solver.cpp:228] Iteration 14500, loss = 0.0204913
    I0515 00:41:28.087330 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:41:28.087350 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0204913 (* 1 = 0.0204913 loss)
    I0515 00:41:28.087364 23147 sgd_solver.cpp:106] Iteration 14500, lr = 0.00010213
    I0515 00:41:36.228142 23147 solver.cpp:228] Iteration 14600, loss = 0.0131296
    I0515 00:41:36.228189 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:41:36.228209 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0131297 (* 1 = 0.0131297 loss)
    I0515 00:41:36.228224 23147 sgd_solver.cpp:106] Iteration 14600, lr = 0.000101819
    I0515 00:41:44.369720 23147 solver.cpp:228] Iteration 14700, loss = 0.0348538
    I0515 00:41:44.369770 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:41:44.369792 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0348538 (* 1 = 0.0348538 loss)
    I0515 00:41:44.369807 23147 sgd_solver.cpp:106] Iteration 14700, lr = 0.00010151
    I0515 00:41:52.466208 23147 solver.cpp:228] Iteration 14800, loss = 0.0130305
    I0515 00:41:52.466316 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:41:52.466337 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0130306 (* 1 = 0.0130306 loss)
    I0515 00:41:52.466353 23147 sgd_solver.cpp:106] Iteration 14800, lr = 0.000101202
    I0515 00:42:00.608110 23147 solver.cpp:228] Iteration 14900, loss = 0.05811
    I0515 00:42:00.608160 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:42:00.608180 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0581101 (* 1 = 0.0581101 loss)
    I0515 00:42:00.608196 23147 sgd_solver.cpp:106] Iteration 14900, lr = 0.000100898
    I0515 00:42:08.668870 23147 solver.cpp:337] Iteration 15000, Testing net (#0)
    I0515 00:42:13.077059 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.840917
    I0515 00:42:13.077117 23147 solver.cpp:404]     Test net output #1: loss_c = 0.605869 (* 1 = 0.605869 loss)
    I0515 00:42:13.132288 23147 solver.cpp:228] Iteration 15000, loss = 0.0172646
    I0515 00:42:13.132349 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:42:13.132380 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0172647 (* 1 = 0.0172647 loss)
    I0515 00:42:13.132405 23147 sgd_solver.cpp:106] Iteration 15000, lr = 0.000100595
    I0515 00:42:21.260390 23147 solver.cpp:228] Iteration 15100, loss = 0.0213131
    I0515 00:42:21.260433 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:42:21.260452 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0213132 (* 1 = 0.0213132 loss)
    I0515 00:42:21.260467 23147 sgd_solver.cpp:106] Iteration 15100, lr = 0.000100294
    I0515 00:42:29.399952 23147 solver.cpp:228] Iteration 15200, loss = 0.0359627
    I0515 00:42:29.400055 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:42:29.400075 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0359627 (* 1 = 0.0359627 loss)
    I0515 00:42:29.400090 23147 sgd_solver.cpp:106] Iteration 15200, lr = 9.99953e-05
    I0515 00:42:37.538851 23147 solver.cpp:228] Iteration 15300, loss = 0.018531
    I0515 00:42:37.538897 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:42:37.538916 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0185311 (* 1 = 0.0185311 loss)
    I0515 00:42:37.538930 23147 sgd_solver.cpp:106] Iteration 15300, lr = 9.96987e-05
    I0515 00:42:45.680243 23147 solver.cpp:228] Iteration 15400, loss = 0.0408572
    I0515 00:42:45.680289 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:42:45.680310 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0408572 (* 1 = 0.0408572 loss)
    I0515 00:42:45.680323 23147 sgd_solver.cpp:106] Iteration 15400, lr = 9.94042e-05
    I0515 00:42:53.820235 23147 solver.cpp:228] Iteration 15500, loss = 0.0277656
    I0515 00:42:53.820279 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:42:53.820299 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0277656 (* 1 = 0.0277656 loss)
    I0515 00:42:53.820314 23147 sgd_solver.cpp:106] Iteration 15500, lr = 9.91117e-05
    I0515 00:43:01.960739 23147 solver.cpp:228] Iteration 15600, loss = 0.0140039
    I0515 00:43:01.960922 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:43:01.960974 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0140039 (* 1 = 0.0140039 loss)
    I0515 00:43:01.960990 23147 sgd_solver.cpp:106] Iteration 15600, lr = 9.88212e-05
    I0515 00:43:10.101842 23147 solver.cpp:228] Iteration 15700, loss = 0.0305408
    I0515 00:43:10.101892 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:43:10.101917 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0305409 (* 1 = 0.0305409 loss)
    I0515 00:43:10.101932 23147 sgd_solver.cpp:106] Iteration 15700, lr = 9.85326e-05
    I0515 00:43:18.243290 23147 solver.cpp:228] Iteration 15800, loss = 0.017033
    I0515 00:43:18.243342 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:43:18.243365 23147 solver.cpp:244]     Train net output #1: loss_c = 0.017033 (* 1 = 0.017033 loss)
    I0515 00:43:18.243378 23147 sgd_solver.cpp:106] Iteration 15800, lr = 9.82461e-05
    I0515 00:43:26.385047 23147 solver.cpp:228] Iteration 15900, loss = 0.0199255
    I0515 00:43:26.385093 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:43:26.385114 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0199256 (* 1 = 0.0199256 loss)
    I0515 00:43:26.385128 23147 sgd_solver.cpp:106] Iteration 15900, lr = 9.79614e-05
    I0515 00:43:34.447048 23147 solver.cpp:337] Iteration 16000, Testing net (#0)
    I0515 00:43:38.837126 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.828
    I0515 00:43:38.837182 23147 solver.cpp:404]     Test net output #1: loss_c = 0.556054 (* 1 = 0.556054 loss)
    I0515 00:43:38.892192 23147 solver.cpp:228] Iteration 16000, loss = 0.0139335
    I0515 00:43:38.892253 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:43:38.892282 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0139335 (* 1 = 0.0139335 loss)
    I0515 00:43:38.892308 23147 sgd_solver.cpp:106] Iteration 16000, lr = 9.76787e-05
    I0515 00:43:47.024988 23147 solver.cpp:228] Iteration 16100, loss = 0.0423227
    I0515 00:43:47.025038 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:43:47.025058 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0423227 (* 1 = 0.0423227 loss)
    I0515 00:43:47.025074 23147 sgd_solver.cpp:106] Iteration 16100, lr = 9.73979e-05
    I0515 00:43:55.166507 23147 solver.cpp:228] Iteration 16200, loss = 0.0132787
    I0515 00:43:55.166551 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:43:55.166571 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0132788 (* 1 = 0.0132788 loss)
    I0515 00:43:55.166585 23147 sgd_solver.cpp:106] Iteration 16200, lr = 9.7119e-05
    I0515 00:44:03.305712 23147 solver.cpp:228] Iteration 16300, loss = 0.0293654
    I0515 00:44:03.305763 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:44:03.305786 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0293655 (* 1 = 0.0293655 loss)
    I0515 00:44:03.305801 23147 sgd_solver.cpp:106] Iteration 16300, lr = 9.68419e-05
    I0515 00:44:11.437906 23147 solver.cpp:228] Iteration 16400, loss = 0.0690171
    I0515 00:44:11.438107 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:44:11.438128 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0690171 (* 1 = 0.0690171 loss)
    I0515 00:44:11.438174 23147 sgd_solver.cpp:106] Iteration 16400, lr = 9.65666e-05
    I0515 00:44:19.578243 23147 solver.cpp:228] Iteration 16500, loss = 0.023747
    I0515 00:44:19.578289 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:44:19.578310 23147 solver.cpp:244]     Train net output #1: loss_c = 0.023747 (* 1 = 0.023747 loss)
    I0515 00:44:19.578323 23147 sgd_solver.cpp:106] Iteration 16500, lr = 9.62932e-05
    I0515 00:44:27.718952 23147 solver.cpp:228] Iteration 16600, loss = 0.0253041
    I0515 00:44:27.719009 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:44:27.719034 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0253041 (* 1 = 0.0253041 loss)
    I0515 00:44:27.719050 23147 sgd_solver.cpp:106] Iteration 16600, lr = 9.60216e-05
    I0515 00:44:35.861151 23147 solver.cpp:228] Iteration 16700, loss = 0.0313033
    I0515 00:44:35.861202 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:44:35.861222 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0313033 (* 1 = 0.0313033 loss)
    I0515 00:44:35.861237 23147 sgd_solver.cpp:106] Iteration 16700, lr = 9.57517e-05
    I0515 00:44:44.002671 23147 solver.cpp:228] Iteration 16800, loss = 0.0382994
    I0515 00:44:44.002768 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:44:44.002790 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0382994 (* 1 = 0.0382994 loss)
    I0515 00:44:44.002805 23147 sgd_solver.cpp:106] Iteration 16800, lr = 9.54836e-05
    I0515 00:44:52.143259 23147 solver.cpp:228] Iteration 16900, loss = 0.0276733
    I0515 00:44:52.143309 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:44:52.143329 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0276733 (* 1 = 0.0276733 loss)
    I0515 00:44:52.143344 23147 sgd_solver.cpp:106] Iteration 16900, lr = 9.52173e-05
    I0515 00:45:00.203158 23147 solver.cpp:337] Iteration 17000, Testing net (#0)
    I0515 00:45:04.596952 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.823167
    I0515 00:45:04.597007 23147 solver.cpp:404]     Test net output #1: loss_c = 0.545696 (* 1 = 0.545696 loss)
    I0515 00:45:04.649507 23147 solver.cpp:228] Iteration 17000, loss = 0.0242608
    I0515 00:45:04.649561 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:45:04.649582 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0242608 (* 1 = 0.0242608 loss)
    I0515 00:45:04.649600 23147 sgd_solver.cpp:106] Iteration 17000, lr = 9.49527e-05
    I0515 00:45:12.745904 23147 solver.cpp:228] Iteration 17100, loss = 0.0134357
    I0515 00:45:12.745965 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:45:12.745995 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0134357 (* 1 = 0.0134357 loss)
    I0515 00:45:12.746016 23147 sgd_solver.cpp:106] Iteration 17100, lr = 9.46898e-05
    I0515 00:45:20.833650 23147 solver.cpp:228] Iteration 17200, loss = 0.0728581
    I0515 00:45:20.833813 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:45:20.833859 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0728581 (* 1 = 0.0728581 loss)
    I0515 00:45:20.833884 23147 sgd_solver.cpp:106] Iteration 17200, lr = 9.44285e-05
    I0515 00:45:28.963819 23147 solver.cpp:228] Iteration 17300, loss = 0.0192044
    I0515 00:45:28.963861 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:45:28.963882 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0192043 (* 1 = 0.0192043 loss)
    I0515 00:45:28.963897 23147 sgd_solver.cpp:106] Iteration 17300, lr = 9.4169e-05
    I0515 00:45:37.104905 23147 solver.cpp:228] Iteration 17400, loss = 0.0250451
    I0515 00:45:37.104956 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:45:37.104980 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0250451 (* 1 = 0.0250451 loss)
    I0515 00:45:37.104995 23147 sgd_solver.cpp:106] Iteration 17400, lr = 9.39111e-05
    I0515 00:45:45.246696 23147 solver.cpp:228] Iteration 17500, loss = 0.0350688
    I0515 00:45:45.246748 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:45:45.246768 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0350688 (* 1 = 0.0350688 loss)
    I0515 00:45:45.246783 23147 sgd_solver.cpp:106] Iteration 17500, lr = 9.36549e-05
    I0515 00:45:53.385367 23147 solver.cpp:228] Iteration 17600, loss = 0.0365827
    I0515 00:45:53.385608 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:45:53.385654 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0365827 (* 1 = 0.0365827 loss)
    I0515 00:45:53.385679 23147 sgd_solver.cpp:106] Iteration 17600, lr = 9.34003e-05
    I0515 00:46:01.527281 23147 solver.cpp:228] Iteration 17700, loss = 0.0306268
    I0515 00:46:01.527328 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:46:01.527348 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0306268 (* 1 = 0.0306268 loss)
    I0515 00:46:01.527362 23147 sgd_solver.cpp:106] Iteration 17700, lr = 9.31473e-05
    I0515 00:46:09.667300 23147 solver.cpp:228] Iteration 17800, loss = 0.0175115
    I0515 00:46:09.667347 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:46:09.667366 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0175115 (* 1 = 0.0175115 loss)
    I0515 00:46:09.667382 23147 sgd_solver.cpp:106] Iteration 17800, lr = 9.28959e-05
    I0515 00:46:17.805879 23147 solver.cpp:228] Iteration 17900, loss = 0.00761407
    I0515 00:46:17.805923 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:46:17.805943 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00761406 (* 1 = 0.00761406 loss)
    I0515 00:46:17.805958 23147 sgd_solver.cpp:106] Iteration 17900, lr = 9.2646e-05
    I0515 00:46:25.862884 23147 solver.cpp:337] Iteration 18000, Testing net (#0)
    I0515 00:46:30.290665 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.82775
    I0515 00:46:30.290714 23147 solver.cpp:404]     Test net output #1: loss_c = 0.571466 (* 1 = 0.571466 loss)
    I0515 00:46:30.346591 23147 solver.cpp:228] Iteration 18000, loss = 0.00628922
    I0515 00:46:30.346657 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:46:30.346685 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0062892 (* 1 = 0.0062892 loss)
    I0515 00:46:30.346710 23147 sgd_solver.cpp:106] Iteration 18000, lr = 9.23978e-05
    I0515 00:46:38.482429 23147 solver.cpp:228] Iteration 18100, loss = 0.0606384
    I0515 00:46:38.482478 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:46:38.482501 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0606384 (* 1 = 0.0606384 loss)
    I0515 00:46:38.482517 23147 sgd_solver.cpp:106] Iteration 18100, lr = 9.2151e-05
    I0515 00:46:46.617554 23147 solver.cpp:228] Iteration 18200, loss = 0.0133653
    I0515 00:46:46.617601 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:46:46.617622 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0133653 (* 1 = 0.0133653 loss)
    I0515 00:46:46.617637 23147 sgd_solver.cpp:106] Iteration 18200, lr = 9.19059e-05
    I0515 00:46:54.752287 23147 solver.cpp:228] Iteration 18300, loss = 0.0122431
    I0515 00:46:54.752334 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:46:54.752354 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0122431 (* 1 = 0.0122431 loss)
    I0515 00:46:54.752369 23147 sgd_solver.cpp:106] Iteration 18300, lr = 9.16622e-05
    I0515 00:47:02.894896 23147 solver.cpp:228] Iteration 18400, loss = 0.0244118
    I0515 00:47:02.895009 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:47:02.895031 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0244118 (* 1 = 0.0244118 loss)
    I0515 00:47:02.895046 23147 sgd_solver.cpp:106] Iteration 18400, lr = 9.142e-05
    I0515 00:47:11.038086 23147 solver.cpp:228] Iteration 18500, loss = 0.0168467
    I0515 00:47:11.038136 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:47:11.038158 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0168467 (* 1 = 0.0168467 loss)
    I0515 00:47:11.038173 23147 sgd_solver.cpp:106] Iteration 18500, lr = 9.11793e-05
    I0515 00:47:19.180961 23147 solver.cpp:228] Iteration 18600, loss = 0.0172773
    I0515 00:47:19.181012 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:47:19.181032 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0172773 (* 1 = 0.0172773 loss)
    I0515 00:47:19.181046 23147 sgd_solver.cpp:106] Iteration 18600, lr = 9.09401e-05
    I0515 00:47:27.322269 23147 solver.cpp:228] Iteration 18700, loss = 0.0401914
    I0515 00:47:27.322312 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:47:27.322335 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0401913 (* 1 = 0.0401913 loss)
    I0515 00:47:27.322350 23147 sgd_solver.cpp:106] Iteration 18700, lr = 9.07024e-05
    I0515 00:47:35.459295 23147 solver.cpp:228] Iteration 18800, loss = 0.0330078
    I0515 00:47:35.459553 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:47:35.459600 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0330077 (* 1 = 0.0330077 loss)
    I0515 00:47:35.459625 23147 sgd_solver.cpp:106] Iteration 18800, lr = 9.0466e-05
    I0515 00:47:43.597378 23147 solver.cpp:228] Iteration 18900, loss = 0.0118151
    I0515 00:47:43.597419 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:47:43.597443 23147 solver.cpp:244]     Train net output #1: loss_c = 0.011815 (* 1 = 0.011815 loss)
    I0515 00:47:43.597458 23147 sgd_solver.cpp:106] Iteration 18900, lr = 9.02312e-05
    I0515 00:47:51.656462 23147 solver.cpp:337] Iteration 19000, Testing net (#0)
    I0515 00:47:56.055980 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.828833
    I0515 00:47:56.056023 23147 solver.cpp:404]     Test net output #1: loss_c = 0.578667 (* 1 = 0.578667 loss)
    I0515 00:47:56.111227 23147 solver.cpp:228] Iteration 19000, loss = 0.0131075
    I0515 00:47:56.111289 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:47:56.111318 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0131074 (* 1 = 0.0131074 loss)
    I0515 00:47:56.111343 23147 sgd_solver.cpp:106] Iteration 19000, lr = 8.99977e-05
    I0515 00:48:04.214128 23147 solver.cpp:228] Iteration 19100, loss = 0.0242665
    I0515 00:48:04.214179 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:48:04.214200 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0242664 (* 1 = 0.0242664 loss)
    I0515 00:48:04.214215 23147 sgd_solver.cpp:106] Iteration 19100, lr = 8.97657e-05
    I0515 00:48:12.352638 23147 solver.cpp:228] Iteration 19200, loss = 0.0375297
    I0515 00:48:12.352753 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:48:12.352783 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0375297 (* 1 = 0.0375297 loss)
    I0515 00:48:12.352805 23147 sgd_solver.cpp:106] Iteration 19200, lr = 8.9535e-05
    I0515 00:48:20.494170 23147 solver.cpp:228] Iteration 19300, loss = 0.0114105
    I0515 00:48:20.494217 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:48:20.494237 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0114104 (* 1 = 0.0114104 loss)
    I0515 00:48:20.494251 23147 sgd_solver.cpp:106] Iteration 19300, lr = 8.93057e-05
    I0515 00:48:28.635649 23147 solver.cpp:228] Iteration 19400, loss = 0.0200308
    I0515 00:48:28.635694 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:48:28.635715 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0200307 (* 1 = 0.0200307 loss)
    I0515 00:48:28.635728 23147 sgd_solver.cpp:106] Iteration 19400, lr = 8.90778e-05
    I0515 00:48:36.775357 23147 solver.cpp:228] Iteration 19500, loss = 0.0168385
    I0515 00:48:36.775404 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:48:36.775424 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0168384 (* 1 = 0.0168384 loss)
    I0515 00:48:36.775439 23147 sgd_solver.cpp:106] Iteration 19500, lr = 8.88512e-05
    I0515 00:48:44.901800 23147 solver.cpp:228] Iteration 19600, loss = 0.0419265
    I0515 00:48:44.902050 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:48:44.902096 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0419265 (* 1 = 0.0419265 loss)
    I0515 00:48:44.902120 23147 sgd_solver.cpp:106] Iteration 19600, lr = 8.8626e-05
    I0515 00:48:53.042037 23147 solver.cpp:228] Iteration 19700, loss = 0.0156689
    I0515 00:48:53.042083 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:48:53.042103 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0156689 (* 1 = 0.0156689 loss)
    I0515 00:48:53.042117 23147 sgd_solver.cpp:106] Iteration 19700, lr = 8.84021e-05
    I0515 00:49:01.176190 23147 solver.cpp:228] Iteration 19800, loss = 0.0156097
    I0515 00:49:01.176232 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:49:01.176251 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0156097 (* 1 = 0.0156097 loss)
    I0515 00:49:01.176266 23147 sgd_solver.cpp:106] Iteration 19800, lr = 8.81795e-05
    I0515 00:49:09.316458 23147 solver.cpp:228] Iteration 19900, loss = 0.0240766
    I0515 00:49:09.316498 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:49:09.316517 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0240766 (* 1 = 0.0240766 loss)
    I0515 00:49:09.316532 23147 sgd_solver.cpp:106] Iteration 19900, lr = 8.79583e-05
    I0515 00:49:17.335325 23147 solver.cpp:337] Iteration 20000, Testing net (#0)
    I0515 00:49:21.739368 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.814
    I0515 00:49:21.739420 23147 solver.cpp:404]     Test net output #1: loss_c = 0.594226 (* 1 = 0.594226 loss)
    I0515 00:49:21.792145 23147 solver.cpp:228] Iteration 20000, loss = 0.0180613
    I0515 00:49:21.792170 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:49:21.792188 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0180613 (* 1 = 0.0180613 loss)
    I0515 00:49:21.792207 23147 sgd_solver.cpp:106] Iteration 20000, lr = 8.77383e-05
    I0515 00:49:29.936725 23147 solver.cpp:228] Iteration 20100, loss = 0.0588164
    I0515 00:49:29.936774 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:49:29.936794 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0588164 (* 1 = 0.0588164 loss)
    I0515 00:49:29.936808 23147 sgd_solver.cpp:106] Iteration 20100, lr = 8.75196e-05
    I0515 00:49:38.076683 23147 solver.cpp:228] Iteration 20200, loss = 0.0185333
    I0515 00:49:38.076736 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:49:38.076756 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0185333 (* 1 = 0.0185333 loss)
    I0515 00:49:38.076772 23147 sgd_solver.cpp:106] Iteration 20200, lr = 8.73021e-05
    I0515 00:49:46.217979 23147 solver.cpp:228] Iteration 20300, loss = 0.0157794
    I0515 00:49:46.218027 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:49:46.218047 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0157794 (* 1 = 0.0157794 loss)
    I0515 00:49:46.218063 23147 sgd_solver.cpp:106] Iteration 20300, lr = 8.70859e-05
    I0515 00:49:54.360013 23147 solver.cpp:228] Iteration 20400, loss = 0.0222302
    I0515 00:49:54.360275 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:49:54.360321 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0222302 (* 1 = 0.0222302 loss)
    I0515 00:49:54.360416 23147 sgd_solver.cpp:106] Iteration 20400, lr = 8.6871e-05
    I0515 00:50:02.501305 23147 solver.cpp:228] Iteration 20500, loss = 0.0615962
    I0515 00:50:02.501356 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 00:50:02.501377 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0615961 (* 1 = 0.0615961 loss)
    I0515 00:50:02.501392 23147 sgd_solver.cpp:106] Iteration 20500, lr = 8.66573e-05
    I0515 00:50:10.633721 23147 solver.cpp:228] Iteration 20600, loss = 0.0270273
    I0515 00:50:10.633771 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:50:10.633795 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0270272 (* 1 = 0.0270272 loss)
    I0515 00:50:10.633810 23147 sgd_solver.cpp:106] Iteration 20600, lr = 8.64448e-05
    I0515 00:50:18.772898 23147 solver.cpp:228] Iteration 20700, loss = 0.0247067
    I0515 00:50:18.772941 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:50:18.772961 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0247067 (* 1 = 0.0247067 loss)
    I0515 00:50:18.772975 23147 sgd_solver.cpp:106] Iteration 20700, lr = 8.62335e-05
    I0515 00:50:26.908249 23147 solver.cpp:228] Iteration 20800, loss = 0.0135354
    I0515 00:50:26.908442 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:50:26.908462 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0135354 (* 1 = 0.0135354 loss)
    I0515 00:50:26.908476 23147 sgd_solver.cpp:106] Iteration 20800, lr = 8.60235e-05
    I0515 00:50:35.006089 23147 solver.cpp:228] Iteration 20900, loss = 0.0230267
    I0515 00:50:35.006146 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:50:35.006175 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0230267 (* 1 = 0.0230267 loss)
    I0515 00:50:35.006196 23147 sgd_solver.cpp:106] Iteration 20900, lr = 8.58146e-05
    I0515 00:50:43.012406 23147 solver.cpp:337] Iteration 21000, Testing net (#0)
    I0515 00:50:47.366817 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.817583
    I0515 00:50:47.366878 23147 solver.cpp:404]     Test net output #1: loss_c = 0.575278 (* 1 = 0.575278 loss)
    I0515 00:50:47.422272 23147 solver.cpp:228] Iteration 21000, loss = 0.00703785
    I0515 00:50:47.422327 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:50:47.422355 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00703778 (* 1 = 0.00703778 loss)
    I0515 00:50:47.422381 23147 sgd_solver.cpp:106] Iteration 21000, lr = 8.56069e-05
    I0515 00:50:55.560691 23147 solver.cpp:228] Iteration 21100, loss = 0.0254488
    I0515 00:50:55.560744 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:50:55.560765 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0254487 (* 1 = 0.0254487 loss)
    I0515 00:50:55.560780 23147 sgd_solver.cpp:106] Iteration 21100, lr = 8.54004e-05
    I0515 00:51:03.656301 23147 solver.cpp:228] Iteration 21200, loss = 0.012013
    I0515 00:51:03.656440 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:51:03.656476 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0120129 (* 1 = 0.0120129 loss)
    I0515 00:51:03.656496 23147 sgd_solver.cpp:106] Iteration 21200, lr = 8.5195e-05
    I0515 00:51:11.797274 23147 solver.cpp:228] Iteration 21300, loss = 0.0188752
    I0515 00:51:11.797314 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:51:11.797338 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0188751 (* 1 = 0.0188751 loss)
    I0515 00:51:11.797353 23147 sgd_solver.cpp:106] Iteration 21300, lr = 8.49908e-05
    I0515 00:51:19.938271 23147 solver.cpp:228] Iteration 21400, loss = 0.0126513
    I0515 00:51:19.938323 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:51:19.938344 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0126513 (* 1 = 0.0126513 loss)
    I0515 00:51:19.938359 23147 sgd_solver.cpp:106] Iteration 21400, lr = 8.47877e-05
    I0515 00:51:28.080008 23147 solver.cpp:228] Iteration 21500, loss = 0.0361349
    I0515 00:51:28.080049 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:51:28.080070 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0361348 (* 1 = 0.0361348 loss)
    I0515 00:51:28.080083 23147 sgd_solver.cpp:106] Iteration 21500, lr = 8.45857e-05
    I0515 00:51:36.189741 23147 solver.cpp:228] Iteration 21600, loss = 0.0134097
    I0515 00:51:36.189878 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:51:36.189924 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0134096 (* 1 = 0.0134096 loss)
    I0515 00:51:36.189949 23147 sgd_solver.cpp:106] Iteration 21600, lr = 8.43849e-05
    I0515 00:51:44.335175 23147 solver.cpp:228] Iteration 21700, loss = 0.0377451
    I0515 00:51:44.335222 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:51:44.335242 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0377451 (* 1 = 0.0377451 loss)
    I0515 00:51:44.335260 23147 sgd_solver.cpp:106] Iteration 21700, lr = 8.41852e-05
    I0515 00:51:52.477237 23147 solver.cpp:228] Iteration 21800, loss = 0.0178802
    I0515 00:51:52.477289 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:51:52.477311 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0178802 (* 1 = 0.0178802 loss)
    I0515 00:51:52.477326 23147 sgd_solver.cpp:106] Iteration 21800, lr = 8.39865e-05
    I0515 00:52:00.619767 23147 solver.cpp:228] Iteration 21900, loss = 0.0478538
    I0515 00:52:00.619817 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:52:00.619838 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0478537 (* 1 = 0.0478537 loss)
    I0515 00:52:00.619853 23147 sgd_solver.cpp:106] Iteration 21900, lr = 8.3789e-05
    I0515 00:52:08.680595 23147 solver.cpp:337] Iteration 22000, Testing net (#0)
    I0515 00:52:13.086330 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.811
    I0515 00:52:13.086382 23147 solver.cpp:404]     Test net output #1: loss_c = 0.619093 (* 1 = 0.619093 loss)
    I0515 00:52:13.138691 23147 solver.cpp:228] Iteration 22000, loss = 0.0257349
    I0515 00:52:13.138754 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:52:13.138778 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0257348 (* 1 = 0.0257348 loss)
    I0515 00:52:13.138797 23147 sgd_solver.cpp:106] Iteration 22000, lr = 8.35925e-05
    I0515 00:52:21.233553 23147 solver.cpp:228] Iteration 22100, loss = 0.0131375
    I0515 00:52:21.233603 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:52:21.233625 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0131374 (* 1 = 0.0131374 loss)
    I0515 00:52:21.233640 23147 sgd_solver.cpp:106] Iteration 22100, lr = 8.33971e-05
    I0515 00:52:29.374547 23147 solver.cpp:228] Iteration 22200, loss = 0.0122873
    I0515 00:52:29.374594 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:52:29.374614 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0122873 (* 1 = 0.0122873 loss)
    I0515 00:52:29.374629 23147 sgd_solver.cpp:106] Iteration 22200, lr = 8.32028e-05
    I0515 00:52:37.514936 23147 solver.cpp:228] Iteration 22300, loss = 0.0314735
    I0515 00:52:37.514987 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:52:37.515008 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0314735 (* 1 = 0.0314735 loss)
    I0515 00:52:37.515023 23147 sgd_solver.cpp:106] Iteration 22300, lr = 8.30096e-05
    I0515 00:52:45.656507 23147 solver.cpp:228] Iteration 22400, loss = 0.0169233
    I0515 00:52:45.656652 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:52:45.656697 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0169233 (* 1 = 0.0169233 loss)
    I0515 00:52:45.656721 23147 sgd_solver.cpp:106] Iteration 22400, lr = 8.28173e-05
    I0515 00:52:53.793527 23147 solver.cpp:228] Iteration 22500, loss = 0.0180376
    I0515 00:52:53.793576 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:52:53.793596 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0180376 (* 1 = 0.0180376 loss)
    I0515 00:52:53.793612 23147 sgd_solver.cpp:106] Iteration 22500, lr = 8.26261e-05
    I0515 00:53:01.928891 23147 solver.cpp:228] Iteration 22600, loss = 0.0510838
    I0515 00:53:01.928937 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:53:01.928959 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0510837 (* 1 = 0.0510837 loss)
    I0515 00:53:01.928974 23147 sgd_solver.cpp:106] Iteration 22600, lr = 8.2436e-05
    I0515 00:53:10.067417 23147 solver.cpp:228] Iteration 22700, loss = 0.0166243
    I0515 00:53:10.067469 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:53:10.067489 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0166243 (* 1 = 0.0166243 loss)
    I0515 00:53:10.067508 23147 sgd_solver.cpp:106] Iteration 22700, lr = 8.22468e-05
    I0515 00:53:18.209187 23147 solver.cpp:228] Iteration 22800, loss = 0.0134358
    I0515 00:53:18.209437 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:53:18.209483 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0134358 (* 1 = 0.0134358 loss)
    I0515 00:53:18.209508 23147 sgd_solver.cpp:106] Iteration 22800, lr = 8.20587e-05
    I0515 00:53:26.315022 23147 solver.cpp:228] Iteration 22900, loss = 0.0125993
    I0515 00:53:26.315075 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:53:26.315104 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0125992 (* 1 = 0.0125992 loss)
    I0515 00:53:26.315125 23147 sgd_solver.cpp:106] Iteration 22900, lr = 8.18716e-05
    I0515 00:53:34.320519 23147 solver.cpp:337] Iteration 23000, Testing net (#0)
    I0515 00:53:38.697077 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.80975
    I0515 00:53:38.697132 23147 solver.cpp:404]     Test net output #1: loss_c = 0.602612 (* 1 = 0.602612 loss)
    I0515 00:53:38.749747 23147 solver.cpp:228] Iteration 23000, loss = 0.0290792
    I0515 00:53:38.749794 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:53:38.749815 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0290791 (* 1 = 0.0290791 loss)
    I0515 00:53:38.749837 23147 sgd_solver.cpp:106] Iteration 23000, lr = 8.16854e-05
    I0515 00:53:46.868305 23147 solver.cpp:228] Iteration 23100, loss = 0.0473904
    I0515 00:53:46.868352 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:53:46.868371 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0473904 (* 1 = 0.0473904 loss)
    I0515 00:53:46.868386 23147 sgd_solver.cpp:106] Iteration 23100, lr = 8.15003e-05
    I0515 00:53:55.009608 23147 solver.cpp:228] Iteration 23200, loss = 0.0136496
    I0515 00:53:55.009685 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:53:55.009706 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0136495 (* 1 = 0.0136495 loss)
    I0515 00:53:55.009721 23147 sgd_solver.cpp:106] Iteration 23200, lr = 8.13161e-05
    I0515 00:54:03.150037 23147 solver.cpp:228] Iteration 23300, loss = 0.0491961
    I0515 00:54:03.150087 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:54:03.150107 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0491961 (* 1 = 0.0491961 loss)
    I0515 00:54:03.150122 23147 sgd_solver.cpp:106] Iteration 23300, lr = 8.11329e-05
    I0515 00:54:11.291621 23147 solver.cpp:228] Iteration 23400, loss = 0.0250394
    I0515 00:54:11.291673 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:54:11.291693 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0250393 (* 1 = 0.0250393 loss)
    I0515 00:54:11.291708 23147 sgd_solver.cpp:106] Iteration 23400, lr = 8.09506e-05
    I0515 00:54:19.433437 23147 solver.cpp:228] Iteration 23500, loss = 0.0355402
    I0515 00:54:19.433490 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:54:19.433511 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0355402 (* 1 = 0.0355402 loss)
    I0515 00:54:19.433526 23147 sgd_solver.cpp:106] Iteration 23500, lr = 8.07693e-05
    I0515 00:54:27.573793 23147 solver.cpp:228] Iteration 23600, loss = 0.0125216
    I0515 00:54:27.573931 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:54:27.573977 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0125216 (* 1 = 0.0125216 loss)
    I0515 00:54:27.574002 23147 sgd_solver.cpp:106] Iteration 23600, lr = 8.0589e-05
    I0515 00:54:35.674636 23147 solver.cpp:228] Iteration 23700, loss = 0.0354034
    I0515 00:54:35.674680 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:54:35.674703 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0354034 (* 1 = 0.0354034 loss)
    I0515 00:54:35.674722 23147 sgd_solver.cpp:106] Iteration 23700, lr = 8.04095e-05
    I0515 00:54:43.816182 23147 solver.cpp:228] Iteration 23800, loss = 0.0307556
    I0515 00:54:43.816234 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:54:43.816254 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0307556 (* 1 = 0.0307556 loss)
    I0515 00:54:43.816269 23147 sgd_solver.cpp:106] Iteration 23800, lr = 8.0231e-05
    I0515 00:54:51.956436 23147 solver.cpp:228] Iteration 23900, loss = 0.019276
    I0515 00:54:51.956477 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:54:51.956497 23147 solver.cpp:244]     Train net output #1: loss_c = 0.019276 (* 1 = 0.019276 loss)
    I0515 00:54:51.956512 23147 sgd_solver.cpp:106] Iteration 23900, lr = 8.00535e-05
    I0515 00:55:00.016621 23147 solver.cpp:337] Iteration 24000, Testing net (#0)
    I0515 00:55:04.425638 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.823333
    I0515 00:55:04.425685 23147 solver.cpp:404]     Test net output #1: loss_c = 0.606009 (* 1 = 0.606009 loss)
    I0515 00:55:04.480870 23147 solver.cpp:228] Iteration 24000, loss = 0.0435108
    I0515 00:55:04.480932 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:55:04.480962 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0435108 (* 1 = 0.0435108 loss)
    I0515 00:55:04.480988 23147 sgd_solver.cpp:106] Iteration 24000, lr = 7.98768e-05
    I0515 00:55:12.606402 23147 solver.cpp:228] Iteration 24100, loss = 0.012891
    I0515 00:55:12.606452 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:55:12.606477 23147 solver.cpp:244]     Train net output #1: loss_c = 0.012891 (* 1 = 0.012891 loss)
    I0515 00:55:12.606492 23147 sgd_solver.cpp:106] Iteration 24100, lr = 7.97011e-05
    I0515 00:55:20.749804 23147 solver.cpp:228] Iteration 24200, loss = 0.0283842
    I0515 00:55:20.749864 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:55:20.749893 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0283841 (* 1 = 0.0283841 loss)
    I0515 00:55:20.749917 23147 sgd_solver.cpp:106] Iteration 24200, lr = 7.95262e-05
    I0515 00:55:28.874480 23147 solver.cpp:228] Iteration 24300, loss = 0.00722313
    I0515 00:55:28.874524 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:55:28.874544 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00722309 (* 1 = 0.00722309 loss)
    I0515 00:55:28.874558 23147 sgd_solver.cpp:106] Iteration 24300, lr = 7.93523e-05
    I0515 00:55:37.014462 23147 solver.cpp:228] Iteration 24400, loss = 0.0147493
    I0515 00:55:37.014600 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:55:37.014647 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0147493 (* 1 = 0.0147493 loss)
    I0515 00:55:37.014672 23147 sgd_solver.cpp:106] Iteration 24400, lr = 7.91792e-05
    I0515 00:55:45.154819 23147 solver.cpp:228] Iteration 24500, loss = 0.0275284
    I0515 00:55:45.154868 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:55:45.154888 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0275284 (* 1 = 0.0275284 loss)
    I0515 00:55:45.154903 23147 sgd_solver.cpp:106] Iteration 24500, lr = 7.9007e-05
    I0515 00:55:53.295802 23147 solver.cpp:228] Iteration 24600, loss = 0.0194907
    I0515 00:55:53.295846 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:55:53.295866 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0194906 (* 1 = 0.0194906 loss)
    I0515 00:55:53.295881 23147 sgd_solver.cpp:106] Iteration 24600, lr = 7.88357e-05
    I0515 00:56:01.436074 23147 solver.cpp:228] Iteration 24700, loss = 0.0220351
    I0515 00:56:01.436117 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:56:01.436136 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0220351 (* 1 = 0.0220351 loss)
    I0515 00:56:01.436151 23147 sgd_solver.cpp:106] Iteration 24700, lr = 7.86652e-05
    I0515 00:56:09.577277 23147 solver.cpp:228] Iteration 24800, loss = 0.0128343
    I0515 00:56:09.577458 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:56:09.577498 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0128342 (* 1 = 0.0128342 loss)
    I0515 00:56:09.577514 23147 sgd_solver.cpp:106] Iteration 24800, lr = 7.84956e-05
    I0515 00:56:17.718441 23147 solver.cpp:228] Iteration 24900, loss = 0.0147168
    I0515 00:56:17.718489 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:56:17.718509 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0147167 (* 1 = 0.0147167 loss)
    I0515 00:56:17.718524 23147 sgd_solver.cpp:106] Iteration 24900, lr = 7.83269e-05
    I0515 00:56:25.779745 23147 solver.cpp:337] Iteration 25000, Testing net (#0)
    I0515 00:56:30.197535 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.826083
    I0515 00:56:30.197592 23147 solver.cpp:404]     Test net output #1: loss_c = 0.533014 (* 1 = 0.533014 loss)
    I0515 00:56:30.249835 23147 solver.cpp:228] Iteration 25000, loss = 0.0149884
    I0515 00:56:30.249858 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:56:30.249878 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0149884 (* 1 = 0.0149884 loss)
    I0515 00:56:30.249897 23147 sgd_solver.cpp:106] Iteration 25000, lr = 7.8159e-05
    I0515 00:56:38.386543 23147 solver.cpp:228] Iteration 25100, loss = 0.0338159
    I0515 00:56:38.386591 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:56:38.386612 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0338158 (* 1 = 0.0338158 loss)
    I0515 00:56:38.386626 23147 sgd_solver.cpp:106] Iteration 25100, lr = 7.79919e-05
    I0515 00:56:46.523800 23147 solver.cpp:228] Iteration 25200, loss = 0.019813
    I0515 00:56:46.523969 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:56:46.524015 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0198129 (* 1 = 0.0198129 loss)
    I0515 00:56:46.524039 23147 sgd_solver.cpp:106] Iteration 25200, lr = 7.78257e-05
    I0515 00:56:54.639788 23147 solver.cpp:228] Iteration 25300, loss = 0.0259382
    I0515 00:56:54.639837 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:56:54.639858 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0259381 (* 1 = 0.0259381 loss)
    I0515 00:56:54.639873 23147 sgd_solver.cpp:106] Iteration 25300, lr = 7.76603e-05
    I0515 00:57:02.769743 23147 solver.cpp:228] Iteration 25400, loss = 0.0199228
    I0515 00:57:02.769794 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:57:02.769816 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0199227 (* 1 = 0.0199227 loss)
    I0515 00:57:02.769830 23147 sgd_solver.cpp:106] Iteration 25400, lr = 7.74957e-05
    I0515 00:57:10.900730 23147 solver.cpp:228] Iteration 25500, loss = 0.0299877
    I0515 00:57:10.900779 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:57:10.900799 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0299876 (* 1 = 0.0299876 loss)
    I0515 00:57:10.900813 23147 sgd_solver.cpp:106] Iteration 25500, lr = 7.73319e-05
    I0515 00:57:19.041474 23147 solver.cpp:228] Iteration 25600, loss = 0.0118424
    I0515 00:57:19.041576 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:57:19.041597 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0118423 (* 1 = 0.0118423 loss)
    I0515 00:57:19.041612 23147 sgd_solver.cpp:106] Iteration 25600, lr = 7.71689e-05
    I0515 00:57:27.179983 23147 solver.cpp:228] Iteration 25700, loss = 0.0130723
    I0515 00:57:27.180023 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:57:27.180044 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0130722 (* 1 = 0.0130722 loss)
    I0515 00:57:27.180059 23147 sgd_solver.cpp:106] Iteration 25700, lr = 7.70068e-05
    I0515 00:57:35.320951 23147 solver.cpp:228] Iteration 25800, loss = 0.0139391
    I0515 00:57:35.321002 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:57:35.321023 23147 solver.cpp:244]     Train net output #1: loss_c = 0.013939 (* 1 = 0.013939 loss)
    I0515 00:57:35.321038 23147 sgd_solver.cpp:106] Iteration 25800, lr = 7.68454e-05
    I0515 00:57:43.463264 23147 solver.cpp:228] Iteration 25900, loss = 0.00576069
    I0515 00:57:43.463317 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:57:43.463337 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00576061 (* 1 = 0.00576061 loss)
    I0515 00:57:43.463351 23147 sgd_solver.cpp:106] Iteration 25900, lr = 7.66848e-05
    I0515 00:57:51.524242 23147 solver.cpp:337] Iteration 26000, Testing net (#0)
    I0515 00:57:55.910481 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.835916
    I0515 00:57:55.910526 23147 solver.cpp:404]     Test net output #1: loss_c = 0.536 (* 1 = 0.536 loss)
    I0515 00:57:55.966447 23147 solver.cpp:228] Iteration 26000, loss = 0.0244103
    I0515 00:57:55.966517 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:57:55.966550 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0244102 (* 1 = 0.0244102 loss)
    I0515 00:57:55.966574 23147 sgd_solver.cpp:106] Iteration 26000, lr = 7.6525e-05
    I0515 00:58:04.104676 23147 solver.cpp:228] Iteration 26100, loss = 0.0214793
    I0515 00:58:04.104727 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:58:04.104748 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0214792 (* 1 = 0.0214792 loss)
    I0515 00:58:04.104761 23147 sgd_solver.cpp:106] Iteration 26100, lr = 7.63659e-05
    I0515 00:58:12.245491 23147 solver.cpp:228] Iteration 26200, loss = 0.0588318
    I0515 00:58:12.245537 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:58:12.245558 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0588317 (* 1 = 0.0588317 loss)
    I0515 00:58:12.245573 23147 sgd_solver.cpp:106] Iteration 26200, lr = 7.62077e-05
    I0515 00:58:20.387375 23147 solver.cpp:228] Iteration 26300, loss = 0.0447833
    I0515 00:58:20.387428 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:58:20.387447 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0447832 (* 1 = 0.0447832 loss)
    I0515 00:58:20.387462 23147 sgd_solver.cpp:106] Iteration 26300, lr = 7.60501e-05
    I0515 00:58:28.529960 23147 solver.cpp:228] Iteration 26400, loss = 0.0315579
    I0515 00:58:28.530122 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:58:28.530167 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0315578 (* 1 = 0.0315578 loss)
    I0515 00:58:28.530192 23147 sgd_solver.cpp:106] Iteration 26400, lr = 7.58934e-05
    I0515 00:58:36.642962 23147 solver.cpp:228] Iteration 26500, loss = 0.00648393
    I0515 00:58:36.643079 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:58:36.643100 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00648385 (* 1 = 0.00648385 loss)
    I0515 00:58:36.643115 23147 sgd_solver.cpp:106] Iteration 26500, lr = 7.57374e-05
    I0515 00:58:44.783298 23147 solver.cpp:228] Iteration 26600, loss = 0.0337267
    I0515 00:58:44.783341 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:58:44.783361 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0337267 (* 1 = 0.0337267 loss)
    I0515 00:58:44.783376 23147 sgd_solver.cpp:106] Iteration 26600, lr = 7.55821e-05
    I0515 00:58:52.919522 23147 solver.cpp:228] Iteration 26700, loss = 0.0261345
    I0515 00:58:52.919582 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:58:52.919611 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0261344 (* 1 = 0.0261344 loss)
    I0515 00:58:52.919632 23147 sgd_solver.cpp:106] Iteration 26700, lr = 7.54276e-05
    I0515 00:59:01.013624 23147 solver.cpp:228] Iteration 26800, loss = 0.0210148
    I0515 00:59:01.013744 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:59:01.013775 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0210147 (* 1 = 0.0210147 loss)
    I0515 00:59:01.013797 23147 sgd_solver.cpp:106] Iteration 26800, lr = 7.52738e-05
    I0515 00:59:09.098902 23147 solver.cpp:228] Iteration 26900, loss = 0.0107678
    I0515 00:59:09.098963 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:59:09.098991 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0107677 (* 1 = 0.0107677 loss)
    I0515 00:59:09.099014 23147 sgd_solver.cpp:106] Iteration 26900, lr = 7.51208e-05
    I0515 00:59:17.105226 23147 solver.cpp:337] Iteration 27000, Testing net (#0)
    I0515 00:59:21.495106 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.836083
    I0515 00:59:21.495152 23147 solver.cpp:404]     Test net output #1: loss_c = 0.53014 (* 1 = 0.53014 loss)
    I0515 00:59:21.550184 23147 solver.cpp:228] Iteration 27000, loss = 0.022103
    I0515 00:59:21.550245 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 00:59:21.550274 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0221029 (* 1 = 0.0221029 loss)
    I0515 00:59:21.550299 23147 sgd_solver.cpp:106] Iteration 27000, lr = 7.49685e-05
    I0515 00:59:29.685953 23147 solver.cpp:228] Iteration 27100, loss = 0.00968795
    I0515 00:59:29.686005 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:59:29.686025 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00968786 (* 1 = 0.00968786 loss)
    I0515 00:59:29.686040 23147 sgd_solver.cpp:106] Iteration 27100, lr = 7.48169e-05
    I0515 00:59:37.828544 23147 solver.cpp:228] Iteration 27200, loss = 0.0509624
    I0515 00:59:37.828732 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 00:59:37.828752 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0509623 (* 1 = 0.0509623 loss)
    I0515 00:59:37.828799 23147 sgd_solver.cpp:106] Iteration 27200, lr = 7.4666e-05
    I0515 00:59:45.968555 23147 solver.cpp:228] Iteration 27300, loss = 0.00630431
    I0515 00:59:45.968596 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:59:45.968619 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00630422 (* 1 = 0.00630422 loss)
    I0515 00:59:45.968634 23147 sgd_solver.cpp:106] Iteration 27300, lr = 7.45158e-05
    I0515 00:59:54.098857 23147 solver.cpp:228] Iteration 27400, loss = 0.0198992
    I0515 00:59:54.098912 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 00:59:54.098932 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0198991 (* 1 = 0.0198991 loss)
    I0515 00:59:54.098945 23147 sgd_solver.cpp:106] Iteration 27400, lr = 7.43663e-05
    I0515 01:00:02.239058 23147 solver.cpp:228] Iteration 27500, loss = 0.0330923
    I0515 01:00:02.239104 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 01:00:02.239127 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0330922 (* 1 = 0.0330922 loss)
    I0515 01:00:02.239143 23147 sgd_solver.cpp:106] Iteration 27500, lr = 7.42175e-05
    I0515 01:00:10.380092 23147 solver.cpp:228] Iteration 27600, loss = 0.0666777
    I0515 01:00:10.380179 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:00:10.380205 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0666776 (* 1 = 0.0666776 loss)
    I0515 01:00:10.380221 23147 sgd_solver.cpp:106] Iteration 27600, lr = 7.40694e-05
    I0515 01:00:18.435259 23147 solver.cpp:228] Iteration 27700, loss = 0.0246491
    I0515 01:00:18.435303 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:00:18.435323 23147 solver.cpp:244]     Train net output #1: loss_c = 0.024649 (* 1 = 0.024649 loss)
    I0515 01:00:18.435338 23147 sgd_solver.cpp:106] Iteration 27700, lr = 7.3922e-05
    I0515 01:00:26.572444 23147 solver.cpp:228] Iteration 27800, loss = 0.0254486
    I0515 01:00:26.572491 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:00:26.572515 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0254485 (* 1 = 0.0254485 loss)
    I0515 01:00:26.572530 23147 sgd_solver.cpp:106] Iteration 27800, lr = 7.37753e-05
    I0515 01:00:34.713388 23147 solver.cpp:228] Iteration 27900, loss = 0.0268264
    I0515 01:00:34.713438 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:00:34.713459 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0268264 (* 1 = 0.0268264 loss)
    I0515 01:00:34.713474 23147 sgd_solver.cpp:106] Iteration 27900, lr = 7.36293e-05
    I0515 01:00:42.774727 23147 solver.cpp:337] Iteration 28000, Testing net (#0)
    I0515 01:00:47.137065 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.82125
    I0515 01:00:47.137132 23147 solver.cpp:404]     Test net output #1: loss_c = 0.570166 (* 1 = 0.570166 loss)
    I0515 01:00:47.193238 23147 solver.cpp:228] Iteration 28000, loss = 0.0228017
    I0515 01:00:47.193285 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:00:47.193315 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0228016 (* 1 = 0.0228016 loss)
    I0515 01:00:47.193341 23147 sgd_solver.cpp:106] Iteration 28000, lr = 7.34839e-05
    I0515 01:00:55.284798 23147 solver.cpp:228] Iteration 28100, loss = 0.0383637
    I0515 01:00:55.284849 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:00:55.284878 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0383636 (* 1 = 0.0383636 loss)
    I0515 01:00:55.284899 23147 sgd_solver.cpp:106] Iteration 28100, lr = 7.33392e-05
    I0515 01:01:03.375885 23147 solver.cpp:228] Iteration 28200, loss = 0.0308465
    I0515 01:01:03.375944 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:01:03.375973 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0308464 (* 1 = 0.0308464 loss)
    I0515 01:01:03.375995 23147 sgd_solver.cpp:106] Iteration 28200, lr = 7.31952e-05
    I0515 01:01:11.465538 23147 solver.cpp:228] Iteration 28300, loss = 0.00443188
    I0515 01:01:11.465598 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:01:11.465628 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0044318 (* 1 = 0.0044318 loss)
    I0515 01:01:11.465649 23147 sgd_solver.cpp:106] Iteration 28300, lr = 7.30518e-05
    I0515 01:01:19.554023 23147 solver.cpp:228] Iteration 28400, loss = 0.00573112
    I0515 01:01:19.554163 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:01:19.554208 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00573105 (* 1 = 0.00573105 loss)
    I0515 01:01:19.554234 23147 sgd_solver.cpp:106] Iteration 28400, lr = 7.29091e-05
    I0515 01:01:27.648690 23147 solver.cpp:228] Iteration 28500, loss = 0.0163559
    I0515 01:01:27.648754 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:01:27.648783 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0163558 (* 1 = 0.0163558 loss)
    I0515 01:01:27.648805 23147 sgd_solver.cpp:106] Iteration 28500, lr = 7.2767e-05
    I0515 01:01:35.733919 23147 solver.cpp:228] Iteration 28600, loss = 0.0213082
    I0515 01:01:35.733979 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:01:35.734009 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0213082 (* 1 = 0.0213082 loss)
    I0515 01:01:35.734030 23147 sgd_solver.cpp:106] Iteration 28600, lr = 7.26256e-05
    I0515 01:01:43.825947 23147 solver.cpp:228] Iteration 28700, loss = 0.0264742
    I0515 01:01:43.826007 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:01:43.826036 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0264741 (* 1 = 0.0264741 loss)
    I0515 01:01:43.826058 23147 sgd_solver.cpp:106] Iteration 28700, lr = 7.24848e-05
    I0515 01:01:51.914167 23147 solver.cpp:228] Iteration 28800, loss = 0.0432869
    I0515 01:01:51.914281 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:01:51.914311 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0432869 (* 1 = 0.0432869 loss)
    I0515 01:01:51.914333 23147 sgd_solver.cpp:106] Iteration 28800, lr = 7.23446e-05
    I0515 01:02:00.002352 23147 solver.cpp:228] Iteration 28900, loss = 0.0163187
    I0515 01:02:00.002423 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:02:00.002454 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0163186 (* 1 = 0.0163186 loss)
    I0515 01:02:00.002478 23147 sgd_solver.cpp:106] Iteration 28900, lr = 7.22051e-05
    I0515 01:02:08.009243 23147 solver.cpp:337] Iteration 29000, Testing net (#0)
    I0515 01:02:12.428354 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.82525
    I0515 01:02:12.428416 23147 solver.cpp:404]     Test net output #1: loss_c = 0.569284 (* 1 = 0.569284 loss)
    I0515 01:02:12.480175 23147 solver.cpp:228] Iteration 29000, loss = 0.0261129
    I0515 01:02:12.480226 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:02:12.480245 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0261129 (* 1 = 0.0261129 loss)
    I0515 01:02:12.480268 23147 sgd_solver.cpp:106] Iteration 29000, lr = 7.20662e-05
    I0515 01:02:20.571758 23147 solver.cpp:228] Iteration 29100, loss = 0.00724283
    I0515 01:02:20.571820 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:02:20.571848 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00724278 (* 1 = 0.00724278 loss)
    I0515 01:02:20.571869 23147 sgd_solver.cpp:106] Iteration 29100, lr = 7.19279e-05
    I0515 01:02:28.657521 23147 solver.cpp:228] Iteration 29200, loss = 0.0454568
    I0515 01:02:28.657753 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:02:28.657801 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0454567 (* 1 = 0.0454567 loss)
    I0515 01:02:28.657826 23147 sgd_solver.cpp:106] Iteration 29200, lr = 7.17902e-05
    I0515 01:02:36.751334 23147 solver.cpp:228] Iteration 29300, loss = 0.036706
    I0515 01:02:36.751395 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:02:36.751425 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0367059 (* 1 = 0.0367059 loss)
    I0515 01:02:36.751446 23147 sgd_solver.cpp:106] Iteration 29300, lr = 7.16532e-05
    I0515 01:02:44.838743 23147 solver.cpp:228] Iteration 29400, loss = 0.0399827
    I0515 01:02:44.838807 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:02:44.838836 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0399826 (* 1 = 0.0399826 loss)
    I0515 01:02:44.838860 23147 sgd_solver.cpp:106] Iteration 29400, lr = 7.15168e-05
    I0515 01:02:52.926928 23147 solver.cpp:228] Iteration 29500, loss = 0.0149497
    I0515 01:02:52.926988 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:02:52.927016 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0149496 (* 1 = 0.0149496 loss)
    I0515 01:02:52.927038 23147 sgd_solver.cpp:106] Iteration 29500, lr = 7.13809e-05
    I0515 01:03:01.015439 23147 solver.cpp:228] Iteration 29600, loss = 0.0498535
    I0515 01:03:01.015604 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 01:03:01.015650 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0498534 (* 1 = 0.0498534 loss)
    I0515 01:03:01.015674 23147 sgd_solver.cpp:106] Iteration 29600, lr = 7.12457e-05
    I0515 01:03:09.127976 23147 solver.cpp:228] Iteration 29700, loss = 0.029525
    I0515 01:03:09.128033 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:03:09.128063 23147 solver.cpp:244]     Train net output #1: loss_c = 0.029525 (* 1 = 0.029525 loss)
    I0515 01:03:09.128087 23147 sgd_solver.cpp:106] Iteration 29700, lr = 7.11111e-05
    I0515 01:03:17.214951 23147 solver.cpp:228] Iteration 29800, loss = 0.0285739
    I0515 01:03:17.215013 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:03:17.215041 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0285738 (* 1 = 0.0285738 loss)
    I0515 01:03:17.215064 23147 sgd_solver.cpp:106] Iteration 29800, lr = 7.0977e-05
    I0515 01:03:25.304556 23147 solver.cpp:228] Iteration 29900, loss = 0.0261595
    I0515 01:03:25.304610 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:03:25.304638 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0261594 (* 1 = 0.0261594 loss)
    I0515 01:03:25.304661 23147 sgd_solver.cpp:106] Iteration 29900, lr = 7.08435e-05
    I0515 01:03:33.311174 23147 solver.cpp:337] Iteration 30000, Testing net (#0)
    I0515 01:03:37.668823 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.825083
    I0515 01:03:37.668889 23147 solver.cpp:404]     Test net output #1: loss_c = 0.542402 (* 1 = 0.542402 loss)
    I0515 01:03:37.724035 23147 solver.cpp:228] Iteration 30000, loss = 0.0507635
    I0515 01:03:37.724097 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:03:37.724128 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0507634 (* 1 = 0.0507634 loss)
    I0515 01:03:37.724153 23147 sgd_solver.cpp:106] Iteration 30000, lr = 7.07107e-05
    I0515 01:03:45.863032 23147 solver.cpp:228] Iteration 30100, loss = 0.0284943
    I0515 01:03:45.863080 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:03:45.863102 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0284942 (* 1 = 0.0284942 loss)
    I0515 01:03:45.863116 23147 sgd_solver.cpp:106] Iteration 30100, lr = 7.05784e-05
    I0515 01:03:54.003374 23147 solver.cpp:228] Iteration 30200, loss = 0.0222054
    I0515 01:03:54.003422 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:03:54.003443 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0222053 (* 1 = 0.0222053 loss)
    I0515 01:03:54.003456 23147 sgd_solver.cpp:106] Iteration 30200, lr = 7.04467e-05
    I0515 01:04:02.142496 23147 solver.cpp:228] Iteration 30300, loss = 0.0197905
    I0515 01:04:02.142546 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:04:02.142566 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0197905 (* 1 = 0.0197905 loss)
    I0515 01:04:02.142581 23147 sgd_solver.cpp:106] Iteration 30300, lr = 7.03155e-05
    I0515 01:04:10.276080 23147 solver.cpp:228] Iteration 30400, loss = 0.00872509
    I0515 01:04:10.276257 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:04:10.276278 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00872506 (* 1 = 0.00872506 loss)
    I0515 01:04:10.276294 23147 sgd_solver.cpp:106] Iteration 30400, lr = 7.01849e-05
    I0515 01:04:18.416972 23147 solver.cpp:228] Iteration 30500, loss = 0.0177805
    I0515 01:04:18.417023 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:04:18.417043 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0177805 (* 1 = 0.0177805 loss)
    I0515 01:04:18.417058 23147 sgd_solver.cpp:106] Iteration 30500, lr = 7.00549e-05
    I0515 01:04:26.555255 23147 solver.cpp:228] Iteration 30600, loss = 0.0212153
    I0515 01:04:26.555305 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:04:26.555325 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0212153 (* 1 = 0.0212153 loss)
    I0515 01:04:26.555340 23147 sgd_solver.cpp:106] Iteration 30600, lr = 6.99255e-05
    I0515 01:04:34.698302 23147 solver.cpp:228] Iteration 30700, loss = 0.0368919
    I0515 01:04:34.698354 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:04:34.698374 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0368919 (* 1 = 0.0368919 loss)
    I0515 01:04:34.698388 23147 sgd_solver.cpp:106] Iteration 30700, lr = 6.97966e-05
    I0515 01:04:42.840852 23147 solver.cpp:228] Iteration 30800, loss = 0.00990242
    I0515 01:04:42.840945 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:04:42.840970 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0099024 (* 1 = 0.0099024 loss)
    I0515 01:04:42.840983 23147 sgd_solver.cpp:106] Iteration 30800, lr = 6.96682e-05
    I0515 01:04:50.981557 23147 solver.cpp:228] Iteration 30900, loss = 0.0366065
    I0515 01:04:50.981609 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:04:50.981629 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0366064 (* 1 = 0.0366064 loss)
    I0515 01:04:50.981644 23147 sgd_solver.cpp:106] Iteration 30900, lr = 6.95405e-05
    I0515 01:04:59.043090 23147 solver.cpp:337] Iteration 31000, Testing net (#0)
    I0515 01:05:03.411659 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.821833
    I0515 01:05:03.411725 23147 solver.cpp:404]     Test net output #1: loss_c = 0.563599 (* 1 = 0.563599 loss)
    I0515 01:05:03.466784 23147 solver.cpp:228] Iteration 31000, loss = 0.015881
    I0515 01:05:03.466817 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:05:03.466845 23147 solver.cpp:244]     Train net output #1: loss_c = 0.015881 (* 1 = 0.015881 loss)
    I0515 01:05:03.466871 23147 sgd_solver.cpp:106] Iteration 31000, lr = 6.94132e-05
    I0515 01:05:11.555142 23147 solver.cpp:228] Iteration 31100, loss = 0.0181753
    I0515 01:05:11.555194 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:05:11.555222 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0181752 (* 1 = 0.0181752 loss)
    I0515 01:05:11.555243 23147 sgd_solver.cpp:106] Iteration 31100, lr = 6.92865e-05
    I0515 01:05:19.642823 23147 solver.cpp:228] Iteration 31200, loss = 0.0113722
    I0515 01:05:19.643052 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:05:19.643100 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0113722 (* 1 = 0.0113722 loss)
    I0515 01:05:19.643124 23147 sgd_solver.cpp:106] Iteration 31200, lr = 6.91603e-05
    I0515 01:05:27.747910 23147 solver.cpp:228] Iteration 31300, loss = 0.0129263
    I0515 01:05:27.747962 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:05:27.747992 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0129263 (* 1 = 0.0129263 loss)
    I0515 01:05:27.748013 23147 sgd_solver.cpp:106] Iteration 31300, lr = 6.90347e-05
    I0515 01:05:35.836670 23147 solver.cpp:228] Iteration 31400, loss = 0.0130533
    I0515 01:05:35.836732 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:05:35.836761 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0130532 (* 1 = 0.0130532 loss)
    I0515 01:05:35.836783 23147 sgd_solver.cpp:106] Iteration 31400, lr = 6.89096e-05
    I0515 01:05:43.924579 23147 solver.cpp:228] Iteration 31500, loss = 0.0108255
    I0515 01:05:43.924638 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:05:43.924667 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0108255 (* 1 = 0.0108255 loss)
    I0515 01:05:43.924690 23147 sgd_solver.cpp:106] Iteration 31500, lr = 6.8785e-05
    I0515 01:05:52.010257 23147 solver.cpp:228] Iteration 31600, loss = 0.00887291
    I0515 01:05:52.010520 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:05:52.010567 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0088729 (* 1 = 0.0088729 loss)
    I0515 01:05:52.010592 23147 sgd_solver.cpp:106] Iteration 31600, lr = 6.8661e-05
    I0515 01:06:00.150343 23147 solver.cpp:228] Iteration 31700, loss = 0.0308613
    I0515 01:06:00.150391 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:06:00.150413 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0308613 (* 1 = 0.0308613 loss)
    I0515 01:06:00.150427 23147 sgd_solver.cpp:106] Iteration 31700, lr = 6.85374e-05
    I0515 01:06:08.292510 23147 solver.cpp:228] Iteration 31800, loss = 0.013584
    I0515 01:06:08.292556 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:06:08.292577 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0135839 (* 1 = 0.0135839 loss)
    I0515 01:06:08.292593 23147 sgd_solver.cpp:106] Iteration 31800, lr = 6.84144e-05
    I0515 01:06:16.435025 23147 solver.cpp:228] Iteration 31900, loss = 0.00802658
    I0515 01:06:16.435070 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:06:16.435092 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00802656 (* 1 = 0.00802656 loss)
    I0515 01:06:16.435107 23147 sgd_solver.cpp:106] Iteration 31900, lr = 6.82919e-05
    I0515 01:06:24.495265 23147 solver.cpp:337] Iteration 32000, Testing net (#0)
    I0515 01:06:28.838801 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.8165
    I0515 01:06:28.838845 23147 solver.cpp:404]     Test net output #1: loss_c = 0.574027 (* 1 = 0.574027 loss)
    I0515 01:06:28.890286 23147 solver.cpp:228] Iteration 32000, loss = 0.02055
    I0515 01:06:28.890338 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:06:28.890360 23147 solver.cpp:244]     Train net output #1: loss_c = 0.02055 (* 1 = 0.02055 loss)
    I0515 01:06:28.890380 23147 sgd_solver.cpp:106] Iteration 32000, lr = 6.817e-05
    I0515 01:06:37.012451 23147 solver.cpp:228] Iteration 32100, loss = 0.0359796
    I0515 01:06:37.012496 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:06:37.012516 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0359795 (* 1 = 0.0359795 loss)
    I0515 01:06:37.012531 23147 sgd_solver.cpp:106] Iteration 32100, lr = 6.80485e-05
    I0515 01:06:45.065397 23147 solver.cpp:228] Iteration 32200, loss = 0.0316459
    I0515 01:06:45.065443 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:06:45.065464 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0316459 (* 1 = 0.0316459 loss)
    I0515 01:06:45.065479 23147 sgd_solver.cpp:106] Iteration 32200, lr = 6.79275e-05
    I0515 01:06:53.119462 23147 solver.cpp:228] Iteration 32300, loss = 0.0148404
    I0515 01:06:53.119511 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:06:53.119539 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0148404 (* 1 = 0.0148404 loss)
    I0515 01:06:53.119554 23147 sgd_solver.cpp:106] Iteration 32300, lr = 6.7807e-05
    I0515 01:07:01.174252 23147 solver.cpp:228] Iteration 32400, loss = 0.0167722
    I0515 01:07:01.174474 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:07:01.174523 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0167721 (* 1 = 0.0167721 loss)
    I0515 01:07:01.174541 23147 sgd_solver.cpp:106] Iteration 32400, lr = 6.76871e-05
    I0515 01:07:09.304679 23147 solver.cpp:228] Iteration 32500, loss = 0.00598823
    I0515 01:07:09.304726 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:07:09.304747 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00598818 (* 1 = 0.00598818 loss)
    I0515 01:07:09.304762 23147 sgd_solver.cpp:106] Iteration 32500, lr = 6.75676e-05
    I0515 01:07:17.443763 23147 solver.cpp:228] Iteration 32600, loss = 0.0128617
    I0515 01:07:17.443814 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:07:17.443835 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0128616 (* 1 = 0.0128616 loss)
    I0515 01:07:17.443850 23147 sgd_solver.cpp:106] Iteration 32600, lr = 6.74486e-05
    I0515 01:07:25.583467 23147 solver.cpp:228] Iteration 32700, loss = 0.0173942
    I0515 01:07:25.583516 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:07:25.583537 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0173942 (* 1 = 0.0173942 loss)
    I0515 01:07:25.583552 23147 sgd_solver.cpp:106] Iteration 32700, lr = 6.73301e-05
    I0515 01:07:33.722947 23147 solver.cpp:228] Iteration 32800, loss = 0.00989038
    I0515 01:07:33.723078 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:07:33.723122 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00989035 (* 1 = 0.00989035 loss)
    I0515 01:07:33.723147 23147 sgd_solver.cpp:106] Iteration 32800, lr = 6.72121e-05
    I0515 01:07:41.863917 23147 solver.cpp:228] Iteration 32900, loss = 0.0181144
    I0515 01:07:41.863963 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:07:41.863982 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0181144 (* 1 = 0.0181144 loss)
    I0515 01:07:41.863998 23147 sgd_solver.cpp:106] Iteration 32900, lr = 6.70945e-05
    I0515 01:07:49.923739 23147 solver.cpp:337] Iteration 33000, Testing net (#0)
    I0515 01:07:54.346107 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.811083
    I0515 01:07:54.346164 23147 solver.cpp:404]     Test net output #1: loss_c = 0.594393 (* 1 = 0.594393 loss)
    I0515 01:07:54.402056 23147 solver.cpp:228] Iteration 33000, loss = 0.0219267
    I0515 01:07:54.402118 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:07:54.402148 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0219267 (* 1 = 0.0219267 loss)
    I0515 01:07:54.402174 23147 sgd_solver.cpp:106] Iteration 33000, lr = 6.69775e-05
    I0515 01:08:02.537148 23147 solver.cpp:228] Iteration 33100, loss = 0.0176201
    I0515 01:08:02.537210 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:08:02.537240 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0176201 (* 1 = 0.0176201 loss)
    I0515 01:08:02.537261 23147 sgd_solver.cpp:106] Iteration 33100, lr = 6.68609e-05
    I0515 01:08:10.622081 23147 solver.cpp:228] Iteration 33200, loss = 0.0136207
    I0515 01:08:10.622318 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:08:10.622364 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0136207 (* 1 = 0.0136207 loss)
    I0515 01:08:10.622390 23147 sgd_solver.cpp:106] Iteration 33200, lr = 6.67448e-05
    I0515 01:08:18.747198 23147 solver.cpp:228] Iteration 33300, loss = 0.0302698
    I0515 01:08:18.747242 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:08:18.747262 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0302698 (* 1 = 0.0302698 loss)
    I0515 01:08:18.747277 23147 sgd_solver.cpp:106] Iteration 33300, lr = 6.66291e-05
    I0515 01:08:26.887595 23147 solver.cpp:228] Iteration 33400, loss = 0.0191269
    I0515 01:08:26.887645 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:08:26.887666 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0191269 (* 1 = 0.0191269 loss)
    I0515 01:08:26.887681 23147 sgd_solver.cpp:106] Iteration 33400, lr = 6.65139e-05
    I0515 01:08:35.028409 23147 solver.cpp:228] Iteration 33500, loss = 0.016734
    I0515 01:08:35.028460 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:08:35.028481 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0167339 (* 1 = 0.0167339 loss)
    I0515 01:08:35.028496 23147 sgd_solver.cpp:106] Iteration 33500, lr = 6.63992e-05
    I0515 01:08:43.168656 23147 solver.cpp:228] Iteration 33600, loss = 0.0410677
    I0515 01:08:43.168768 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:08:43.168790 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0410677 (* 1 = 0.0410677 loss)
    I0515 01:08:43.168805 23147 sgd_solver.cpp:106] Iteration 33600, lr = 6.6285e-05
    I0515 01:08:51.308001 23147 solver.cpp:228] Iteration 33700, loss = 0.069521
    I0515 01:08:51.308050 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:08:51.308073 23147 solver.cpp:244]     Train net output #1: loss_c = 0.069521 (* 1 = 0.069521 loss)
    I0515 01:08:51.308086 23147 sgd_solver.cpp:106] Iteration 33700, lr = 6.61712e-05
    I0515 01:08:59.448132 23147 solver.cpp:228] Iteration 33800, loss = 0.00713377
    I0515 01:08:59.448182 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:08:59.448202 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00713376 (* 1 = 0.00713376 loss)
    I0515 01:08:59.448217 23147 sgd_solver.cpp:106] Iteration 33800, lr = 6.60578e-05
    I0515 01:09:07.590549 23147 solver.cpp:228] Iteration 33900, loss = 0.00589067
    I0515 01:09:07.590602 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:09:07.590623 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00589065 (* 1 = 0.00589065 loss)
    I0515 01:09:07.590639 23147 sgd_solver.cpp:106] Iteration 33900, lr = 6.5945e-05
    I0515 01:09:15.652588 23147 solver.cpp:337] Iteration 34000, Testing net (#0)
    I0515 01:09:20.068979 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.822833
    I0515 01:09:20.069037 23147 solver.cpp:404]     Test net output #1: loss_c = 0.567199 (* 1 = 0.567199 loss)
    I0515 01:09:20.124352 23147 solver.cpp:228] Iteration 34000, loss = 0.0382689
    I0515 01:09:20.124410 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:09:20.124441 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0382688 (* 1 = 0.0382688 loss)
    I0515 01:09:20.124467 23147 sgd_solver.cpp:106] Iteration 34000, lr = 6.58325e-05
    I0515 01:09:28.176669 23147 solver.cpp:228] Iteration 34100, loss = 0.0130042
    I0515 01:09:28.176713 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:09:28.176733 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0130042 (* 1 = 0.0130042 loss)
    I0515 01:09:28.176748 23147 sgd_solver.cpp:106] Iteration 34100, lr = 6.57205e-05
    I0515 01:09:36.229698 23147 solver.cpp:228] Iteration 34200, loss = 0.0141635
    I0515 01:09:36.229743 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:09:36.229768 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0141634 (* 1 = 0.0141634 loss)
    I0515 01:09:36.229786 23147 sgd_solver.cpp:106] Iteration 34200, lr = 6.5609e-05
    I0515 01:09:44.283769 23147 solver.cpp:228] Iteration 34300, loss = 0.0137532
    I0515 01:09:44.283813 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:09:44.283838 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0137532 (* 1 = 0.0137532 loss)
    I0515 01:09:44.283854 23147 sgd_solver.cpp:106] Iteration 34300, lr = 6.54979e-05
    I0515 01:09:52.424597 23147 solver.cpp:228] Iteration 34400, loss = 0.0460431
    I0515 01:09:52.424798 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:09:52.424818 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0460431 (* 1 = 0.0460431 loss)
    I0515 01:09:52.424834 23147 sgd_solver.cpp:106] Iteration 34400, lr = 6.53872e-05
    I0515 01:10:00.557572 23147 solver.cpp:228] Iteration 34500, loss = 0.0110463
    I0515 01:10:00.557618 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:10:00.557637 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0110463 (* 1 = 0.0110463 loss)
    I0515 01:10:00.557652 23147 sgd_solver.cpp:106] Iteration 34500, lr = 6.5277e-05
    I0515 01:10:08.699070 23147 solver.cpp:228] Iteration 34600, loss = 0.0184293
    I0515 01:10:08.699115 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:10:08.699136 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0184292 (* 1 = 0.0184292 loss)
    I0515 01:10:08.699149 23147 sgd_solver.cpp:106] Iteration 34600, lr = 6.51672e-05
    I0515 01:10:16.836565 23147 solver.cpp:228] Iteration 34700, loss = 0.0201745
    I0515 01:10:16.836611 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:10:16.836630 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0201745 (* 1 = 0.0201745 loss)
    I0515 01:10:16.836644 23147 sgd_solver.cpp:106] Iteration 34700, lr = 6.50578e-05
    I0515 01:10:24.990233 23147 solver.cpp:228] Iteration 34800, loss = 0.0384744
    I0515 01:10:24.990394 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:10:24.990440 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0384744 (* 1 = 0.0384744 loss)
    I0515 01:10:24.990465 23147 sgd_solver.cpp:106] Iteration 34800, lr = 6.49489e-05
    I0515 01:10:33.102991 23147 solver.cpp:228] Iteration 34900, loss = 0.0246729
    I0515 01:10:33.103036 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:10:33.103056 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0246728 (* 1 = 0.0246728 loss)
    I0515 01:10:33.103070 23147 sgd_solver.cpp:106] Iteration 34900, lr = 6.48403e-05
    I0515 01:10:41.164495 23147 solver.cpp:337] Iteration 35000, Testing net (#0)
    I0515 01:10:45.570686 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.8245
    I0515 01:10:45.570736 23147 solver.cpp:404]     Test net output #1: loss_c = 0.580255 (* 1 = 0.580255 loss)
    I0515 01:10:45.623255 23147 solver.cpp:228] Iteration 35000, loss = 0.0451987
    I0515 01:10:45.623319 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:10:45.623350 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0451986 (* 1 = 0.0451986 loss)
    I0515 01:10:45.623374 23147 sgd_solver.cpp:106] Iteration 35000, lr = 6.47322e-05
    I0515 01:10:53.729979 23147 solver.cpp:228] Iteration 35100, loss = 0.0138711
    I0515 01:10:53.730017 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:10:53.730037 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0138711 (* 1 = 0.0138711 loss)
    I0515 01:10:53.730052 23147 sgd_solver.cpp:106] Iteration 35100, lr = 6.46246e-05
    I0515 01:11:01.871240 23147 solver.cpp:228] Iteration 35200, loss = 0.0106921
    I0515 01:11:01.871312 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:11:01.871332 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0106921 (* 1 = 0.0106921 loss)
    I0515 01:11:01.871347 23147 sgd_solver.cpp:106] Iteration 35200, lr = 6.45173e-05
    I0515 01:11:10.010956 23147 solver.cpp:228] Iteration 35300, loss = 0.0362194
    I0515 01:11:10.010995 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:11:10.011015 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0362194 (* 1 = 0.0362194 loss)
    I0515 01:11:10.011030 23147 sgd_solver.cpp:106] Iteration 35300, lr = 6.44105e-05
    I0515 01:11:18.150630 23147 solver.cpp:228] Iteration 35400, loss = 0.00644249
    I0515 01:11:18.150681 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:11:18.150702 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00644244 (* 1 = 0.00644244 loss)
    I0515 01:11:18.150717 23147 sgd_solver.cpp:106] Iteration 35400, lr = 6.4304e-05
    I0515 01:11:26.291760 23147 solver.cpp:228] Iteration 35500, loss = 0.0243621
    I0515 01:11:26.291805 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:11:26.291826 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0243621 (* 1 = 0.0243621 loss)
    I0515 01:11:26.291841 23147 sgd_solver.cpp:106] Iteration 35500, lr = 6.4198e-05
    I0515 01:11:34.427788 23147 solver.cpp:228] Iteration 35600, loss = 0.0205978
    I0515 01:11:34.428055 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:11:34.428102 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0205978 (* 1 = 0.0205978 loss)
    I0515 01:11:34.428128 23147 sgd_solver.cpp:106] Iteration 35600, lr = 6.40924e-05
    I0515 01:11:42.532871 23147 solver.cpp:228] Iteration 35700, loss = 0.0213384
    I0515 01:11:42.532929 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:11:42.532958 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0213383 (* 1 = 0.0213383 loss)
    I0515 01:11:42.532980 23147 sgd_solver.cpp:106] Iteration 35700, lr = 6.39872e-05
    I0515 01:11:50.621541 23147 solver.cpp:228] Iteration 35800, loss = 0.0295671
    I0515 01:11:50.621592 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:11:50.621621 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0295671 (* 1 = 0.0295671 loss)
    I0515 01:11:50.621644 23147 sgd_solver.cpp:106] Iteration 35800, lr = 6.38823e-05
    I0515 01:11:58.710093 23147 solver.cpp:228] Iteration 35900, loss = 0.0113071
    I0515 01:11:58.710146 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:11:58.710175 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0113071 (* 1 = 0.0113071 loss)
    I0515 01:11:58.710196 23147 sgd_solver.cpp:106] Iteration 35900, lr = 6.37779e-05
    I0515 01:12:06.720365 23147 solver.cpp:337] Iteration 36000, Testing net (#0)
    I0515 01:12:11.127306 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.832833
    I0515 01:12:11.127369 23147 solver.cpp:404]     Test net output #1: loss_c = 0.528346 (* 1 = 0.528346 loss)
    I0515 01:12:11.178793 23147 solver.cpp:228] Iteration 36000, loss = 0.0221674
    I0515 01:12:11.178843 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:12:11.178864 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0221673 (* 1 = 0.0221673 loss)
    I0515 01:12:11.178886 23147 sgd_solver.cpp:106] Iteration 36000, lr = 6.36739e-05
    I0515 01:12:19.326874 23147 solver.cpp:228] Iteration 36100, loss = 0.00795659
    I0515 01:12:19.326925 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:12:19.326946 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00795654 (* 1 = 0.00795654 loss)
    I0515 01:12:19.326961 23147 sgd_solver.cpp:106] Iteration 36100, lr = 6.35703e-05
    I0515 01:12:27.467397 23147 solver.cpp:228] Iteration 36200, loss = 0.0152067
    I0515 01:12:27.467447 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:12:27.467466 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0152067 (* 1 = 0.0152067 loss)
    I0515 01:12:27.467481 23147 sgd_solver.cpp:106] Iteration 36200, lr = 6.34671e-05
    I0515 01:12:35.606755 23147 solver.cpp:228] Iteration 36300, loss = 0.0311386
    I0515 01:12:35.606803 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:12:35.606823 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0311385 (* 1 = 0.0311385 loss)
    I0515 01:12:35.606838 23147 sgd_solver.cpp:106] Iteration 36300, lr = 6.33642e-05
    I0515 01:12:43.746112 23147 solver.cpp:228] Iteration 36400, loss = 0.0280337
    I0515 01:12:43.746368 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:12:43.746414 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0280336 (* 1 = 0.0280336 loss)
    I0515 01:12:43.746438 23147 sgd_solver.cpp:106] Iteration 36400, lr = 6.32618e-05
    I0515 01:12:51.883211 23147 solver.cpp:228] Iteration 36500, loss = 0.0108437
    I0515 01:12:51.883257 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:12:51.883276 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0108437 (* 1 = 0.0108437 loss)
    I0515 01:12:51.883291 23147 sgd_solver.cpp:106] Iteration 36500, lr = 6.31597e-05
    I0515 01:13:00.020928 23147 solver.cpp:228] Iteration 36600, loss = 0.015234
    I0515 01:13:00.020970 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:13:00.020989 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0152339 (* 1 = 0.0152339 loss)
    I0515 01:13:00.021005 23147 sgd_solver.cpp:106] Iteration 36600, lr = 6.30581e-05
    I0515 01:13:08.160662 23147 solver.cpp:228] Iteration 36700, loss = 0.0312128
    I0515 01:13:08.160704 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:13:08.160727 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0312128 (* 1 = 0.0312128 loss)
    I0515 01:13:08.160742 23147 sgd_solver.cpp:106] Iteration 36700, lr = 6.29568e-05
    I0515 01:13:16.302474 23147 solver.cpp:228] Iteration 36800, loss = 0.0634332
    I0515 01:13:16.302615 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 01:13:16.302660 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0634331 (* 1 = 0.0634331 loss)
    I0515 01:13:16.302685 23147 sgd_solver.cpp:106] Iteration 36800, lr = 6.28558e-05
    I0515 01:13:24.440922 23147 solver.cpp:228] Iteration 36900, loss = 0.0218632
    I0515 01:13:24.440968 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:13:24.440986 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0218631 (* 1 = 0.0218631 loss)
    I0515 01:13:24.441001 23147 sgd_solver.cpp:106] Iteration 36900, lr = 6.27553e-05
    I0515 01:13:32.497020 23147 solver.cpp:337] Iteration 37000, Testing net (#0)
    I0515 01:13:36.900205 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.836666
    I0515 01:13:36.900249 23147 solver.cpp:404]     Test net output #1: loss_c = 0.546316 (* 1 = 0.546316 loss)
    I0515 01:13:36.956398 23147 solver.cpp:228] Iteration 37000, loss = 0.0251654
    I0515 01:13:36.956476 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:13:36.956506 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0251653 (* 1 = 0.0251653 loss)
    I0515 01:13:36.956531 23147 sgd_solver.cpp:106] Iteration 37000, lr = 6.26551e-05
    I0515 01:13:45.096216 23147 solver.cpp:228] Iteration 37100, loss = 0.00871162
    I0515 01:13:45.096266 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:13:45.096289 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00871156 (* 1 = 0.00871156 loss)
    I0515 01:13:45.096303 23147 sgd_solver.cpp:106] Iteration 37100, lr = 6.25553e-05
    I0515 01:13:53.236534 23147 solver.cpp:228] Iteration 37200, loss = 0.0122121
    I0515 01:13:53.236657 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:13:53.236677 23147 solver.cpp:244]     Train net output #1: loss_c = 0.012212 (* 1 = 0.012212 loss)
    I0515 01:13:53.236692 23147 sgd_solver.cpp:106] Iteration 37200, lr = 6.24559e-05
    I0515 01:14:01.378592 23147 solver.cpp:228] Iteration 37300, loss = 0.0219507
    I0515 01:14:01.378645 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:14:01.378665 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0219506 (* 1 = 0.0219506 loss)
    I0515 01:14:01.378680 23147 sgd_solver.cpp:106] Iteration 37300, lr = 6.23568e-05
    I0515 01:14:09.520041 23147 solver.cpp:228] Iteration 37400, loss = 0.0277835
    I0515 01:14:09.520094 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:14:09.520114 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0277834 (* 1 = 0.0277834 loss)
    I0515 01:14:09.520129 23147 sgd_solver.cpp:106] Iteration 37400, lr = 6.22582e-05
    I0515 01:14:17.661254 23147 solver.cpp:228] Iteration 37500, loss = 0.0126698
    I0515 01:14:17.661309 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:14:17.661330 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0126698 (* 1 = 0.0126698 loss)
    I0515 01:14:17.661345 23147 sgd_solver.cpp:106] Iteration 37500, lr = 6.21598e-05
    I0515 01:14:25.803282 23147 solver.cpp:228] Iteration 37600, loss = 0.0183385
    I0515 01:14:25.803485 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:14:25.803510 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0183384 (* 1 = 0.0183384 loss)
    I0515 01:14:25.803526 23147 sgd_solver.cpp:106] Iteration 37600, lr = 6.20619e-05
    I0515 01:14:33.942118 23147 solver.cpp:228] Iteration 37700, loss = 0.0242225
    I0515 01:14:33.942167 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:14:33.942185 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0242224 (* 1 = 0.0242224 loss)
    I0515 01:14:33.942201 23147 sgd_solver.cpp:106] Iteration 37700, lr = 6.19643e-05
    I0515 01:14:42.079212 23147 solver.cpp:228] Iteration 37800, loss = 0.0142473
    I0515 01:14:42.079259 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:14:42.079283 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0142472 (* 1 = 0.0142472 loss)
    I0515 01:14:42.079298 23147 sgd_solver.cpp:106] Iteration 37800, lr = 6.1867e-05
    I0515 01:14:50.217716 23147 solver.cpp:228] Iteration 37900, loss = 0.0501021
    I0515 01:14:50.217767 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:14:50.217787 23147 solver.cpp:244]     Train net output #1: loss_c = 0.050102 (* 1 = 0.050102 loss)
    I0515 01:14:50.217802 23147 sgd_solver.cpp:106] Iteration 37900, lr = 6.17701e-05
    I0515 01:14:58.279844 23147 solver.cpp:337] Iteration 38000, Testing net (#0)
    I0515 01:15:02.703800 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.842417
    I0515 01:15:02.703853 23147 solver.cpp:404]     Test net output #1: loss_c = 0.529293 (* 1 = 0.529293 loss)
    I0515 01:15:02.756497 23147 solver.cpp:228] Iteration 38000, loss = 0.0248975
    I0515 01:15:02.756531 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:15:02.756551 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0248974 (* 1 = 0.0248974 loss)
    I0515 01:15:02.756569 23147 sgd_solver.cpp:106] Iteration 38000, lr = 6.16736e-05
    I0515 01:15:10.886440 23147 solver.cpp:228] Iteration 38100, loss = 0.027636
    I0515 01:15:10.886487 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:15:10.886507 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0276359 (* 1 = 0.0276359 loss)
    I0515 01:15:10.886521 23147 sgd_solver.cpp:106] Iteration 38100, lr = 6.15774e-05
    I0515 01:15:19.022907 23147 solver.cpp:228] Iteration 38200, loss = 0.0328616
    I0515 01:15:19.022958 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:15:19.022979 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0328616 (* 1 = 0.0328616 loss)
    I0515 01:15:19.022992 23147 sgd_solver.cpp:106] Iteration 38200, lr = 6.14815e-05
    I0515 01:15:27.162092 23147 solver.cpp:228] Iteration 38300, loss = 0.0162534
    I0515 01:15:27.162134 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:15:27.162158 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0162533 (* 1 = 0.0162533 loss)
    I0515 01:15:27.162174 23147 sgd_solver.cpp:106] Iteration 38300, lr = 6.1386e-05
    I0515 01:15:35.303169 23147 solver.cpp:228] Iteration 38400, loss = 0.0267065
    I0515 01:15:35.303328 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:15:35.303372 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0267064 (* 1 = 0.0267064 loss)
    I0515 01:15:35.303397 23147 sgd_solver.cpp:106] Iteration 38400, lr = 6.12909e-05
    I0515 01:15:43.443081 23147 solver.cpp:228] Iteration 38500, loss = 0.0249189
    I0515 01:15:43.443128 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:15:43.443147 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0249189 (* 1 = 0.0249189 loss)
    I0515 01:15:43.443162 23147 sgd_solver.cpp:106] Iteration 38500, lr = 6.11961e-05
    I0515 01:15:51.573590 23147 solver.cpp:228] Iteration 38600, loss = 0.0312554
    I0515 01:15:51.573635 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:15:51.573654 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0312554 (* 1 = 0.0312554 loss)
    I0515 01:15:51.573669 23147 sgd_solver.cpp:106] Iteration 38600, lr = 6.11016e-05
    I0515 01:15:59.704900 23147 solver.cpp:228] Iteration 38700, loss = 0.0231129
    I0515 01:15:59.704946 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:15:59.704964 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0231128 (* 1 = 0.0231128 loss)
    I0515 01:15:59.704978 23147 sgd_solver.cpp:106] Iteration 38700, lr = 6.10075e-05
    I0515 01:16:07.844820 23147 solver.cpp:228] Iteration 38800, loss = 0.0105228
    I0515 01:16:07.844954 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:16:07.844974 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0105228 (* 1 = 0.0105228 loss)
    I0515 01:16:07.845001 23147 sgd_solver.cpp:106] Iteration 38800, lr = 6.09137e-05
    I0515 01:16:15.970160 23147 solver.cpp:228] Iteration 38900, loss = 0.0279784
    I0515 01:16:15.970203 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:16:15.970223 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0279783 (* 1 = 0.0279783 loss)
    I0515 01:16:15.970238 23147 sgd_solver.cpp:106] Iteration 38900, lr = 6.08203e-05
    I0515 01:16:24.028851 23147 solver.cpp:337] Iteration 39000, Testing net (#0)
    I0515 01:16:28.412364 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.841667
    I0515 01:16:28.412426 23147 solver.cpp:404]     Test net output #1: loss_c = 0.556747 (* 1 = 0.556747 loss)
    I0515 01:16:28.463860 23147 solver.cpp:228] Iteration 39000, loss = 0.0103507
    I0515 01:16:28.463909 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:16:28.463930 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0103506 (* 1 = 0.0103506 loss)
    I0515 01:16:28.463949 23147 sgd_solver.cpp:106] Iteration 39000, lr = 6.07272e-05
    I0515 01:16:36.607378 23147 solver.cpp:228] Iteration 39100, loss = 0.0180897
    I0515 01:16:36.607429 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:16:36.607450 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0180896 (* 1 = 0.0180896 loss)
    I0515 01:16:36.607465 23147 sgd_solver.cpp:106] Iteration 39100, lr = 6.06344e-05
    I0515 01:16:44.747591 23147 solver.cpp:228] Iteration 39200, loss = 0.0337429
    I0515 01:16:44.747704 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:16:44.747725 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0337429 (* 1 = 0.0337429 loss)
    I0515 01:16:44.747740 23147 sgd_solver.cpp:106] Iteration 39200, lr = 6.05419e-05
    I0515 01:16:52.887812 23147 solver.cpp:228] Iteration 39300, loss = 0.0467151
    I0515 01:16:52.887861 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:16:52.887881 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0467151 (* 1 = 0.0467151 loss)
    I0515 01:16:52.887895 23147 sgd_solver.cpp:106] Iteration 39300, lr = 6.04498e-05
    I0515 01:17:01.026716 23147 solver.cpp:228] Iteration 39400, loss = 0.0287876
    I0515 01:17:01.026767 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:17:01.026788 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0287875 (* 1 = 0.0287875 loss)
    I0515 01:17:01.026803 23147 sgd_solver.cpp:106] Iteration 39400, lr = 6.0358e-05
    I0515 01:17:09.166236 23147 solver.cpp:228] Iteration 39500, loss = 0.0203775
    I0515 01:17:09.166287 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:17:09.166309 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0203774 (* 1 = 0.0203774 loss)
    I0515 01:17:09.166323 23147 sgd_solver.cpp:106] Iteration 39500, lr = 6.02665e-05
    I0515 01:17:17.304010 23147 solver.cpp:228] Iteration 39600, loss = 0.0126547
    I0515 01:17:17.304262 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:17:17.304308 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0126546 (* 1 = 0.0126546 loss)
    I0515 01:17:17.304333 23147 sgd_solver.cpp:106] Iteration 39600, lr = 6.01754e-05
    I0515 01:17:25.446432 23147 solver.cpp:228] Iteration 39700, loss = 0.0151156
    I0515 01:17:25.446483 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:17:25.446504 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0151155 (* 1 = 0.0151155 loss)
    I0515 01:17:25.446519 23147 sgd_solver.cpp:106] Iteration 39700, lr = 6.00845e-05
    I0515 01:17:33.578569 23147 solver.cpp:228] Iteration 39800, loss = 0.0213871
    I0515 01:17:33.578615 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:17:33.578639 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0213871 (* 1 = 0.0213871 loss)
    I0515 01:17:33.578654 23147 sgd_solver.cpp:106] Iteration 39800, lr = 5.9994e-05
    I0515 01:17:41.717468 23147 solver.cpp:228] Iteration 39900, loss = 0.00812651
    I0515 01:17:41.717519 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:17:41.717538 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00812644 (* 1 = 0.00812644 loss)
    I0515 01:17:41.717553 23147 sgd_solver.cpp:106] Iteration 39900, lr = 5.99038e-05
    I0515 01:17:49.777889 23147 solver.cpp:337] Iteration 40000, Testing net (#0)
    I0515 01:17:54.174664 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.830167
    I0515 01:17:54.174732 23147 solver.cpp:404]     Test net output #1: loss_c = 0.541321 (* 1 = 0.541321 loss)
    I0515 01:17:54.229662 23147 solver.cpp:228] Iteration 40000, loss = 0.0245645
    I0515 01:17:54.229704 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:17:54.229732 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0245644 (* 1 = 0.0245644 loss)
    I0515 01:17:54.229756 23147 sgd_solver.cpp:106] Iteration 40000, lr = 5.98139e-05
    I0515 01:18:02.316545 23147 solver.cpp:228] Iteration 40100, loss = 0.0269361
    I0515 01:18:02.316601 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:18:02.316629 23147 solver.cpp:244]     Train net output #1: loss_c = 0.026936 (* 1 = 0.026936 loss)
    I0515 01:18:02.316650 23147 sgd_solver.cpp:106] Iteration 40100, lr = 5.97244e-05
    I0515 01:18:10.411134 23147 solver.cpp:228] Iteration 40200, loss = 0.00827368
    I0515 01:18:10.411192 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:18:10.411221 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00827361 (* 1 = 0.00827361 loss)
    I0515 01:18:10.411242 23147 sgd_solver.cpp:106] Iteration 40200, lr = 5.96351e-05
    I0515 01:18:18.503252 23147 solver.cpp:228] Iteration 40300, loss = 0.0257086
    I0515 01:18:18.503309 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:18:18.503337 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0257085 (* 1 = 0.0257085 loss)
    I0515 01:18:18.503360 23147 sgd_solver.cpp:106] Iteration 40300, lr = 5.95462e-05
    I0515 01:18:26.585522 23147 solver.cpp:228] Iteration 40400, loss = 0.0458735
    I0515 01:18:26.585677 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:18:26.585722 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0458734 (* 1 = 0.0458734 loss)
    I0515 01:18:26.585747 23147 sgd_solver.cpp:106] Iteration 40400, lr = 5.94576e-05
    I0515 01:18:34.721472 23147 solver.cpp:228] Iteration 40500, loss = 0.0243841
    I0515 01:18:34.721521 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:18:34.721544 23147 solver.cpp:244]     Train net output #1: loss_c = 0.024384 (* 1 = 0.024384 loss)
    I0515 01:18:34.721560 23147 sgd_solver.cpp:106] Iteration 40500, lr = 5.93692e-05
    I0515 01:18:42.858312 23147 solver.cpp:228] Iteration 40600, loss = 0.0141991
    I0515 01:18:42.858358 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:18:42.858378 23147 solver.cpp:244]     Train net output #1: loss_c = 0.014199 (* 1 = 0.014199 loss)
    I0515 01:18:42.858393 23147 sgd_solver.cpp:106] Iteration 40600, lr = 5.92812e-05
    I0515 01:18:50.950218 23147 solver.cpp:228] Iteration 40700, loss = 0.0513812
    I0515 01:18:50.950268 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 01:18:50.950289 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0513811 (* 1 = 0.0513811 loss)
    I0515 01:18:50.950304 23147 sgd_solver.cpp:106] Iteration 40700, lr = 5.91935e-05
    I0515 01:18:59.090999 23147 solver.cpp:228] Iteration 40800, loss = 0.0202733
    I0515 01:18:59.091238 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:18:59.091282 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0202733 (* 1 = 0.0202733 loss)
    I0515 01:18:59.091306 23147 sgd_solver.cpp:106] Iteration 40800, lr = 5.91061e-05
    I0515 01:19:07.221973 23147 solver.cpp:228] Iteration 40900, loss = 0.0439924
    I0515 01:19:07.222017 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:19:07.222036 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0439924 (* 1 = 0.0439924 loss)
    I0515 01:19:07.222051 23147 sgd_solver.cpp:106] Iteration 40900, lr = 5.9019e-05
    I0515 01:19:15.264945 23147 solver.cpp:337] Iteration 41000, Testing net (#0)
    I0515 01:19:19.687737 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.829167
    I0515 01:19:19.687791 23147 solver.cpp:404]     Test net output #1: loss_c = 0.509881 (* 1 = 0.509881 loss)
    I0515 01:19:19.743058 23147 solver.cpp:228] Iteration 41000, loss = 0.0113523
    I0515 01:19:19.743115 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:19:19.743145 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0113523 (* 1 = 0.0113523 loss)
    I0515 01:19:19.743170 23147 sgd_solver.cpp:106] Iteration 41000, lr = 5.89322e-05
    I0515 01:19:27.873152 23147 solver.cpp:228] Iteration 41100, loss = 0.0245931
    I0515 01:19:27.873194 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:19:27.873214 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0245931 (* 1 = 0.0245931 loss)
    I0515 01:19:27.873230 23147 sgd_solver.cpp:106] Iteration 41100, lr = 5.88456e-05
    I0515 01:19:36.006302 23147 solver.cpp:228] Iteration 41200, loss = 0.0257589
    I0515 01:19:36.006564 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:19:36.006609 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0257589 (* 1 = 0.0257589 loss)
    I0515 01:19:36.006634 23147 sgd_solver.cpp:106] Iteration 41200, lr = 5.87594e-05
    I0515 01:19:44.106209 23147 solver.cpp:228] Iteration 41300, loss = 0.0154719
    I0515 01:19:44.106262 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:19:44.106292 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0154718 (* 1 = 0.0154718 loss)
    I0515 01:19:44.106313 23147 sgd_solver.cpp:106] Iteration 41300, lr = 5.86735e-05
    I0515 01:19:52.193267 23147 solver.cpp:228] Iteration 41400, loss = 0.0186231
    I0515 01:19:52.193320 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:19:52.193347 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0186231 (* 1 = 0.0186231 loss)
    I0515 01:19:52.193368 23147 sgd_solver.cpp:106] Iteration 41400, lr = 5.85879e-05
    I0515 01:20:00.278745 23147 solver.cpp:228] Iteration 41500, loss = 0.0230841
    I0515 01:20:00.278802 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:20:00.278831 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0230841 (* 1 = 0.0230841 loss)
    I0515 01:20:00.278853 23147 sgd_solver.cpp:106] Iteration 41500, lr = 5.85025e-05
    I0515 01:20:08.363380 23147 solver.cpp:228] Iteration 41600, loss = 0.0113279
    I0515 01:20:08.363543 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:20:08.363589 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0113278 (* 1 = 0.0113278 loss)
    I0515 01:20:08.363612 23147 sgd_solver.cpp:106] Iteration 41600, lr = 5.84175e-05
    I0515 01:20:16.487644 23147 solver.cpp:228] Iteration 41700, loss = 0.0212443
    I0515 01:20:16.487689 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:20:16.487714 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0212443 (* 1 = 0.0212443 loss)
    I0515 01:20:16.487728 23147 sgd_solver.cpp:106] Iteration 41700, lr = 5.83327e-05
    I0515 01:20:24.598326 23147 solver.cpp:228] Iteration 41800, loss = 0.011474
    I0515 01:20:24.598376 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:20:24.598400 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0114739 (* 1 = 0.0114739 loss)
    I0515 01:20:24.598417 23147 sgd_solver.cpp:106] Iteration 41800, lr = 5.82482e-05
    I0515 01:20:32.712491 23147 solver.cpp:228] Iteration 41900, loss = 0.012943
    I0515 01:20:32.712537 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:20:32.712556 23147 solver.cpp:244]     Train net output #1: loss_c = 0.012943 (* 1 = 0.012943 loss)
    I0515 01:20:32.712570 23147 sgd_solver.cpp:106] Iteration 41900, lr = 5.8164e-05
    I0515 01:20:40.766135 23147 solver.cpp:337] Iteration 42000, Testing net (#0)
    I0515 01:20:45.148041 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.809
    I0515 01:20:45.148092 23147 solver.cpp:404]     Test net output #1: loss_c = 0.543062 (* 1 = 0.543062 loss)
    I0515 01:20:45.199414 23147 solver.cpp:228] Iteration 42000, loss = 0.0737943
    I0515 01:20:45.199455 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 01:20:45.199473 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0737943 (* 1 = 0.0737943 loss)
    I0515 01:20:45.199493 23147 sgd_solver.cpp:106] Iteration 42000, lr = 5.80801e-05
    I0515 01:20:53.328305 23147 solver.cpp:228] Iteration 42100, loss = 0.0185475
    I0515 01:20:53.328354 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:20:53.328377 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0185475 (* 1 = 0.0185475 loss)
    I0515 01:20:53.328392 23147 sgd_solver.cpp:106] Iteration 42100, lr = 5.79965e-05
    I0515 01:21:01.460400 23147 solver.cpp:228] Iteration 42200, loss = 0.0260715
    I0515 01:21:01.460438 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:21:01.460458 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0260715 (* 1 = 0.0260715 loss)
    I0515 01:21:01.460474 23147 sgd_solver.cpp:106] Iteration 42200, lr = 5.79131e-05
    I0515 01:21:09.599416 23147 solver.cpp:228] Iteration 42300, loss = 0.0124498
    I0515 01:21:09.599463 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:21:09.599483 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0124497 (* 1 = 0.0124497 loss)
    I0515 01:21:09.599496 23147 sgd_solver.cpp:106] Iteration 42300, lr = 5.78301e-05
    I0515 01:21:17.736691 23147 solver.cpp:228] Iteration 42400, loss = 0.0456511
    I0515 01:21:17.736775 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:21:17.736795 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0456511 (* 1 = 0.0456511 loss)
    I0515 01:21:17.736811 23147 sgd_solver.cpp:106] Iteration 42400, lr = 5.77473e-05
    I0515 01:21:25.831166 23147 solver.cpp:228] Iteration 42500, loss = 0.020918
    I0515 01:21:25.831214 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:21:25.831234 23147 solver.cpp:244]     Train net output #1: loss_c = 0.020918 (* 1 = 0.020918 loss)
    I0515 01:21:25.831249 23147 sgd_solver.cpp:106] Iteration 42500, lr = 5.76648e-05
    I0515 01:21:33.968731 23147 solver.cpp:228] Iteration 42600, loss = 0.0146328
    I0515 01:21:33.968782 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:21:33.968802 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0146328 (* 1 = 0.0146328 loss)
    I0515 01:21:33.968817 23147 sgd_solver.cpp:106] Iteration 42600, lr = 5.75825e-05
    I0515 01:21:42.092022 23147 solver.cpp:228] Iteration 42700, loss = 0.0183652
    I0515 01:21:42.092074 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:21:42.092095 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0183652 (* 1 = 0.0183652 loss)
    I0515 01:21:42.092110 23147 sgd_solver.cpp:106] Iteration 42700, lr = 5.75006e-05
    I0515 01:21:50.232467 23147 solver.cpp:228] Iteration 42800, loss = 0.0136332
    I0515 01:21:50.232635 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:21:50.232656 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0136332 (* 1 = 0.0136332 loss)
    I0515 01:21:50.232673 23147 sgd_solver.cpp:106] Iteration 42800, lr = 5.74189e-05
    I0515 01:21:58.373920 23147 solver.cpp:228] Iteration 42900, loss = 0.0200218
    I0515 01:21:58.373971 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:21:58.373994 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0200217 (* 1 = 0.0200217 loss)
    I0515 01:21:58.374008 23147 sgd_solver.cpp:106] Iteration 42900, lr = 5.73374e-05
    I0515 01:22:06.433794 23147 solver.cpp:337] Iteration 43000, Testing net (#0)
    I0515 01:22:10.851935 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.83675
    I0515 01:22:10.851994 23147 solver.cpp:404]     Test net output #1: loss_c = 0.533005 (* 1 = 0.533005 loss)
    I0515 01:22:10.904278 23147 solver.cpp:228] Iteration 43000, loss = 0.0162668
    I0515 01:22:10.904333 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:22:10.904362 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0162667 (* 1 = 0.0162667 loss)
    I0515 01:22:10.904388 23147 sgd_solver.cpp:106] Iteration 43000, lr = 5.72563e-05
    I0515 01:22:19.038727 23147 solver.cpp:228] Iteration 43100, loss = 0.0129114
    I0515 01:22:19.038771 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:22:19.038791 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0129113 (* 1 = 0.0129113 loss)
    I0515 01:22:19.038806 23147 sgd_solver.cpp:106] Iteration 43100, lr = 5.71754e-05
    I0515 01:22:27.096930 23147 solver.cpp:228] Iteration 43200, loss = 0.0375897
    I0515 01:22:27.097092 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:22:27.097131 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0375897 (* 1 = 0.0375897 loss)
    I0515 01:22:27.097148 23147 sgd_solver.cpp:106] Iteration 43200, lr = 5.70948e-05
    I0515 01:22:35.224705 23147 solver.cpp:228] Iteration 43300, loss = 0.0303729
    I0515 01:22:35.224756 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:22:35.224777 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0303728 (* 1 = 0.0303728 loss)
    I0515 01:22:35.224792 23147 sgd_solver.cpp:106] Iteration 43300, lr = 5.70144e-05
    I0515 01:22:43.365475 23147 solver.cpp:228] Iteration 43400, loss = 0.00960835
    I0515 01:22:43.365525 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:22:43.365546 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0096083 (* 1 = 0.0096083 loss)
    I0515 01:22:43.365561 23147 sgd_solver.cpp:106] Iteration 43400, lr = 5.69343e-05
    I0515 01:22:51.504883 23147 solver.cpp:228] Iteration 43500, loss = 0.00991799
    I0515 01:22:51.504932 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:22:51.504956 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00991795 (* 1 = 0.00991795 loss)
    I0515 01:22:51.504971 23147 sgd_solver.cpp:106] Iteration 43500, lr = 5.68545e-05
    I0515 01:22:59.642810 23147 solver.cpp:228] Iteration 43600, loss = 0.0151432
    I0515 01:22:59.642897 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:22:59.642917 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0151432 (* 1 = 0.0151432 loss)
    I0515 01:22:59.642931 23147 sgd_solver.cpp:106] Iteration 43600, lr = 5.67749e-05
    I0515 01:23:07.784118 23147 solver.cpp:228] Iteration 43700, loss = 0.0136715
    I0515 01:23:07.784162 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:23:07.784180 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0136714 (* 1 = 0.0136714 loss)
    I0515 01:23:07.784194 23147 sgd_solver.cpp:106] Iteration 43700, lr = 5.66956e-05
    I0515 01:23:15.921653 23147 solver.cpp:228] Iteration 43800, loss = 0.0282132
    I0515 01:23:15.921699 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:23:15.921717 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0282131 (* 1 = 0.0282131 loss)
    I0515 01:23:15.921732 23147 sgd_solver.cpp:106] Iteration 43800, lr = 5.66165e-05
    I0515 01:23:24.063071 23147 solver.cpp:228] Iteration 43900, loss = 0.0294359
    I0515 01:23:24.063115 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:23:24.063135 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0294358 (* 1 = 0.0294358 loss)
    I0515 01:23:24.063150 23147 sgd_solver.cpp:106] Iteration 43900, lr = 5.65377e-05
    I0515 01:23:32.123364 23147 solver.cpp:337] Iteration 44000, Testing net (#0)
    I0515 01:23:36.536056 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.827166
    I0515 01:23:36.536106 23147 solver.cpp:404]     Test net output #1: loss_c = 0.541891 (* 1 = 0.541891 loss)
    I0515 01:23:36.591104 23147 solver.cpp:228] Iteration 44000, loss = 0.0185453
    I0515 01:23:36.591164 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:23:36.591194 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0185452 (* 1 = 0.0185452 loss)
    I0515 01:23:36.591219 23147 sgd_solver.cpp:106] Iteration 44000, lr = 5.64592e-05
    I0515 01:23:44.668660 23147 solver.cpp:228] Iteration 44100, loss = 0.0182301
    I0515 01:23:44.668706 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:23:44.668727 23147 solver.cpp:244]     Train net output #1: loss_c = 0.01823 (* 1 = 0.01823 loss)
    I0515 01:23:44.668742 23147 sgd_solver.cpp:106] Iteration 44100, lr = 5.63809e-05
    I0515 01:23:52.810469 23147 solver.cpp:228] Iteration 44200, loss = 0.015325
    I0515 01:23:52.810509 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:23:52.810530 23147 solver.cpp:244]     Train net output #1: loss_c = 0.015325 (* 1 = 0.015325 loss)
    I0515 01:23:52.810544 23147 sgd_solver.cpp:106] Iteration 44200, lr = 5.63029e-05
    I0515 01:24:00.919153 23147 solver.cpp:228] Iteration 44300, loss = 0.0314615
    I0515 01:24:00.919209 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:24:00.919239 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0314614 (* 1 = 0.0314614 loss)
    I0515 01:24:00.919260 23147 sgd_solver.cpp:106] Iteration 44300, lr = 5.62251e-05
    I0515 01:24:09.007060 23147 solver.cpp:228] Iteration 44400, loss = 0.0206795
    I0515 01:24:09.007267 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:24:09.007299 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0206794 (* 1 = 0.0206794 loss)
    I0515 01:24:09.007320 23147 sgd_solver.cpp:106] Iteration 44400, lr = 5.61475e-05
    I0515 01:24:17.095880 23147 solver.cpp:228] Iteration 44500, loss = 0.0173648
    I0515 01:24:17.095930 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:24:17.095959 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0173647 (* 1 = 0.0173647 loss)
    I0515 01:24:17.095980 23147 sgd_solver.cpp:106] Iteration 44500, lr = 5.60703e-05
    I0515 01:24:25.184124 23147 solver.cpp:228] Iteration 44600, loss = 0.0130978
    I0515 01:24:25.184185 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:24:25.184214 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0130977 (* 1 = 0.0130977 loss)
    I0515 01:24:25.184236 23147 sgd_solver.cpp:106] Iteration 44600, lr = 5.59932e-05
    I0515 01:24:33.275095 23147 solver.cpp:228] Iteration 44700, loss = 0.023351
    I0515 01:24:33.275156 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:24:33.275185 23147 solver.cpp:244]     Train net output #1: loss_c = 0.023351 (* 1 = 0.023351 loss)
    I0515 01:24:33.275207 23147 sgd_solver.cpp:106] Iteration 44700, lr = 5.59164e-05
    I0515 01:24:41.381281 23147 solver.cpp:228] Iteration 44800, loss = 0.0411187
    I0515 01:24:41.381395 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:24:41.381419 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0411186 (* 1 = 0.0411186 loss)
    I0515 01:24:41.381438 23147 sgd_solver.cpp:106] Iteration 44800, lr = 5.58399e-05
    I0515 01:24:49.523353 23147 solver.cpp:228] Iteration 44900, loss = 0.0281795
    I0515 01:24:49.523402 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:24:49.523422 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0281795 (* 1 = 0.0281795 loss)
    I0515 01:24:49.523437 23147 sgd_solver.cpp:106] Iteration 44900, lr = 5.57636e-05
    I0515 01:24:57.575512 23147 solver.cpp:337] Iteration 45000, Testing net (#0)
    I0515 01:25:01.950747 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.827166
    I0515 01:25:01.950801 23147 solver.cpp:404]     Test net output #1: loss_c = 0.525594 (* 1 = 0.525594 loss)
    I0515 01:25:02.005877 23147 solver.cpp:228] Iteration 45000, loss = 0.0237206
    I0515 01:25:02.005936 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:25:02.005967 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0237206 (* 1 = 0.0237206 loss)
    I0515 01:25:02.005995 23147 sgd_solver.cpp:106] Iteration 45000, lr = 5.56875e-05
    I0515 01:25:10.109318 23147 solver.cpp:228] Iteration 45100, loss = 0.071701
    I0515 01:25:10.109365 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:25:10.109387 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0717009 (* 1 = 0.0717009 loss)
    I0515 01:25:10.109403 23147 sgd_solver.cpp:106] Iteration 45100, lr = 5.56117e-05
    I0515 01:25:18.248179 23147 solver.cpp:228] Iteration 45200, loss = 0.0162412
    I0515 01:25:18.248381 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:25:18.248433 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0162411 (* 1 = 0.0162411 loss)
    I0515 01:25:18.248448 23147 sgd_solver.cpp:106] Iteration 45200, lr = 5.55361e-05
    I0515 01:25:26.387346 23147 solver.cpp:228] Iteration 45300, loss = 0.0208434
    I0515 01:25:26.387395 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:25:26.387415 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0208433 (* 1 = 0.0208433 loss)
    I0515 01:25:26.387430 23147 sgd_solver.cpp:106] Iteration 45300, lr = 5.54608e-05
    I0515 01:25:34.526376 23147 solver.cpp:228] Iteration 45400, loss = 0.0159084
    I0515 01:25:34.526423 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:25:34.526446 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0159083 (* 1 = 0.0159083 loss)
    I0515 01:25:34.526461 23147 sgd_solver.cpp:106] Iteration 45400, lr = 5.53857e-05
    I0515 01:25:42.661743 23147 solver.cpp:228] Iteration 45500, loss = 0.0275284
    I0515 01:25:42.661795 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:25:42.661815 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0275284 (* 1 = 0.0275284 loss)
    I0515 01:25:42.661830 23147 sgd_solver.cpp:106] Iteration 45500, lr = 5.53108e-05
    I0515 01:25:50.801164 23147 solver.cpp:228] Iteration 45600, loss = 0.022867
    I0515 01:25:50.801301 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:25:50.801347 23147 solver.cpp:244]     Train net output #1: loss_c = 0.022867 (* 1 = 0.022867 loss)
    I0515 01:25:50.801370 23147 sgd_solver.cpp:106] Iteration 45600, lr = 5.52362e-05
    I0515 01:25:58.896282 23147 solver.cpp:228] Iteration 45700, loss = 0.0199782
    I0515 01:25:58.896333 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:25:58.896363 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0199782 (* 1 = 0.0199782 loss)
    I0515 01:25:58.896384 23147 sgd_solver.cpp:106] Iteration 45700, lr = 5.51618e-05
    I0515 01:26:06.986275 23147 solver.cpp:228] Iteration 45800, loss = 0.00535945
    I0515 01:26:06.986337 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:26:06.986366 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00535939 (* 1 = 0.00535939 loss)
    I0515 01:26:06.986387 23147 sgd_solver.cpp:106] Iteration 45800, lr = 5.50877e-05
    I0515 01:26:15.072309 23147 solver.cpp:228] Iteration 45900, loss = 0.0205956
    I0515 01:26:15.072368 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:26:15.072397 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0205955 (* 1 = 0.0205955 loss)
    I0515 01:26:15.072419 23147 sgd_solver.cpp:106] Iteration 45900, lr = 5.50137e-05
    I0515 01:26:23.080929 23147 solver.cpp:337] Iteration 46000, Testing net (#0)
    I0515 01:26:27.460253 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.833917
    I0515 01:26:27.460304 23147 solver.cpp:404]     Test net output #1: loss_c = 0.540366 (* 1 = 0.540366 loss)
    I0515 01:26:27.511642 23147 solver.cpp:228] Iteration 46000, loss = 0.0234169
    I0515 01:26:27.511678 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:26:27.511698 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0234168 (* 1 = 0.0234168 loss)
    I0515 01:26:27.511716 23147 sgd_solver.cpp:106] Iteration 46000, lr = 5.494e-05
    I0515 01:26:35.649777 23147 solver.cpp:228] Iteration 46100, loss = 0.0230678
    I0515 01:26:35.649828 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:26:35.649848 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0230678 (* 1 = 0.0230678 loss)
    I0515 01:26:35.649863 23147 sgd_solver.cpp:106] Iteration 46100, lr = 5.48666e-05
    I0515 01:26:43.784639 23147 solver.cpp:228] Iteration 46200, loss = 0.0287666
    I0515 01:26:43.784690 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:26:43.784710 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0287666 (* 1 = 0.0287666 loss)
    I0515 01:26:43.784725 23147 sgd_solver.cpp:106] Iteration 46200, lr = 5.47933e-05
    I0515 01:26:51.924046 23147 solver.cpp:228] Iteration 46300, loss = 0.0205899
    I0515 01:26:51.924095 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:26:51.924115 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0205899 (* 1 = 0.0205899 loss)
    I0515 01:26:51.924130 23147 sgd_solver.cpp:106] Iteration 46300, lr = 5.47203e-05
    I0515 01:27:00.066315 23147 solver.cpp:228] Iteration 46400, loss = 0.0259807
    I0515 01:27:00.066457 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:27:00.066501 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0259806 (* 1 = 0.0259806 loss)
    I0515 01:27:00.066525 23147 sgd_solver.cpp:106] Iteration 46400, lr = 5.46475e-05
    I0515 01:27:08.122997 23147 solver.cpp:228] Iteration 46500, loss = 0.00811163
    I0515 01:27:08.123036 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:27:08.123056 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00811159 (* 1 = 0.00811159 loss)
    I0515 01:27:08.123070 23147 sgd_solver.cpp:106] Iteration 46500, lr = 5.4575e-05
    I0515 01:27:16.177486 23147 solver.cpp:228] Iteration 46600, loss = 0.0227876
    I0515 01:27:16.177533 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:27:16.177552 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0227876 (* 1 = 0.0227876 loss)
    I0515 01:27:16.177567 23147 sgd_solver.cpp:106] Iteration 46600, lr = 5.45027e-05
    I0515 01:27:24.228705 23147 solver.cpp:228] Iteration 46700, loss = 0.0587201
    I0515 01:27:24.228747 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:27:24.228767 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0587201 (* 1 = 0.0587201 loss)
    I0515 01:27:24.228781 23147 sgd_solver.cpp:106] Iteration 46700, lr = 5.44305e-05
    I0515 01:27:32.284205 23147 solver.cpp:228] Iteration 46800, loss = 0.0389121
    I0515 01:27:32.284330 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:27:32.284368 23147 solver.cpp:244]     Train net output #1: loss_c = 0.038912 (* 1 = 0.038912 loss)
    I0515 01:27:32.284384 23147 sgd_solver.cpp:106] Iteration 46800, lr = 5.43587e-05
    I0515 01:27:40.424119 23147 solver.cpp:228] Iteration 46900, loss = 0.0133629
    I0515 01:27:40.424168 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:27:40.424188 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0133628 (* 1 = 0.0133628 loss)
    I0515 01:27:40.424204 23147 sgd_solver.cpp:106] Iteration 46900, lr = 5.4287e-05
    I0515 01:27:48.484297 23147 solver.cpp:337] Iteration 47000, Testing net (#0)
    I0515 01:27:52.896539 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.833167
    I0515 01:27:52.896591 23147 solver.cpp:404]     Test net output #1: loss_c = 0.525665 (* 1 = 0.525665 loss)
    I0515 01:27:52.952172 23147 solver.cpp:228] Iteration 47000, loss = 0.0332377
    I0515 01:27:52.952257 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:27:52.952288 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0332376 (* 1 = 0.0332376 loss)
    I0515 01:27:52.952316 23147 sgd_solver.cpp:106] Iteration 47000, lr = 5.42155e-05
    I0515 01:28:01.065354 23147 solver.cpp:228] Iteration 47100, loss = 0.0130688
    I0515 01:28:01.065402 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:28:01.065424 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0130687 (* 1 = 0.0130687 loss)
    I0515 01:28:01.065439 23147 sgd_solver.cpp:106] Iteration 47100, lr = 5.41443e-05
    I0515 01:28:09.206503 23147 solver.cpp:228] Iteration 47200, loss = 0.0480058
    I0515 01:28:09.206694 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:28:09.206715 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0480058 (* 1 = 0.0480058 loss)
    I0515 01:28:09.206730 23147 sgd_solver.cpp:106] Iteration 47200, lr = 5.40733e-05
    I0515 01:28:17.346638 23147 solver.cpp:228] Iteration 47300, loss = 0.0331139
    I0515 01:28:17.346683 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:28:17.346704 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0331138 (* 1 = 0.0331138 loss)
    I0515 01:28:17.346719 23147 sgd_solver.cpp:106] Iteration 47300, lr = 5.40025e-05
    I0515 01:28:25.487313 23147 solver.cpp:228] Iteration 47400, loss = 0.0132754
    I0515 01:28:25.487361 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:28:25.487383 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0132754 (* 1 = 0.0132754 loss)
    I0515 01:28:25.487397 23147 sgd_solver.cpp:106] Iteration 47400, lr = 5.39319e-05
    I0515 01:28:33.627933 23147 solver.cpp:228] Iteration 47500, loss = 0.0164152
    I0515 01:28:33.627982 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:28:33.628003 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0164152 (* 1 = 0.0164152 loss)
    I0515 01:28:33.628018 23147 sgd_solver.cpp:106] Iteration 47500, lr = 5.38616e-05
    I0515 01:28:41.769955 23147 solver.cpp:228] Iteration 47600, loss = 0.0130339
    I0515 01:28:41.770071 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:28:41.770094 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0130339 (* 1 = 0.0130339 loss)
    I0515 01:28:41.770109 23147 sgd_solver.cpp:106] Iteration 47600, lr = 5.37914e-05
    I0515 01:28:49.902845 23147 solver.cpp:228] Iteration 47700, loss = 0.0229093
    I0515 01:28:49.902897 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:28:49.902918 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0229092 (* 1 = 0.0229092 loss)
    I0515 01:28:49.902936 23147 sgd_solver.cpp:106] Iteration 47700, lr = 5.37215e-05
    I0515 01:28:58.044292 23147 solver.cpp:228] Iteration 47800, loss = 0.0236876
    I0515 01:28:58.044337 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:28:58.044356 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0236875 (* 1 = 0.0236875 loss)
    I0515 01:28:58.044370 23147 sgd_solver.cpp:106] Iteration 47800, lr = 5.36518e-05
    I0515 01:29:06.179656 23147 solver.cpp:228] Iteration 47900, loss = 0.0152192
    I0515 01:29:06.179704 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:29:06.179723 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0152191 (* 1 = 0.0152191 loss)
    I0515 01:29:06.179738 23147 sgd_solver.cpp:106] Iteration 47900, lr = 5.35823e-05
    I0515 01:29:14.239847 23147 solver.cpp:337] Iteration 48000, Testing net (#0)
    I0515 01:29:18.599251 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.836
    I0515 01:29:18.599303 23147 solver.cpp:404]     Test net output #1: loss_c = 0.50484 (* 1 = 0.50484 loss)
    I0515 01:29:18.650424 23147 solver.cpp:228] Iteration 48000, loss = 0.0239618
    I0515 01:29:18.650475 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:29:18.650496 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0239618 (* 1 = 0.0239618 loss)
    I0515 01:29:18.650516 23147 sgd_solver.cpp:106] Iteration 48000, lr = 5.3513e-05
    I0515 01:29:26.739212 23147 solver.cpp:228] Iteration 48100, loss = 0.0121385
    I0515 01:29:26.739274 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:29:26.739302 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0121384 (* 1 = 0.0121384 loss)
    I0515 01:29:26.739325 23147 sgd_solver.cpp:106] Iteration 48100, lr = 5.34439e-05
    I0515 01:29:34.826737 23147 solver.cpp:228] Iteration 48200, loss = 0.0152076
    I0515 01:29:34.826794 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:29:34.826824 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0152075 (* 1 = 0.0152075 loss)
    I0515 01:29:34.826846 23147 sgd_solver.cpp:106] Iteration 48200, lr = 5.3375e-05
    I0515 01:29:42.941608 23147 solver.cpp:228] Iteration 48300, loss = 0.0217252
    I0515 01:29:42.941656 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:29:42.941679 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0217251 (* 1 = 0.0217251 loss)
    I0515 01:29:42.941694 23147 sgd_solver.cpp:106] Iteration 48300, lr = 5.33063e-05
    I0515 01:29:51.083197 23147 solver.cpp:228] Iteration 48400, loss = 0.00765192
    I0515 01:29:51.083446 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:29:51.083492 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00765188 (* 1 = 0.00765188 loss)
    I0515 01:29:51.083528 23147 sgd_solver.cpp:106] Iteration 48400, lr = 5.32378e-05
    I0515 01:29:59.148473 23147 solver.cpp:228] Iteration 48500, loss = 0.0150612
    I0515 01:29:59.148529 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:29:59.148552 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0150611 (* 1 = 0.0150611 loss)
    I0515 01:29:59.148567 23147 sgd_solver.cpp:106] Iteration 48500, lr = 5.31696e-05
    I0515 01:30:07.287108 23147 solver.cpp:228] Iteration 48600, loss = 0.025743
    I0515 01:30:07.287156 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:30:07.287176 23147 solver.cpp:244]     Train net output #1: loss_c = 0.025743 (* 1 = 0.025743 loss)
    I0515 01:30:07.287190 23147 sgd_solver.cpp:106] Iteration 48600, lr = 5.31015e-05
    I0515 01:30:15.428277 23147 solver.cpp:228] Iteration 48700, loss = 0.0141812
    I0515 01:30:15.428320 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:30:15.428341 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0141812 (* 1 = 0.0141812 loss)
    I0515 01:30:15.428355 23147 sgd_solver.cpp:106] Iteration 48700, lr = 5.30336e-05
    I0515 01:30:23.529897 23147 solver.cpp:228] Iteration 48800, loss = 0.0125872
    I0515 01:30:23.530045 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:30:23.530089 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0125872 (* 1 = 0.0125872 loss)
    I0515 01:30:23.530114 23147 sgd_solver.cpp:106] Iteration 48800, lr = 5.2966e-05
    I0515 01:30:31.670832 23147 solver.cpp:228] Iteration 48900, loss = 0.0132488
    I0515 01:30:31.670883 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:30:31.670903 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0132487 (* 1 = 0.0132487 loss)
    I0515 01:30:31.670918 23147 sgd_solver.cpp:106] Iteration 48900, lr = 5.28985e-05
    I0515 01:30:39.692734 23147 solver.cpp:337] Iteration 49000, Testing net (#0)
    I0515 01:30:44.057940 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.827417
    I0515 01:30:44.057999 23147 solver.cpp:404]     Test net output #1: loss_c = 0.532431 (* 1 = 0.532431 loss)
    I0515 01:30:44.113392 23147 solver.cpp:228] Iteration 49000, loss = 0.0752709
    I0515 01:30:44.113456 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:30:44.113486 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0752708 (* 1 = 0.0752708 loss)
    I0515 01:30:44.113512 23147 sgd_solver.cpp:106] Iteration 49000, lr = 5.28313e-05
    I0515 01:30:52.224941 23147 solver.cpp:228] Iteration 49100, loss = 0.0103378
    I0515 01:30:52.224992 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:30:52.225011 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0103377 (* 1 = 0.0103377 loss)
    I0515 01:30:52.225026 23147 sgd_solver.cpp:106] Iteration 49100, lr = 5.27642e-05
    I0515 01:31:00.356192 23147 solver.cpp:228] Iteration 49200, loss = 0.00918273
    I0515 01:31:00.356364 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:31:00.356395 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00918266 (* 1 = 0.00918266 loss)
    I0515 01:31:00.356432 23147 sgd_solver.cpp:106] Iteration 49200, lr = 5.26973e-05
    I0515 01:31:08.445128 23147 solver.cpp:228] Iteration 49300, loss = 0.0475017
    I0515 01:31:08.445183 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:31:08.445212 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0475016 (* 1 = 0.0475016 loss)
    I0515 01:31:08.445233 23147 sgd_solver.cpp:106] Iteration 49300, lr = 5.26307e-05
    I0515 01:31:16.535883 23147 solver.cpp:228] Iteration 49400, loss = 0.0105927
    I0515 01:31:16.535943 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:31:16.535972 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0105926 (* 1 = 0.0105926 loss)
    I0515 01:31:16.535994 23147 sgd_solver.cpp:106] Iteration 49400, lr = 5.25642e-05
    I0515 01:31:24.623025 23147 solver.cpp:228] Iteration 49500, loss = 0.0157335
    I0515 01:31:24.623086 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:31:24.623116 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0157334 (* 1 = 0.0157334 loss)
    I0515 01:31:24.623138 23147 sgd_solver.cpp:106] Iteration 49500, lr = 5.24979e-05
    I0515 01:31:32.711836 23147 solver.cpp:228] Iteration 49600, loss = 0.0111841
    I0515 01:31:32.711936 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:31:32.711966 23147 solver.cpp:244]     Train net output #1: loss_c = 0.011184 (* 1 = 0.011184 loss)
    I0515 01:31:32.711988 23147 sgd_solver.cpp:106] Iteration 49600, lr = 5.24319e-05
    I0515 01:31:40.807140 23147 solver.cpp:228] Iteration 49700, loss = 0.0183719
    I0515 01:31:40.807196 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:31:40.807225 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0183719 (* 1 = 0.0183719 loss)
    I0515 01:31:40.807246 23147 sgd_solver.cpp:106] Iteration 49700, lr = 5.2366e-05
    I0515 01:31:48.891526 23147 solver.cpp:228] Iteration 49800, loss = 0.055916
    I0515 01:31:48.891587 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:31:48.891615 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0559159 (* 1 = 0.0559159 loss)
    I0515 01:31:48.891638 23147 sgd_solver.cpp:106] Iteration 49800, lr = 5.23003e-05
    I0515 01:31:56.977913 23147 solver.cpp:228] Iteration 49900, loss = 0.0222698
    I0515 01:31:56.977973 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:31:56.978003 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0222698 (* 1 = 0.0222698 loss)
    I0515 01:31:56.978024 23147 sgd_solver.cpp:106] Iteration 49900, lr = 5.22348e-05
    I0515 01:32:04.984802 23147 solver.cpp:454] Snapshotting to binary proto file dvia_train_iter_50000.caffemodel
    I0515 01:32:05.015193 23147 sgd_solver.cpp:273] Snapshotting solver state to binary proto file dvia_train_iter_50000.solverstate
    I0515 01:32:05.017134 23147 solver.cpp:337] Iteration 50000, Testing net (#0)
    I0515 01:32:09.358764 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.831917
    I0515 01:32:09.358830 23147 solver.cpp:404]     Test net output #1: loss_c = 0.534062 (* 1 = 0.534062 loss)
    I0515 01:32:09.413784 23147 solver.cpp:228] Iteration 50000, loss = 0.0221028
    I0515 01:32:09.413830 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:32:09.413857 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0221027 (* 1 = 0.0221027 loss)
    I0515 01:32:09.413882 23147 sgd_solver.cpp:106] Iteration 50000, lr = 5.21695e-05
    I0515 01:32:17.551647 23147 solver.cpp:228] Iteration 50100, loss = 0.0135175
    I0515 01:32:17.551692 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:32:17.551712 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0135174 (* 1 = 0.0135174 loss)
    I0515 01:32:17.551725 23147 sgd_solver.cpp:106] Iteration 50100, lr = 5.21044e-05
    I0515 01:32:25.640102 23147 solver.cpp:228] Iteration 50200, loss = 0.0371105
    I0515 01:32:25.640151 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:32:25.640172 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0371105 (* 1 = 0.0371105 loss)
    I0515 01:32:25.640187 23147 sgd_solver.cpp:106] Iteration 50200, lr = 5.20394e-05
    I0515 01:32:33.779943 23147 solver.cpp:228] Iteration 50300, loss = 0.0377832
    I0515 01:32:33.779992 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:32:33.780012 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0377831 (* 1 = 0.0377831 loss)
    I0515 01:32:33.780028 23147 sgd_solver.cpp:106] Iteration 50300, lr = 5.19747e-05
    I0515 01:32:41.918498 23147 solver.cpp:228] Iteration 50400, loss = 0.0187989
    I0515 01:32:41.918756 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:32:41.918807 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0187988 (* 1 = 0.0187988 loss)
    I0515 01:32:41.918831 23147 sgd_solver.cpp:106] Iteration 50400, lr = 5.19101e-05
    I0515 01:32:50.059489 23147 solver.cpp:228] Iteration 50500, loss = 0.0160933
    I0515 01:32:50.059545 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:32:50.059566 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0160933 (* 1 = 0.0160933 loss)
    I0515 01:32:50.059581 23147 sgd_solver.cpp:106] Iteration 50500, lr = 5.18458e-05
    I0515 01:32:58.199750 23147 solver.cpp:228] Iteration 50600, loss = 0.0117175
    I0515 01:32:58.199800 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:32:58.199820 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0117175 (* 1 = 0.0117175 loss)
    I0515 01:32:58.199836 23147 sgd_solver.cpp:106] Iteration 50600, lr = 5.17816e-05
    I0515 01:33:06.342872 23147 solver.cpp:228] Iteration 50700, loss = 0.0247564
    I0515 01:33:06.342929 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:33:06.342958 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0247564 (* 1 = 0.0247564 loss)
    I0515 01:33:06.342980 23147 sgd_solver.cpp:106] Iteration 50700, lr = 5.17176e-05
    I0515 01:33:14.430969 23147 solver.cpp:228] Iteration 50800, loss = 0.00704864
    I0515 01:33:14.431229 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:33:14.431274 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00704859 (* 1 = 0.00704859 loss)
    I0515 01:33:14.431301 23147 sgd_solver.cpp:106] Iteration 50800, lr = 5.16538e-05
    I0515 01:33:22.569584 23147 solver.cpp:228] Iteration 50900, loss = 0.0121771
    I0515 01:33:22.569636 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:33:22.569659 23147 solver.cpp:244]     Train net output #1: loss_c = 0.012177 (* 1 = 0.012177 loss)
    I0515 01:33:22.569674 23147 sgd_solver.cpp:106] Iteration 50900, lr = 5.15902e-05
    I0515 01:33:30.626557 23147 solver.cpp:337] Iteration 51000, Testing net (#0)
    I0515 01:33:34.990746 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.835333
    I0515 01:33:34.990816 23147 solver.cpp:404]     Test net output #1: loss_c = 0.53281 (* 1 = 0.53281 loss)
    I0515 01:33:35.046947 23147 solver.cpp:228] Iteration 51000, loss = 0.00924796
    I0515 01:33:35.047019 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:33:35.047049 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0092479 (* 1 = 0.0092479 loss)
    I0515 01:33:35.047075 23147 sgd_solver.cpp:106] Iteration 51000, lr = 5.15267e-05
    I0515 01:33:43.163624 23147 solver.cpp:228] Iteration 51100, loss = 0.0230289
    I0515 01:33:43.163677 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:33:43.163705 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0230288 (* 1 = 0.0230288 loss)
    I0515 01:33:43.163728 23147 sgd_solver.cpp:106] Iteration 51100, lr = 5.14635e-05
    I0515 01:33:51.249464 23147 solver.cpp:228] Iteration 51200, loss = 0.020653
    I0515 01:33:51.249706 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:33:51.249752 23147 solver.cpp:244]     Train net output #1: loss_c = 0.020653 (* 1 = 0.020653 loss)
    I0515 01:33:51.249776 23147 sgd_solver.cpp:106] Iteration 51200, lr = 5.14004e-05
    I0515 01:33:59.378821 23147 solver.cpp:228] Iteration 51300, loss = 0.00699644
    I0515 01:33:59.378868 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:33:59.378890 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0069964 (* 1 = 0.0069964 loss)
    I0515 01:33:59.378903 23147 sgd_solver.cpp:106] Iteration 51300, lr = 5.13375e-05
    I0515 01:34:07.518615 23147 solver.cpp:228] Iteration 51400, loss = 0.0380322
    I0515 01:34:07.518666 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:34:07.518688 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0380321 (* 1 = 0.0380321 loss)
    I0515 01:34:07.518703 23147 sgd_solver.cpp:106] Iteration 51400, lr = 5.12748e-05
    I0515 01:34:15.608116 23147 solver.cpp:228] Iteration 51500, loss = 0.0075941
    I0515 01:34:15.608165 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:34:15.608186 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00759406 (* 1 = 0.00759406 loss)
    I0515 01:34:15.608203 23147 sgd_solver.cpp:106] Iteration 51500, lr = 5.12122e-05
    I0515 01:34:23.697515 23147 solver.cpp:228] Iteration 51600, loss = 0.0330003
    I0515 01:34:23.697624 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:34:23.697654 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0330003 (* 1 = 0.0330003 loss)
    I0515 01:34:23.697675 23147 sgd_solver.cpp:106] Iteration 51600, lr = 5.11499e-05
    I0515 01:34:31.784121 23147 solver.cpp:228] Iteration 51700, loss = 0.0182464
    I0515 01:34:31.784176 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:34:31.784205 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0182464 (* 1 = 0.0182464 loss)
    I0515 01:34:31.784227 23147 sgd_solver.cpp:106] Iteration 51700, lr = 5.10877e-05
    I0515 01:34:39.870121 23147 solver.cpp:228] Iteration 51800, loss = 0.0316351
    I0515 01:34:39.870173 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:34:39.870201 23147 solver.cpp:244]     Train net output #1: loss_c = 0.031635 (* 1 = 0.031635 loss)
    I0515 01:34:39.870224 23147 sgd_solver.cpp:106] Iteration 51800, lr = 5.10257e-05
    I0515 01:34:47.954686 23147 solver.cpp:228] Iteration 51900, loss = 0.0219377
    I0515 01:34:47.954741 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:34:47.954768 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0219377 (* 1 = 0.0219377 loss)
    I0515 01:34:47.954789 23147 sgd_solver.cpp:106] Iteration 51900, lr = 5.09638e-05
    I0515 01:34:55.967308 23147 solver.cpp:337] Iteration 52000, Testing net (#0)
    I0515 01:35:00.374460 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.820083
    I0515 01:35:00.374503 23147 solver.cpp:404]     Test net output #1: loss_c = 0.560616 (* 1 = 0.560616 loss)
    I0515 01:35:00.429530 23147 solver.cpp:228] Iteration 52000, loss = 0.0111079
    I0515 01:35:00.429590 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:35:00.429620 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0111078 (* 1 = 0.0111078 loss)
    I0515 01:35:00.429646 23147 sgd_solver.cpp:106] Iteration 52000, lr = 5.09022e-05
    I0515 01:35:08.538494 23147 solver.cpp:228] Iteration 52100, loss = 0.0156561
    I0515 01:35:08.538554 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:35:08.538583 23147 solver.cpp:244]     Train net output #1: loss_c = 0.015656 (* 1 = 0.015656 loss)
    I0515 01:35:08.538605 23147 sgd_solver.cpp:106] Iteration 52100, lr = 5.08407e-05
    I0515 01:35:16.659014 23147 solver.cpp:228] Iteration 52200, loss = 0.0308851
    I0515 01:35:16.659070 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:35:16.659097 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0308851 (* 1 = 0.0308851 loss)
    I0515 01:35:16.659119 23147 sgd_solver.cpp:106] Iteration 52200, lr = 5.07794e-05
    I0515 01:35:24.763612 23147 solver.cpp:228] Iteration 52300, loss = 0.0178963
    I0515 01:35:24.763665 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:35:24.763694 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0178963 (* 1 = 0.0178963 loss)
    I0515 01:35:24.763715 23147 sgd_solver.cpp:106] Iteration 52300, lr = 5.07182e-05
    I0515 01:35:32.851361 23147 solver.cpp:228] Iteration 52400, loss = 0.0111974
    I0515 01:35:32.851591 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:35:32.851639 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0111974 (* 1 = 0.0111974 loss)
    I0515 01:35:32.851662 23147 sgd_solver.cpp:106] Iteration 52400, lr = 5.06572e-05
    I0515 01:35:40.980305 23147 solver.cpp:228] Iteration 52500, loss = 0.00951491
    I0515 01:35:40.980351 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:35:40.980373 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00951487 (* 1 = 0.00951487 loss)
    I0515 01:35:40.980388 23147 sgd_solver.cpp:106] Iteration 52500, lr = 5.05964e-05
    I0515 01:35:49.122349 23147 solver.cpp:228] Iteration 52600, loss = 0.0177179
    I0515 01:35:49.122400 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:35:49.122419 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0177179 (* 1 = 0.0177179 loss)
    I0515 01:35:49.122434 23147 sgd_solver.cpp:106] Iteration 52600, lr = 5.05358e-05
    I0515 01:35:57.213757 23147 solver.cpp:228] Iteration 52700, loss = 0.0151506
    I0515 01:35:57.213811 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:35:57.213840 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0151506 (* 1 = 0.0151506 loss)
    I0515 01:35:57.213860 23147 sgd_solver.cpp:106] Iteration 52700, lr = 5.04753e-05
    I0515 01:36:05.303555 23147 solver.cpp:228] Iteration 52800, loss = 0.021099
    I0515 01:36:05.303695 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:36:05.303740 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0210989 (* 1 = 0.0210989 loss)
    I0515 01:36:05.303766 23147 sgd_solver.cpp:106] Iteration 52800, lr = 5.04151e-05
    I0515 01:36:13.435335 23147 solver.cpp:228] Iteration 52900, loss = 0.0124883
    I0515 01:36:13.435376 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:36:13.435400 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0124883 (* 1 = 0.0124883 loss)
    I0515 01:36:13.435415 23147 sgd_solver.cpp:106] Iteration 52900, lr = 5.03549e-05
    I0515 01:36:21.495435 23147 solver.cpp:337] Iteration 53000, Testing net (#0)
    I0515 01:36:25.889503 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.833083
    I0515 01:36:25.889559 23147 solver.cpp:404]     Test net output #1: loss_c = 0.527911 (* 1 = 0.527911 loss)
    I0515 01:36:25.941184 23147 solver.cpp:228] Iteration 53000, loss = 0.0123367
    I0515 01:36:25.941226 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:36:25.941246 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0123367 (* 1 = 0.0123367 loss)
    I0515 01:36:25.941262 23147 sgd_solver.cpp:106] Iteration 53000, lr = 5.0295e-05
    I0515 01:36:34.075795 23147 solver.cpp:228] Iteration 53100, loss = 0.00812962
    I0515 01:36:34.075848 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:36:34.075870 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00812958 (* 1 = 0.00812958 loss)
    I0515 01:36:34.075886 23147 sgd_solver.cpp:106] Iteration 53100, lr = 5.02352e-05
    I0515 01:36:42.216312 23147 solver.cpp:228] Iteration 53200, loss = 0.0129988
    I0515 01:36:42.216593 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:36:42.216639 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0129988 (* 1 = 0.0129988 loss)
    I0515 01:36:42.216662 23147 sgd_solver.cpp:106] Iteration 53200, lr = 5.01756e-05
    I0515 01:36:50.355520 23147 solver.cpp:228] Iteration 53300, loss = 0.0181277
    I0515 01:36:50.355566 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:36:50.355586 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0181277 (* 1 = 0.0181277 loss)
    I0515 01:36:50.355600 23147 sgd_solver.cpp:106] Iteration 53300, lr = 5.01161e-05
    I0515 01:36:58.493613 23147 solver.cpp:228] Iteration 53400, loss = 0.00730952
    I0515 01:36:58.493659 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:36:58.493680 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00730949 (* 1 = 0.00730949 loss)
    I0515 01:36:58.493695 23147 sgd_solver.cpp:106] Iteration 53400, lr = 5.00568e-05
    I0515 01:37:06.632771 23147 solver.cpp:228] Iteration 53500, loss = 0.0152764
    I0515 01:37:06.632817 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:37:06.632836 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0152764 (* 1 = 0.0152764 loss)
    I0515 01:37:06.632850 23147 sgd_solver.cpp:106] Iteration 53500, lr = 4.99977e-05
    I0515 01:37:14.763803 23147 solver.cpp:228] Iteration 53600, loss = 0.0275664
    I0515 01:37:14.763944 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:37:14.763991 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0275664 (* 1 = 0.0275664 loss)
    I0515 01:37:14.764015 23147 sgd_solver.cpp:106] Iteration 53600, lr = 4.99387e-05
    I0515 01:37:22.854295 23147 solver.cpp:228] Iteration 53700, loss = 0.00808468
    I0515 01:37:22.854357 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:37:22.854385 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00808463 (* 1 = 0.00808463 loss)
    I0515 01:37:22.854408 23147 sgd_solver.cpp:106] Iteration 53700, lr = 4.98799e-05
    I0515 01:37:30.941689 23147 solver.cpp:228] Iteration 53800, loss = 0.00878932
    I0515 01:37:30.941741 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:37:30.941771 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00878926 (* 1 = 0.00878926 loss)
    I0515 01:37:30.941792 23147 sgd_solver.cpp:106] Iteration 53800, lr = 4.98212e-05
    I0515 01:37:39.044610 23147 solver.cpp:228] Iteration 53900, loss = 0.00955431
    I0515 01:37:39.044664 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:37:39.044684 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00955424 (* 1 = 0.00955424 loss)
    I0515 01:37:39.044698 23147 sgd_solver.cpp:106] Iteration 53900, lr = 4.97627e-05
    I0515 01:37:47.108085 23147 solver.cpp:337] Iteration 54000, Testing net (#0)
    I0515 01:37:51.528509 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.8325
    I0515 01:37:51.528565 23147 solver.cpp:404]     Test net output #1: loss_c = 0.501918 (* 1 = 0.501918 loss)
    I0515 01:37:51.580166 23147 solver.cpp:228] Iteration 54000, loss = 0.0144816
    I0515 01:37:51.580193 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:37:51.580210 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0144816 (* 1 = 0.0144816 loss)
    I0515 01:37:51.580230 23147 sgd_solver.cpp:106] Iteration 54000, lr = 4.97044e-05
    I0515 01:37:59.719039 23147 solver.cpp:228] Iteration 54100, loss = 0.00757451
    I0515 01:37:59.719090 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:37:59.719110 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00757443 (* 1 = 0.00757443 loss)
    I0515 01:37:59.719125 23147 sgd_solver.cpp:106] Iteration 54100, lr = 4.96463e-05
    I0515 01:38:07.808154 23147 solver.cpp:228] Iteration 54200, loss = 0.0150387
    I0515 01:38:07.808212 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:38:07.808240 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0150386 (* 1 = 0.0150386 loss)
    I0515 01:38:07.808262 23147 sgd_solver.cpp:106] Iteration 54200, lr = 4.95882e-05
    I0515 01:38:15.928447 23147 solver.cpp:228] Iteration 54300, loss = 0.024232
    I0515 01:38:15.928490 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:38:15.928510 23147 solver.cpp:244]     Train net output #1: loss_c = 0.024232 (* 1 = 0.024232 loss)
    I0515 01:38:15.928526 23147 sgd_solver.cpp:106] Iteration 54300, lr = 4.95304e-05
    I0515 01:38:24.069329 23147 solver.cpp:228] Iteration 54400, loss = 0.0276376
    I0515 01:38:24.069492 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:38:24.069538 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0276376 (* 1 = 0.0276376 loss)
    I0515 01:38:24.069556 23147 sgd_solver.cpp:106] Iteration 54400, lr = 4.94727e-05
    I0515 01:38:32.208664 23147 solver.cpp:228] Iteration 54500, loss = 0.00718013
    I0515 01:38:32.208714 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:38:32.208735 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00718005 (* 1 = 0.00718005 loss)
    I0515 01:38:32.208750 23147 sgd_solver.cpp:106] Iteration 54500, lr = 4.94152e-05
    I0515 01:38:40.350471 23147 solver.cpp:228] Iteration 54600, loss = 0.014041
    I0515 01:38:40.350513 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:38:40.350533 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0140409 (* 1 = 0.0140409 loss)
    I0515 01:38:40.350548 23147 sgd_solver.cpp:106] Iteration 54600, lr = 4.93578e-05
    I0515 01:38:48.491034 23147 solver.cpp:228] Iteration 54700, loss = 0.0139885
    I0515 01:38:48.491080 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:38:48.491099 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0139884 (* 1 = 0.0139884 loss)
    I0515 01:38:48.491113 23147 sgd_solver.cpp:106] Iteration 54700, lr = 4.93006e-05
    I0515 01:38:56.631481 23147 solver.cpp:228] Iteration 54800, loss = 0.0116787
    I0515 01:38:56.631587 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:38:56.631608 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0116786 (* 1 = 0.0116786 loss)
    I0515 01:38:56.631623 23147 sgd_solver.cpp:106] Iteration 54800, lr = 4.92435e-05
    I0515 01:39:04.772974 23147 solver.cpp:228] Iteration 54900, loss = 0.0137508
    I0515 01:39:04.773025 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:39:04.773046 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0137507 (* 1 = 0.0137507 loss)
    I0515 01:39:04.773061 23147 sgd_solver.cpp:106] Iteration 54900, lr = 4.91866e-05
    I0515 01:39:12.834081 23147 solver.cpp:337] Iteration 55000, Testing net (#0)
    I0515 01:39:17.248795 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.82925
    I0515 01:39:17.248852 23147 solver.cpp:404]     Test net output #1: loss_c = 0.514292 (* 1 = 0.514292 loss)
    I0515 01:39:17.302772 23147 solver.cpp:228] Iteration 55000, loss = 0.0335095
    I0515 01:39:17.302824 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:39:17.302852 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0335094 (* 1 = 0.0335094 loss)
    I0515 01:39:17.302876 23147 sgd_solver.cpp:106] Iteration 55000, lr = 4.91298e-05
    I0515 01:39:25.390432 23147 solver.cpp:228] Iteration 55100, loss = 0.0249152
    I0515 01:39:25.390488 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:39:25.390517 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0249151 (* 1 = 0.0249151 loss)
    I0515 01:39:25.390538 23147 sgd_solver.cpp:106] Iteration 55100, lr = 4.90732e-05
    I0515 01:39:33.476352 23147 solver.cpp:228] Iteration 55200, loss = 0.0261659
    I0515 01:39:33.476505 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:39:33.476552 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0261658 (* 1 = 0.0261658 loss)
    I0515 01:39:33.476577 23147 sgd_solver.cpp:106] Iteration 55200, lr = 4.90167e-05
    I0515 01:39:41.609552 23147 solver.cpp:228] Iteration 55300, loss = 0.02243
    I0515 01:39:41.609606 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:39:41.609627 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0224299 (* 1 = 0.0224299 loss)
    I0515 01:39:41.609645 23147 sgd_solver.cpp:106] Iteration 55300, lr = 4.89604e-05
    I0515 01:39:49.744173 23147 solver.cpp:228] Iteration 55400, loss = 0.00641121
    I0515 01:39:49.744211 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:39:49.744230 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00641111 (* 1 = 0.00641111 loss)
    I0515 01:39:49.744245 23147 sgd_solver.cpp:106] Iteration 55400, lr = 4.89043e-05
    I0515 01:39:57.883235 23147 solver.cpp:228] Iteration 55500, loss = 0.0149371
    I0515 01:39:57.883285 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:39:57.883306 23147 solver.cpp:244]     Train net output #1: loss_c = 0.014937 (* 1 = 0.014937 loss)
    I0515 01:39:57.883319 23147 sgd_solver.cpp:106] Iteration 55500, lr = 4.88483e-05
    I0515 01:40:06.023195 23147 solver.cpp:228] Iteration 55600, loss = 0.0315449
    I0515 01:40:06.023640 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:40:06.023694 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0315448 (* 1 = 0.0315448 loss)
    I0515 01:40:06.023717 23147 sgd_solver.cpp:106] Iteration 55600, lr = 4.87924e-05
    I0515 01:40:14.114830 23147 solver.cpp:228] Iteration 55700, loss = 0.0118852
    I0515 01:40:14.114876 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:40:14.114897 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0118851 (* 1 = 0.0118851 loss)
    I0515 01:40:14.114912 23147 sgd_solver.cpp:106] Iteration 55700, lr = 4.87367e-05
    I0515 01:40:22.255579 23147 solver.cpp:228] Iteration 55800, loss = 0.0125877
    I0515 01:40:22.255622 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:40:22.255645 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0125876 (* 1 = 0.0125876 loss)
    I0515 01:40:22.255661 23147 sgd_solver.cpp:106] Iteration 55800, lr = 4.86811e-05
    I0515 01:40:30.351485 23147 solver.cpp:228] Iteration 55900, loss = 0.0248616
    I0515 01:40:30.351541 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:40:30.351570 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0248615 (* 1 = 0.0248615 loss)
    I0515 01:40:30.351593 23147 sgd_solver.cpp:106] Iteration 55900, lr = 4.86257e-05
    I0515 01:40:38.364210 23147 solver.cpp:337] Iteration 56000, Testing net (#0)
    I0515 01:40:42.772572 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.8295
    I0515 01:40:42.772620 23147 solver.cpp:404]     Test net output #1: loss_c = 0.529153 (* 1 = 0.529153 loss)
    I0515 01:40:42.824064 23147 solver.cpp:228] Iteration 56000, loss = 0.0181277
    I0515 01:40:42.824131 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:40:42.824152 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0181276 (* 1 = 0.0181276 loss)
    I0515 01:40:42.824170 23147 sgd_solver.cpp:106] Iteration 56000, lr = 4.85704e-05
    I0515 01:40:50.962031 23147 solver.cpp:228] Iteration 56100, loss = 0.00927626
    I0515 01:40:50.962079 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:40:50.962098 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00927617 (* 1 = 0.00927617 loss)
    I0515 01:40:50.962113 23147 sgd_solver.cpp:106] Iteration 56100, lr = 4.85153e-05
    I0515 01:40:59.101267 23147 solver.cpp:228] Iteration 56200, loss = 0.0360695
    I0515 01:40:59.101320 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:40:59.101343 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0360694 (* 1 = 0.0360694 loss)
    I0515 01:40:59.101357 23147 sgd_solver.cpp:106] Iteration 56200, lr = 4.84603e-05
    I0515 01:41:07.243316 23147 solver.cpp:228] Iteration 56300, loss = 0.00914851
    I0515 01:41:07.243366 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:41:07.243389 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00914839 (* 1 = 0.00914839 loss)
    I0515 01:41:07.243404 23147 sgd_solver.cpp:106] Iteration 56300, lr = 4.84055e-05
    I0515 01:41:15.383177 23147 solver.cpp:228] Iteration 56400, loss = 0.0339183
    I0515 01:41:15.383383 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:41:15.383405 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0339182 (* 1 = 0.0339182 loss)
    I0515 01:41:15.383437 23147 sgd_solver.cpp:106] Iteration 56400, lr = 4.83508e-05
    I0515 01:41:23.524862 23147 solver.cpp:228] Iteration 56500, loss = 0.0219187
    I0515 01:41:23.524911 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:41:23.524932 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0219186 (* 1 = 0.0219186 loss)
    I0515 01:41:23.524946 23147 sgd_solver.cpp:106] Iteration 56500, lr = 4.82963e-05
    I0515 01:41:31.666854 23147 solver.cpp:228] Iteration 56600, loss = 0.0272283
    I0515 01:41:31.666900 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:41:31.666920 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0272282 (* 1 = 0.0272282 loss)
    I0515 01:41:31.666935 23147 sgd_solver.cpp:106] Iteration 56600, lr = 4.82419e-05
    I0515 01:41:39.804689 23147 solver.cpp:228] Iteration 56700, loss = 0.0197117
    I0515 01:41:39.804736 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:41:39.804756 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0197116 (* 1 = 0.0197116 loss)
    I0515 01:41:39.804770 23147 sgd_solver.cpp:106] Iteration 56700, lr = 4.81876e-05
    I0515 01:41:47.934674 23147 solver.cpp:228] Iteration 56800, loss = 0.0177401
    I0515 01:41:47.934814 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:41:47.934859 23147 solver.cpp:244]     Train net output #1: loss_c = 0.01774 (* 1 = 0.01774 loss)
    I0515 01:41:47.934882 23147 sgd_solver.cpp:106] Iteration 56800, lr = 4.81335e-05
    I0515 01:41:56.074218 23147 solver.cpp:228] Iteration 56900, loss = 0.0216679
    I0515 01:41:56.074259 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:41:56.074280 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0216678 (* 1 = 0.0216678 loss)
    I0515 01:41:56.074295 23147 sgd_solver.cpp:106] Iteration 56900, lr = 4.80795e-05
    I0515 01:42:04.123546 23147 solver.cpp:337] Iteration 57000, Testing net (#0)
    I0515 01:42:08.525025 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.823417
    I0515 01:42:08.525074 23147 solver.cpp:404]     Test net output #1: loss_c = 0.556211 (* 1 = 0.556211 loss)
    I0515 01:42:08.577258 23147 solver.cpp:228] Iteration 57000, loss = 0.0396872
    I0515 01:42:08.577281 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:42:08.577299 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0396871 (* 1 = 0.0396871 loss)
    I0515 01:42:08.577316 23147 sgd_solver.cpp:106] Iteration 57000, lr = 4.80257e-05
    I0515 01:42:16.708078 23147 solver.cpp:228] Iteration 57100, loss = 0.0169228
    I0515 01:42:16.708120 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:42:16.708139 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0169227 (* 1 = 0.0169227 loss)
    I0515 01:42:16.708154 23147 sgd_solver.cpp:106] Iteration 57100, lr = 4.7972e-05
    I0515 01:42:24.849709 23147 solver.cpp:228] Iteration 57200, loss = 0.0181563
    I0515 01:42:24.849805 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:42:24.849825 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0181561 (* 1 = 0.0181561 loss)
    I0515 01:42:24.849839 23147 sgd_solver.cpp:106] Iteration 57200, lr = 4.79185e-05
    I0515 01:42:32.983397 23147 solver.cpp:228] Iteration 57300, loss = 0.0195465
    I0515 01:42:32.983441 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:42:32.983460 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0195464 (* 1 = 0.0195464 loss)
    I0515 01:42:32.983475 23147 sgd_solver.cpp:106] Iteration 57300, lr = 4.78651e-05
    I0515 01:42:41.118504 23147 solver.cpp:228] Iteration 57400, loss = 0.0127275
    I0515 01:42:41.118549 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:42:41.118568 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0127274 (* 1 = 0.0127274 loss)
    I0515 01:42:41.118584 23147 sgd_solver.cpp:106] Iteration 57400, lr = 4.78118e-05
    I0515 01:42:49.249124 23147 solver.cpp:228] Iteration 57500, loss = 0.0158205
    I0515 01:42:49.249174 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:42:49.249194 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0158204 (* 1 = 0.0158204 loss)
    I0515 01:42:49.249209 23147 sgd_solver.cpp:106] Iteration 57500, lr = 4.77587e-05
    I0515 01:42:57.390008 23147 solver.cpp:228] Iteration 57600, loss = 0.015307
    I0515 01:42:57.390148 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:42:57.390168 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0153069 (* 1 = 0.0153069 loss)
    I0515 01:42:57.390183 23147 sgd_solver.cpp:106] Iteration 57600, lr = 4.77057e-05
    I0515 01:43:05.530123 23147 solver.cpp:228] Iteration 57700, loss = 0.00849359
    I0515 01:43:05.530171 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:43:05.530191 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00849349 (* 1 = 0.00849349 loss)
    I0515 01:43:05.530205 23147 sgd_solver.cpp:106] Iteration 57700, lr = 4.76528e-05
    I0515 01:43:13.671363 23147 solver.cpp:228] Iteration 57800, loss = 0.00848664
    I0515 01:43:13.671407 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:43:13.671427 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00848654 (* 1 = 0.00848654 loss)
    I0515 01:43:13.671442 23147 sgd_solver.cpp:106] Iteration 57800, lr = 4.76001e-05
    I0515 01:43:21.810991 23147 solver.cpp:228] Iteration 57900, loss = 0.017039
    I0515 01:43:21.811036 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:43:21.811055 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0170389 (* 1 = 0.0170389 loss)
    I0515 01:43:21.811070 23147 sgd_solver.cpp:106] Iteration 57900, lr = 4.75475e-05
    I0515 01:43:29.870647 23147 solver.cpp:337] Iteration 58000, Testing net (#0)
    I0515 01:43:34.281905 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.832666
    I0515 01:43:34.281962 23147 solver.cpp:404]     Test net output #1: loss_c = 0.527437 (* 1 = 0.527437 loss)
    I0515 01:43:34.333329 23147 solver.cpp:228] Iteration 58000, loss = 0.0390535
    I0515 01:43:34.333369 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:43:34.333389 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0390535 (* 1 = 0.0390535 loss)
    I0515 01:43:34.333406 23147 sgd_solver.cpp:106] Iteration 58000, lr = 4.7495e-05
    I0515 01:43:42.471809 23147 solver.cpp:228] Iteration 58100, loss = 0.0345135
    I0515 01:43:42.471855 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:43:42.471875 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0345134 (* 1 = 0.0345134 loss)
    I0515 01:43:42.471890 23147 sgd_solver.cpp:106] Iteration 58100, lr = 4.74427e-05
    I0515 01:43:50.580651 23147 solver.cpp:228] Iteration 58200, loss = 0.0172484
    I0515 01:43:50.580710 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:43:50.580739 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0172483 (* 1 = 0.0172483 loss)
    I0515 01:43:50.580761 23147 sgd_solver.cpp:106] Iteration 58200, lr = 4.73905e-05
    I0515 01:43:58.692775 23147 solver.cpp:228] Iteration 58300, loss = 0.0188957
    I0515 01:43:58.692836 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:43:58.692864 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0188956 (* 1 = 0.0188956 loss)
    I0515 01:43:58.692886 23147 sgd_solver.cpp:106] Iteration 58300, lr = 4.73385e-05
    I0515 01:44:06.783967 23147 solver.cpp:228] Iteration 58400, loss = 0.0398637
    I0515 01:44:06.784129 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 01:44:06.784173 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0398636 (* 1 = 0.0398636 loss)
    I0515 01:44:06.784198 23147 sgd_solver.cpp:106] Iteration 58400, lr = 4.72866e-05
    I0515 01:44:14.924203 23147 solver.cpp:228] Iteration 58500, loss = 0.027076
    I0515 01:44:14.924254 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:44:14.924274 23147 solver.cpp:244]     Train net output #1: loss_c = 0.027076 (* 1 = 0.027076 loss)
    I0515 01:44:14.924289 23147 sgd_solver.cpp:106] Iteration 58500, lr = 4.72348e-05
    I0515 01:44:23.061117 23147 solver.cpp:228] Iteration 58600, loss = 0.0147429
    I0515 01:44:23.061156 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:44:23.061179 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0147429 (* 1 = 0.0147429 loss)
    I0515 01:44:23.061194 23147 sgd_solver.cpp:106] Iteration 58600, lr = 4.71831e-05
    I0515 01:44:31.200845 23147 solver.cpp:228] Iteration 58700, loss = 0.0120852
    I0515 01:44:31.200896 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:44:31.200919 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0120851 (* 1 = 0.0120851 loss)
    I0515 01:44:31.200934 23147 sgd_solver.cpp:106] Iteration 58700, lr = 4.71316e-05
    I0515 01:44:39.331516 23147 solver.cpp:228] Iteration 58800, loss = 0.0115112
    I0515 01:44:39.331717 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:44:39.331738 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0115112 (* 1 = 0.0115112 loss)
    I0515 01:44:39.331755 23147 sgd_solver.cpp:106] Iteration 58800, lr = 4.70802e-05
    I0515 01:44:47.386766 23147 solver.cpp:228] Iteration 58900, loss = 0.0118362
    I0515 01:44:47.386816 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:44:47.386842 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0118361 (* 1 = 0.0118361 loss)
    I0515 01:44:47.386855 23147 sgd_solver.cpp:106] Iteration 58900, lr = 4.7029e-05
    I0515 01:44:55.359952 23147 solver.cpp:337] Iteration 59000, Testing net (#0)
    I0515 01:44:59.772367 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.83625
    I0515 01:44:59.772420 23147 solver.cpp:404]     Test net output #1: loss_c = 0.538167 (* 1 = 0.538167 loss)
    I0515 01:44:59.827569 23147 solver.cpp:228] Iteration 59000, loss = 0.0168333
    I0515 01:44:59.827628 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:44:59.827656 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0168332 (* 1 = 0.0168332 loss)
    I0515 01:44:59.827682 23147 sgd_solver.cpp:106] Iteration 59000, lr = 4.69779e-05
    I0515 01:45:07.965637 23147 solver.cpp:228] Iteration 59100, loss = 0.0111728
    I0515 01:45:07.965677 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:45:07.965698 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0111727 (* 1 = 0.0111727 loss)
    I0515 01:45:07.965713 23147 sgd_solver.cpp:106] Iteration 59100, lr = 4.69269e-05
    I0515 01:45:16.103906 23147 solver.cpp:228] Iteration 59200, loss = 0.0153735
    I0515 01:45:16.104012 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:45:16.104032 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0153734 (* 1 = 0.0153734 loss)
    I0515 01:45:16.104048 23147 sgd_solver.cpp:106] Iteration 59200, lr = 4.6876e-05
    I0515 01:45:24.241796 23147 solver.cpp:228] Iteration 59300, loss = 0.0146497
    I0515 01:45:24.241847 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:45:24.241866 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0146496 (* 1 = 0.0146496 loss)
    I0515 01:45:24.241881 23147 sgd_solver.cpp:106] Iteration 59300, lr = 4.68252e-05
    I0515 01:45:32.381557 23147 solver.cpp:228] Iteration 59400, loss = 0.0102625
    I0515 01:45:32.381597 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:45:32.381618 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0102624 (* 1 = 0.0102624 loss)
    I0515 01:45:32.381633 23147 sgd_solver.cpp:106] Iteration 59400, lr = 4.67746e-05
    I0515 01:45:40.519377 23147 solver.cpp:228] Iteration 59500, loss = 0.020288
    I0515 01:45:40.519426 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:45:40.519446 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0202879 (* 1 = 0.0202879 loss)
    I0515 01:45:40.519461 23147 sgd_solver.cpp:106] Iteration 59500, lr = 4.67241e-05
    I0515 01:45:48.654314 23147 solver.cpp:228] Iteration 59600, loss = 0.00890775
    I0515 01:45:48.654544 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:45:48.654592 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00890765 (* 1 = 0.00890765 loss)
    I0515 01:45:48.654618 23147 sgd_solver.cpp:106] Iteration 59600, lr = 4.66738e-05
    I0515 01:45:56.750406 23147 solver.cpp:228] Iteration 59700, loss = 0.0135658
    I0515 01:45:56.750465 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:45:56.750494 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0135657 (* 1 = 0.0135657 loss)
    I0515 01:45:56.750516 23147 sgd_solver.cpp:106] Iteration 59700, lr = 4.66236e-05
    I0515 01:46:04.845075 23147 solver.cpp:228] Iteration 59800, loss = 0.013357
    I0515 01:46:04.845134 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:46:04.845163 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0133569 (* 1 = 0.0133569 loss)
    I0515 01:46:04.845185 23147 sgd_solver.cpp:106] Iteration 59800, lr = 4.65735e-05
    I0515 01:46:12.930552 23147 solver.cpp:228] Iteration 59900, loss = 0.010241
    I0515 01:46:12.930611 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:46:12.930641 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0102409 (* 1 = 0.0102409 loss)
    I0515 01:46:12.930663 23147 sgd_solver.cpp:106] Iteration 59900, lr = 4.65235e-05
    I0515 01:46:20.938560 23147 solver.cpp:337] Iteration 60000, Testing net (#0)
    I0515 01:46:25.352949 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.824333
    I0515 01:46:25.353005 23147 solver.cpp:404]     Test net output #1: loss_c = 0.549146 (* 1 = 0.549146 loss)
    I0515 01:46:25.404428 23147 solver.cpp:228] Iteration 60000, loss = 0.0100574
    I0515 01:46:25.404486 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:46:25.404506 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0100573 (* 1 = 0.0100573 loss)
    I0515 01:46:25.404525 23147 sgd_solver.cpp:106] Iteration 60000, lr = 4.64736e-05
    I0515 01:46:33.529331 23147 solver.cpp:228] Iteration 60100, loss = 0.0103281
    I0515 01:46:33.529378 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:46:33.529399 23147 solver.cpp:244]     Train net output #1: loss_c = 0.010328 (* 1 = 0.010328 loss)
    I0515 01:46:33.529413 23147 sgd_solver.cpp:106] Iteration 60100, lr = 4.64239e-05
    I0515 01:46:41.669227 23147 solver.cpp:228] Iteration 60200, loss = 0.00928128
    I0515 01:46:41.669268 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:46:41.669288 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00928116 (* 1 = 0.00928116 loss)
    I0515 01:46:41.669303 23147 sgd_solver.cpp:106] Iteration 60200, lr = 4.63743e-05
    I0515 01:46:49.808377 23147 solver.cpp:228] Iteration 60300, loss = 0.00632548
    I0515 01:46:49.808428 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:46:49.808449 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00632536 (* 1 = 0.00632536 loss)
    I0515 01:46:49.808464 23147 sgd_solver.cpp:106] Iteration 60300, lr = 4.63248e-05
    I0515 01:46:57.949918 23147 solver.cpp:228] Iteration 60400, loss = 0.0309707
    I0515 01:46:57.950080 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:46:57.950125 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0309706 (* 1 = 0.0309706 loss)
    I0515 01:46:57.950150 23147 sgd_solver.cpp:106] Iteration 60400, lr = 4.62754e-05
    I0515 01:47:06.091213 23147 solver.cpp:228] Iteration 60500, loss = 0.0189333
    I0515 01:47:06.091264 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:47:06.091285 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0189332 (* 1 = 0.0189332 loss)
    I0515 01:47:06.091300 23147 sgd_solver.cpp:106] Iteration 60500, lr = 4.62262e-05
    I0515 01:47:14.230679 23147 solver.cpp:228] Iteration 60600, loss = 0.0163778
    I0515 01:47:14.230720 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:47:14.230741 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0163777 (* 1 = 0.0163777 loss)
    I0515 01:47:14.230756 23147 sgd_solver.cpp:106] Iteration 60600, lr = 4.61771e-05
    I0515 01:47:22.364562 23147 solver.cpp:228] Iteration 60700, loss = 0.019343
    I0515 01:47:22.364612 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:47:22.364636 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0193429 (* 1 = 0.0193429 loss)
    I0515 01:47:22.364651 23147 sgd_solver.cpp:106] Iteration 60700, lr = 4.61281e-05
    I0515 01:47:30.505090 23147 solver.cpp:228] Iteration 60800, loss = 0.0716544
    I0515 01:47:30.505342 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:47:30.505388 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0716543 (* 1 = 0.0716543 loss)
    I0515 01:47:30.505414 23147 sgd_solver.cpp:106] Iteration 60800, lr = 4.60792e-05
    I0515 01:47:38.601557 23147 solver.cpp:228] Iteration 60900, loss = 0.0110463
    I0515 01:47:38.601610 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:47:38.601639 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0110462 (* 1 = 0.0110462 loss)
    I0515 01:47:38.601660 23147 sgd_solver.cpp:106] Iteration 60900, lr = 4.60305e-05
    I0515 01:47:46.602090 23147 solver.cpp:337] Iteration 61000, Testing net (#0)
    I0515 01:47:50.965153 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.82525
    I0515 01:47:50.965221 23147 solver.cpp:404]     Test net output #1: loss_c = 0.539174 (* 1 = 0.539174 loss)
    I0515 01:47:51.020455 23147 solver.cpp:228] Iteration 61000, loss = 0.0439885
    I0515 01:47:51.020499 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 01:47:51.020527 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0439883 (* 1 = 0.0439883 loss)
    I0515 01:47:51.020555 23147 sgd_solver.cpp:106] Iteration 61000, lr = 4.59818e-05
    I0515 01:47:59.112350 23147 solver.cpp:228] Iteration 61100, loss = 0.0126117
    I0515 01:47:59.112411 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:47:59.112439 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0126116 (* 1 = 0.0126116 loss)
    I0515 01:47:59.112462 23147 sgd_solver.cpp:106] Iteration 61100, lr = 4.59333e-05
    I0515 01:48:07.201248 23147 solver.cpp:228] Iteration 61200, loss = 0.0346686
    I0515 01:48:07.201417 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:48:07.201462 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0346685 (* 1 = 0.0346685 loss)
    I0515 01:48:07.201486 23147 sgd_solver.cpp:106] Iteration 61200, lr = 4.58849e-05
    I0515 01:48:15.339184 23147 solver.cpp:228] Iteration 61300, loss = 0.0155699
    I0515 01:48:15.339223 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:48:15.339246 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0155698 (* 1 = 0.0155698 loss)
    I0515 01:48:15.339259 23147 sgd_solver.cpp:106] Iteration 61300, lr = 4.58366e-05
    I0515 01:48:23.481875 23147 solver.cpp:228] Iteration 61400, loss = 0.01712
    I0515 01:48:23.481917 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:48:23.481938 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0171199 (* 1 = 0.0171199 loss)
    I0515 01:48:23.481953 23147 sgd_solver.cpp:106] Iteration 61400, lr = 4.57885e-05
    I0515 01:48:31.623081 23147 solver.cpp:228] Iteration 61500, loss = 0.0283645
    I0515 01:48:31.623132 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:48:31.623155 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0283644 (* 1 = 0.0283644 loss)
    I0515 01:48:31.623170 23147 sgd_solver.cpp:106] Iteration 61500, lr = 4.57405e-05
    I0515 01:48:39.759302 23147 solver.cpp:228] Iteration 61600, loss = 0.0114765
    I0515 01:48:39.759413 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:48:39.759435 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0114764 (* 1 = 0.0114764 loss)
    I0515 01:48:39.759450 23147 sgd_solver.cpp:106] Iteration 61600, lr = 4.56925e-05
    I0515 01:48:47.897465 23147 solver.cpp:228] Iteration 61700, loss = 0.0204587
    I0515 01:48:47.897511 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:48:47.897533 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0204586 (* 1 = 0.0204586 loss)
    I0515 01:48:47.897548 23147 sgd_solver.cpp:106] Iteration 61700, lr = 4.56447e-05
    I0515 01:48:56.037544 23147 solver.cpp:228] Iteration 61800, loss = 0.0102465
    I0515 01:48:56.037596 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:48:56.037618 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0102464 (* 1 = 0.0102464 loss)
    I0515 01:48:56.037634 23147 sgd_solver.cpp:106] Iteration 61800, lr = 4.5597e-05
    I0515 01:49:04.176185 23147 solver.cpp:228] Iteration 61900, loss = 0.0264832
    I0515 01:49:04.176239 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:49:04.176259 23147 solver.cpp:244]     Train net output #1: loss_c = 0.026483 (* 1 = 0.026483 loss)
    I0515 01:49:04.176273 23147 sgd_solver.cpp:106] Iteration 61900, lr = 4.55495e-05
    I0515 01:49:12.228734 23147 solver.cpp:337] Iteration 62000, Testing net (#0)
    I0515 01:49:16.569368 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.821583
    I0515 01:49:16.569416 23147 solver.cpp:404]     Test net output #1: loss_c = 0.533267 (* 1 = 0.533267 loss)
    I0515 01:49:16.620734 23147 solver.cpp:228] Iteration 62000, loss = 0.0147866
    I0515 01:49:16.620795 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:49:16.620820 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0147865 (* 1 = 0.0147865 loss)
    I0515 01:49:16.620836 23147 sgd_solver.cpp:106] Iteration 62000, lr = 4.5502e-05
    I0515 01:49:24.763562 23147 solver.cpp:228] Iteration 62100, loss = 0.00545798
    I0515 01:49:24.763612 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:49:24.763636 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00545784 (* 1 = 0.00545784 loss)
    I0515 01:49:24.763650 23147 sgd_solver.cpp:106] Iteration 62100, lr = 4.54547e-05
    I0515 01:49:32.905285 23147 solver.cpp:228] Iteration 62200, loss = 0.0849976
    I0515 01:49:32.905336 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:49:32.905356 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0849974 (* 1 = 0.0849974 loss)
    I0515 01:49:32.905370 23147 sgd_solver.cpp:106] Iteration 62200, lr = 4.54074e-05
    I0515 01:49:41.047833 23147 solver.cpp:228] Iteration 62300, loss = 0.0199885
    I0515 01:49:41.047884 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:49:41.047904 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0199884 (* 1 = 0.0199884 loss)
    I0515 01:49:41.047919 23147 sgd_solver.cpp:106] Iteration 62300, lr = 4.53603e-05
    I0515 01:49:49.187099 23147 solver.cpp:228] Iteration 62400, loss = 0.00763549
    I0515 01:49:49.187198 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:49:49.187221 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00763535 (* 1 = 0.00763535 loss)
    I0515 01:49:49.187235 23147 sgd_solver.cpp:106] Iteration 62400, lr = 4.53133e-05
    I0515 01:49:57.329594 23147 solver.cpp:228] Iteration 62500, loss = 0.0126659
    I0515 01:49:57.329653 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:49:57.329674 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0126658 (* 1 = 0.0126658 loss)
    I0515 01:49:57.329687 23147 sgd_solver.cpp:106] Iteration 62500, lr = 4.52665e-05
    I0515 01:50:05.471050 23147 solver.cpp:228] Iteration 62600, loss = 0.0205943
    I0515 01:50:05.471099 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:50:05.471119 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0205942 (* 1 = 0.0205942 loss)
    I0515 01:50:05.471134 23147 sgd_solver.cpp:106] Iteration 62600, lr = 4.52197e-05
    I0515 01:50:13.607914 23147 solver.cpp:228] Iteration 62700, loss = 0.00566539
    I0515 01:50:13.607961 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:50:13.607981 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00566527 (* 1 = 0.00566527 loss)
    I0515 01:50:13.607995 23147 sgd_solver.cpp:106] Iteration 62700, lr = 4.5173e-05
    I0515 01:50:21.736038 23147 solver.cpp:228] Iteration 62800, loss = 0.0326901
    I0515 01:50:21.736273 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:50:21.736318 23147 solver.cpp:244]     Train net output #1: loss_c = 0.03269 (* 1 = 0.03269 loss)
    I0515 01:50:21.736342 23147 sgd_solver.cpp:106] Iteration 62800, lr = 4.51265e-05
    I0515 01:50:29.833255 23147 solver.cpp:228] Iteration 62900, loss = 0.00951846
    I0515 01:50:29.833312 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:50:29.833340 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00951834 (* 1 = 0.00951834 loss)
    I0515 01:50:29.833362 23147 sgd_solver.cpp:106] Iteration 62900, lr = 4.508e-05
    I0515 01:50:37.840590 23147 solver.cpp:337] Iteration 63000, Testing net (#0)
    I0515 01:50:42.191076 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.823
    I0515 01:50:42.191126 23147 solver.cpp:404]     Test net output #1: loss_c = 0.570068 (* 1 = 0.570068 loss)
    I0515 01:50:42.242478 23147 solver.cpp:228] Iteration 63000, loss = 0.018731
    I0515 01:50:42.242537 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:50:42.242558 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0187309 (* 1 = 0.0187309 loss)
    I0515 01:50:42.242575 23147 sgd_solver.cpp:106] Iteration 63000, lr = 4.50337e-05
    I0515 01:50:50.309751 23147 solver.cpp:228] Iteration 63100, loss = 0.00787227
    I0515 01:50:50.309803 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:50:50.309829 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00787217 (* 1 = 0.00787217 loss)
    I0515 01:50:50.309844 23147 sgd_solver.cpp:106] Iteration 63100, lr = 4.49875e-05
    I0515 01:50:58.363349 23147 solver.cpp:228] Iteration 63200, loss = 0.0506587
    I0515 01:50:58.363481 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:50:58.363523 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0506586 (* 1 = 0.0506586 loss)
    I0515 01:50:58.363540 23147 sgd_solver.cpp:106] Iteration 63200, lr = 4.49414e-05
    I0515 01:51:06.447846 23147 solver.cpp:228] Iteration 63300, loss = 0.0278711
    I0515 01:51:06.447892 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:51:06.447914 23147 solver.cpp:244]     Train net output #1: loss_c = 0.027871 (* 1 = 0.027871 loss)
    I0515 01:51:06.447929 23147 sgd_solver.cpp:106] Iteration 63300, lr = 4.48954e-05
    I0515 01:51:14.499593 23147 solver.cpp:228] Iteration 63400, loss = 0.0230053
    I0515 01:51:14.499650 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:51:14.499675 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0230052 (* 1 = 0.0230052 loss)
    I0515 01:51:14.499688 23147 sgd_solver.cpp:106] Iteration 63400, lr = 4.48495e-05
    I0515 01:51:22.555428 23147 solver.cpp:228] Iteration 63500, loss = 0.0104406
    I0515 01:51:22.555483 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:51:22.555507 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0104405 (* 1 = 0.0104405 loss)
    I0515 01:51:22.555526 23147 sgd_solver.cpp:106] Iteration 63500, lr = 4.48038e-05
    I0515 01:51:30.693310 23147 solver.cpp:228] Iteration 63600, loss = 0.0184815
    I0515 01:51:30.693452 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:51:30.693498 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0184814 (* 1 = 0.0184814 loss)
    I0515 01:51:30.693523 23147 sgd_solver.cpp:106] Iteration 63600, lr = 4.47581e-05
    I0515 01:51:38.832825 23147 solver.cpp:228] Iteration 63700, loss = 0.0115616
    I0515 01:51:38.832870 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:51:38.832893 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0115615 (* 1 = 0.0115615 loss)
    I0515 01:51:38.832907 23147 sgd_solver.cpp:106] Iteration 63700, lr = 4.47125e-05
    I0515 01:51:46.969892 23147 solver.cpp:228] Iteration 63800, loss = 0.0273261
    I0515 01:51:46.969941 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:51:46.969964 23147 solver.cpp:244]     Train net output #1: loss_c = 0.027326 (* 1 = 0.027326 loss)
    I0515 01:51:46.969980 23147 sgd_solver.cpp:106] Iteration 63800, lr = 4.46671e-05
    I0515 01:51:55.109738 23147 solver.cpp:228] Iteration 63900, loss = 0.023238
    I0515 01:51:55.109789 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:51:55.109812 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0232379 (* 1 = 0.0232379 loss)
    I0515 01:51:55.109827 23147 sgd_solver.cpp:106] Iteration 63900, lr = 4.46218e-05
    I0515 01:52:03.172024 23147 solver.cpp:337] Iteration 64000, Testing net (#0)
    I0515 01:52:07.586797 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.83275
    I0515 01:52:07.586851 23147 solver.cpp:404]     Test net output #1: loss_c = 0.534957 (* 1 = 0.534957 loss)
    I0515 01:52:07.643030 23147 solver.cpp:228] Iteration 64000, loss = 0.0110814
    I0515 01:52:07.643101 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:52:07.643131 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0110812 (* 1 = 0.0110812 loss)
    I0515 01:52:07.643157 23147 sgd_solver.cpp:106] Iteration 64000, lr = 4.45765e-05
    I0515 01:52:15.777933 23147 solver.cpp:228] Iteration 64100, loss = 0.0107406
    I0515 01:52:15.777984 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:52:15.778004 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0107405 (* 1 = 0.0107405 loss)
    I0515 01:52:15.778018 23147 sgd_solver.cpp:106] Iteration 64100, lr = 4.45314e-05
    I0515 01:52:23.918166 23147 solver.cpp:228] Iteration 64200, loss = 0.0219428
    I0515 01:52:23.918215 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:52:23.918236 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0219427 (* 1 = 0.0219427 loss)
    I0515 01:52:23.918251 23147 sgd_solver.cpp:106] Iteration 64200, lr = 4.44864e-05
    I0515 01:52:32.060135 23147 solver.cpp:228] Iteration 64300, loss = 0.0224706
    I0515 01:52:32.060185 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:52:32.060206 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0224705 (* 1 = 0.0224705 loss)
    I0515 01:52:32.060221 23147 sgd_solver.cpp:106] Iteration 64300, lr = 4.44415e-05
    I0515 01:52:40.200078 23147 solver.cpp:228] Iteration 64400, loss = 0.0213738
    I0515 01:52:40.200188 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:52:40.200209 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0213737 (* 1 = 0.0213737 loss)
    I0515 01:52:40.200224 23147 sgd_solver.cpp:106] Iteration 64400, lr = 4.43967e-05
    I0515 01:52:48.340162 23147 solver.cpp:228] Iteration 64500, loss = 0.00594381
    I0515 01:52:48.340211 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:52:48.340232 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0059437 (* 1 = 0.0059437 loss)
    I0515 01:52:48.340247 23147 sgd_solver.cpp:106] Iteration 64500, lr = 4.4352e-05
    I0515 01:52:56.430624 23147 solver.cpp:228] Iteration 64600, loss = 0.0141043
    I0515 01:52:56.430676 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:52:56.430696 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0141041 (* 1 = 0.0141041 loss)
    I0515 01:52:56.430711 23147 sgd_solver.cpp:106] Iteration 64600, lr = 4.43074e-05
    I0515 01:53:04.570021 23147 solver.cpp:228] Iteration 64700, loss = 0.0136419
    I0515 01:53:04.570072 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:53:04.570092 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0136418 (* 1 = 0.0136418 loss)
    I0515 01:53:04.570107 23147 sgd_solver.cpp:106] Iteration 64700, lr = 4.42629e-05
    I0515 01:53:12.709477 23147 solver.cpp:228] Iteration 64800, loss = 0.00670306
    I0515 01:53:12.709578 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:53:12.709599 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00670294 (* 1 = 0.00670294 loss)
    I0515 01:53:12.709615 23147 sgd_solver.cpp:106] Iteration 64800, lr = 4.42185e-05
    I0515 01:53:20.849057 23147 solver.cpp:228] Iteration 64900, loss = 0.0139791
    I0515 01:53:20.849107 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:53:20.849128 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0139789 (* 1 = 0.0139789 loss)
    I0515 01:53:20.849143 23147 sgd_solver.cpp:106] Iteration 64900, lr = 4.41742e-05
    I0515 01:53:28.908545 23147 solver.cpp:337] Iteration 65000, Testing net (#0)
    I0515 01:53:33.266861 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.83
    I0515 01:53:33.266923 23147 solver.cpp:404]     Test net output #1: loss_c = 0.540496 (* 1 = 0.540496 loss)
    I0515 01:53:33.318455 23147 solver.cpp:228] Iteration 65000, loss = 0.00877609
    I0515 01:53:33.318531 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:53:33.318552 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00877597 (* 1 = 0.00877597 loss)
    I0515 01:53:33.318572 23147 sgd_solver.cpp:106] Iteration 65000, lr = 4.413e-05
    I0515 01:53:41.375895 23147 solver.cpp:228] Iteration 65100, loss = 0.00982112
    I0515 01:53:41.375942 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:53:41.375962 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00982101 (* 1 = 0.00982101 loss)
    I0515 01:53:41.375977 23147 sgd_solver.cpp:106] Iteration 65100, lr = 4.40859e-05
    I0515 01:53:49.429154 23147 solver.cpp:228] Iteration 65200, loss = 0.00934062
    I0515 01:53:49.429360 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:53:49.429381 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0093405 (* 1 = 0.0093405 loss)
    I0515 01:53:49.429397 23147 sgd_solver.cpp:106] Iteration 65200, lr = 4.4042e-05
    I0515 01:53:57.566160 23147 solver.cpp:228] Iteration 65300, loss = 0.0050754
    I0515 01:53:57.566206 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:53:57.566226 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00507528 (* 1 = 0.00507528 loss)
    I0515 01:53:57.566241 23147 sgd_solver.cpp:106] Iteration 65300, lr = 4.39981e-05
    I0515 01:54:05.703282 23147 solver.cpp:228] Iteration 65400, loss = 0.00793475
    I0515 01:54:05.703328 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:54:05.703348 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00793462 (* 1 = 0.00793462 loss)
    I0515 01:54:05.703362 23147 sgd_solver.cpp:106] Iteration 65400, lr = 4.39543e-05
    I0515 01:54:13.841625 23147 solver.cpp:228] Iteration 65500, loss = 0.027906
    I0515 01:54:13.841671 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:54:13.841691 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0279058 (* 1 = 0.0279058 loss)
    I0515 01:54:13.841706 23147 sgd_solver.cpp:106] Iteration 65500, lr = 4.39106e-05
    I0515 01:54:21.960634 23147 solver.cpp:228] Iteration 65600, loss = 0.00774929
    I0515 01:54:21.960731 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:54:21.960760 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00774914 (* 1 = 0.00774914 loss)
    I0515 01:54:21.960782 23147 sgd_solver.cpp:106] Iteration 65600, lr = 4.38671e-05
    I0515 01:54:30.045680 23147 solver.cpp:228] Iteration 65700, loss = 0.0153672
    I0515 01:54:30.045734 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:54:30.045763 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0153671 (* 1 = 0.0153671 loss)
    I0515 01:54:30.045785 23147 sgd_solver.cpp:106] Iteration 65700, lr = 4.38236e-05
    I0515 01:54:38.158072 23147 solver.cpp:228] Iteration 65800, loss = 0.0187812
    I0515 01:54:38.158119 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:54:38.158138 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0187811 (* 1 = 0.0187811 loss)
    I0515 01:54:38.158154 23147 sgd_solver.cpp:106] Iteration 65800, lr = 4.37802e-05
    I0515 01:54:46.295155 23147 solver.cpp:228] Iteration 65900, loss = 0.0133112
    I0515 01:54:46.295200 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:54:46.295220 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0133111 (* 1 = 0.0133111 loss)
    I0515 01:54:46.295234 23147 sgd_solver.cpp:106] Iteration 65900, lr = 4.3737e-05
    I0515 01:54:54.356082 23147 solver.cpp:337] Iteration 66000, Testing net (#0)
    I0515 01:54:58.771790 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.827917
    I0515 01:54:58.771847 23147 solver.cpp:404]     Test net output #1: loss_c = 0.560315 (* 1 = 0.560315 loss)
    I0515 01:54:58.823197 23147 solver.cpp:228] Iteration 66000, loss = 0.0233441
    I0515 01:54:58.823232 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:54:58.823251 23147 solver.cpp:244]     Train net output #1: loss_c = 0.023344 (* 1 = 0.023344 loss)
    I0515 01:54:58.823271 23147 sgd_solver.cpp:106] Iteration 66000, lr = 4.36938e-05
    I0515 01:55:06.960947 23147 solver.cpp:228] Iteration 66100, loss = 0.0114945
    I0515 01:55:06.960994 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:55:06.961015 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0114943 (* 1 = 0.0114943 loss)
    I0515 01:55:06.961033 23147 sgd_solver.cpp:106] Iteration 66100, lr = 4.36507e-05
    I0515 01:55:15.101223 23147 solver.cpp:228] Iteration 66200, loss = 0.0101101
    I0515 01:55:15.101274 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:55:15.101294 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0101099 (* 1 = 0.0101099 loss)
    I0515 01:55:15.101310 23147 sgd_solver.cpp:106] Iteration 66200, lr = 4.36078e-05
    I0515 01:55:23.236631 23147 solver.cpp:228] Iteration 66300, loss = 0.0164998
    I0515 01:55:23.236673 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:55:23.236693 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0164997 (* 1 = 0.0164997 loss)
    I0515 01:55:23.236707 23147 sgd_solver.cpp:106] Iteration 66300, lr = 4.35649e-05
    I0515 01:55:31.379132 23147 solver.cpp:228] Iteration 66400, loss = 0.0275679
    I0515 01:55:31.379217 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:55:31.379238 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0275678 (* 1 = 0.0275678 loss)
    I0515 01:55:31.379252 23147 sgd_solver.cpp:106] Iteration 66400, lr = 4.35221e-05
    I0515 01:55:39.517693 23147 solver.cpp:228] Iteration 66500, loss = 0.00683458
    I0515 01:55:39.517746 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:55:39.517771 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00683444 (* 1 = 0.00683444 loss)
    I0515 01:55:39.517786 23147 sgd_solver.cpp:106] Iteration 66500, lr = 4.34794e-05
    I0515 01:55:47.657899 23147 solver.cpp:228] Iteration 66600, loss = 0.00760193
    I0515 01:55:47.657950 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:55:47.657974 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0076018 (* 1 = 0.0076018 loss)
    I0515 01:55:47.657989 23147 sgd_solver.cpp:106] Iteration 66600, lr = 4.34369e-05
    I0515 01:55:55.795392 23147 solver.cpp:228] Iteration 66700, loss = 0.0177909
    I0515 01:55:55.795442 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:55:55.795462 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0177908 (* 1 = 0.0177908 loss)
    I0515 01:55:55.795477 23147 sgd_solver.cpp:106] Iteration 66700, lr = 4.33944e-05
    I0515 01:56:03.936494 23147 solver.cpp:228] Iteration 66800, loss = 0.0186451
    I0515 01:56:03.936591 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:56:03.936615 23147 solver.cpp:244]     Train net output #1: loss_c = 0.018645 (* 1 = 0.018645 loss)
    I0515 01:56:03.936630 23147 sgd_solver.cpp:106] Iteration 66800, lr = 4.3352e-05
    I0515 01:56:12.078433 23147 solver.cpp:228] Iteration 66900, loss = 0.0168779
    I0515 01:56:12.078475 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:56:12.078495 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0168778 (* 1 = 0.0168778 loss)
    I0515 01:56:12.078508 23147 sgd_solver.cpp:106] Iteration 66900, lr = 4.33097e-05
    I0515 01:56:20.134343 23147 solver.cpp:337] Iteration 67000, Testing net (#0)
    I0515 01:56:24.542093 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.824917
    I0515 01:56:24.542150 23147 solver.cpp:404]     Test net output #1: loss_c = 0.546008 (* 1 = 0.546008 loss)
    I0515 01:56:24.597422 23147 solver.cpp:228] Iteration 67000, loss = 0.015124
    I0515 01:56:24.597491 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:56:24.597520 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0151239 (* 1 = 0.0151239 loss)
    I0515 01:56:24.597546 23147 sgd_solver.cpp:106] Iteration 67000, lr = 4.32675e-05
    I0515 01:56:32.732863 23147 solver.cpp:228] Iteration 67100, loss = 0.0172664
    I0515 01:56:32.732909 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:56:32.732928 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0172663 (* 1 = 0.0172663 loss)
    I0515 01:56:32.732944 23147 sgd_solver.cpp:106] Iteration 67100, lr = 4.32254e-05
    I0515 01:56:40.863085 23147 solver.cpp:228] Iteration 67200, loss = 0.0305698
    I0515 01:56:40.863337 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:56:40.863387 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0305697 (* 1 = 0.0305697 loss)
    I0515 01:56:40.863412 23147 sgd_solver.cpp:106] Iteration 67200, lr = 4.31834e-05
    I0515 01:56:49.003834 23147 solver.cpp:228] Iteration 67300, loss = 0.0255625
    I0515 01:56:49.003878 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:56:49.003900 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0255624 (* 1 = 0.0255624 loss)
    I0515 01:56:49.003914 23147 sgd_solver.cpp:106] Iteration 67300, lr = 4.31415e-05
    I0515 01:56:57.142343 23147 solver.cpp:228] Iteration 67400, loss = 0.00741399
    I0515 01:56:57.142388 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:56:57.142412 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00741388 (* 1 = 0.00741388 loss)
    I0515 01:56:57.142426 23147 sgd_solver.cpp:106] Iteration 67400, lr = 4.30997e-05
    I0515 01:57:05.269106 23147 solver.cpp:228] Iteration 67500, loss = 0.0505238
    I0515 01:57:05.269150 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:57:05.269170 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0505237 (* 1 = 0.0505237 loss)
    I0515 01:57:05.269183 23147 sgd_solver.cpp:106] Iteration 67500, lr = 4.3058e-05
    I0515 01:57:13.406960 23147 solver.cpp:228] Iteration 67600, loss = 0.0282888
    I0515 01:57:13.407224 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:57:13.407271 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0282887 (* 1 = 0.0282887 loss)
    I0515 01:57:13.407295 23147 sgd_solver.cpp:106] Iteration 67600, lr = 4.30164e-05
    I0515 01:57:21.537679 23147 solver.cpp:228] Iteration 67700, loss = 0.0300139
    I0515 01:57:21.537729 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:57:21.537750 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0300138 (* 1 = 0.0300138 loss)
    I0515 01:57:21.537765 23147 sgd_solver.cpp:106] Iteration 67700, lr = 4.29748e-05
    I0515 01:57:29.678571 23147 solver.cpp:228] Iteration 67800, loss = 0.0485609
    I0515 01:57:29.678622 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:57:29.678642 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0485608 (* 1 = 0.0485608 loss)
    I0515 01:57:29.678656 23147 sgd_solver.cpp:106] Iteration 67800, lr = 4.29334e-05
    I0515 01:57:37.816424 23147 solver.cpp:228] Iteration 67900, loss = 0.0216526
    I0515 01:57:37.816481 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:57:37.816506 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0216525 (* 1 = 0.0216525 loss)
    I0515 01:57:37.816521 23147 sgd_solver.cpp:106] Iteration 67900, lr = 4.28921e-05
    I0515 01:57:45.874402 23147 solver.cpp:337] Iteration 68000, Testing net (#0)
    I0515 01:57:50.292479 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.836833
    I0515 01:57:50.292532 23147 solver.cpp:404]     Test net output #1: loss_c = 0.51611 (* 1 = 0.51611 loss)
    I0515 01:57:50.344774 23147 solver.cpp:228] Iteration 68000, loss = 0.0215972
    I0515 01:57:50.344796 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:57:50.344815 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0215971 (* 1 = 0.0215971 loss)
    I0515 01:57:50.344832 23147 sgd_solver.cpp:106] Iteration 68000, lr = 4.28508e-05
    I0515 01:57:58.484616 23147 solver.cpp:228] Iteration 68100, loss = 0.0182458
    I0515 01:57:58.484663 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:57:58.484683 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0182457 (* 1 = 0.0182457 loss)
    I0515 01:57:58.484697 23147 sgd_solver.cpp:106] Iteration 68100, lr = 4.28097e-05
    I0515 01:58:06.625107 23147 solver.cpp:228] Iteration 68200, loss = 0.0150884
    I0515 01:58:06.625151 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:58:06.625171 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0150883 (* 1 = 0.0150883 loss)
    I0515 01:58:06.625186 23147 sgd_solver.cpp:106] Iteration 68200, lr = 4.27686e-05
    I0515 01:58:14.765496 23147 solver.cpp:228] Iteration 68300, loss = 0.024519
    I0515 01:58:14.765542 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:58:14.765561 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0245189 (* 1 = 0.0245189 loss)
    I0515 01:58:14.765575 23147 sgd_solver.cpp:106] Iteration 68300, lr = 4.27276e-05
    I0515 01:58:22.897234 23147 solver.cpp:228] Iteration 68400, loss = 0.0119131
    I0515 01:58:22.897442 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:58:22.897500 23147 solver.cpp:244]     Train net output #1: loss_c = 0.011913 (* 1 = 0.011913 loss)
    I0515 01:58:22.897514 23147 sgd_solver.cpp:106] Iteration 68400, lr = 4.26867e-05
    I0515 01:58:31.038673 23147 solver.cpp:228] Iteration 68500, loss = 0.0134482
    I0515 01:58:31.038719 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:58:31.038739 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0134481 (* 1 = 0.0134481 loss)
    I0515 01:58:31.038754 23147 sgd_solver.cpp:106] Iteration 68500, lr = 4.26459e-05
    I0515 01:58:39.170536 23147 solver.cpp:228] Iteration 68600, loss = 0.00697662
    I0515 01:58:39.170581 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:58:39.170601 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00697653 (* 1 = 0.00697653 loss)
    I0515 01:58:39.170616 23147 sgd_solver.cpp:106] Iteration 68600, lr = 4.26052e-05
    I0515 01:58:47.309581 23147 solver.cpp:228] Iteration 68700, loss = 0.0106674
    I0515 01:58:47.309626 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:58:47.309646 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0106673 (* 1 = 0.0106673 loss)
    I0515 01:58:47.309660 23147 sgd_solver.cpp:106] Iteration 68700, lr = 4.25646e-05
    I0515 01:58:55.451318 23147 solver.cpp:228] Iteration 68800, loss = 0.0274553
    I0515 01:58:55.451479 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 01:58:55.451532 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0274552 (* 1 = 0.0274552 loss)
    I0515 01:58:55.451557 23147 sgd_solver.cpp:106] Iteration 68800, lr = 4.25241e-05
    I0515 01:59:03.590157 23147 solver.cpp:228] Iteration 68900, loss = 0.0132633
    I0515 01:59:03.590209 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:59:03.590229 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0132632 (* 1 = 0.0132632 loss)
    I0515 01:59:03.590245 23147 sgd_solver.cpp:106] Iteration 68900, lr = 4.24837e-05
    I0515 01:59:11.648448 23147 solver.cpp:337] Iteration 69000, Testing net (#0)
    I0515 01:59:16.046672 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.825583
    I0515 01:59:16.046731 23147 solver.cpp:404]     Test net output #1: loss_c = 0.550119 (* 1 = 0.550119 loss)
    I0515 01:59:16.102869 23147 solver.cpp:228] Iteration 69000, loss = 0.0147098
    I0515 01:59:16.102918 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:59:16.102946 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0147097 (* 1 = 0.0147097 loss)
    I0515 01:59:16.102973 23147 sgd_solver.cpp:106] Iteration 69000, lr = 4.24434e-05
    I0515 01:59:24.219750 23147 solver.cpp:228] Iteration 69100, loss = 0.0389495
    I0515 01:59:24.219805 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:59:24.219833 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0389494 (* 1 = 0.0389494 loss)
    I0515 01:59:24.219856 23147 sgd_solver.cpp:106] Iteration 69100, lr = 4.24031e-05
    I0515 01:59:32.309532 23147 solver.cpp:228] Iteration 69200, loss = 0.0311673
    I0515 01:59:32.309767 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:59:32.309798 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0311673 (* 1 = 0.0311673 loss)
    I0515 01:59:32.309820 23147 sgd_solver.cpp:106] Iteration 69200, lr = 4.23629e-05
    I0515 01:59:40.400892 23147 solver.cpp:228] Iteration 69300, loss = 0.0061707
    I0515 01:59:40.400948 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:59:40.400976 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00617063 (* 1 = 0.00617063 loss)
    I0515 01:59:40.400997 23147 sgd_solver.cpp:106] Iteration 69300, lr = 4.23229e-05
    I0515 01:59:48.490875 23147 solver.cpp:228] Iteration 69400, loss = 0.0246236
    I0515 01:59:48.490936 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 01:59:48.490965 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0246236 (* 1 = 0.0246236 loss)
    I0515 01:59:48.490988 23147 sgd_solver.cpp:106] Iteration 69400, lr = 4.22829e-05
    I0515 01:59:56.579037 23147 solver.cpp:228] Iteration 69500, loss = 0.00955943
    I0515 01:59:56.579099 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 01:59:56.579128 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00955934 (* 1 = 0.00955934 loss)
    I0515 01:59:56.579150 23147 sgd_solver.cpp:106] Iteration 69500, lr = 4.2243e-05
    I0515 02:00:04.672832 23147 solver.cpp:228] Iteration 69600, loss = 0.00848984
    I0515 02:00:04.672942 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:00:04.672972 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00848975 (* 1 = 0.00848975 loss)
    I0515 02:00:04.672994 23147 sgd_solver.cpp:106] Iteration 69600, lr = 4.22032e-05
    I0515 02:00:12.758534 23147 solver.cpp:228] Iteration 69700, loss = 0.00805824
    I0515 02:00:12.758595 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:00:12.758625 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00805815 (* 1 = 0.00805815 loss)
    I0515 02:00:12.758646 23147 sgd_solver.cpp:106] Iteration 69700, lr = 4.21635e-05
    I0515 02:00:20.851696 23147 solver.cpp:228] Iteration 69800, loss = 0.0100537
    I0515 02:00:20.851752 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:00:20.851780 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0100536 (* 1 = 0.0100536 loss)
    I0515 02:00:20.851801 23147 sgd_solver.cpp:106] Iteration 69800, lr = 4.21238e-05
    I0515 02:00:28.944427 23147 solver.cpp:228] Iteration 69900, loss = 0.0196077
    I0515 02:00:28.944485 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:00:28.944515 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0196076 (* 1 = 0.0196076 loss)
    I0515 02:00:28.944536 23147 sgd_solver.cpp:106] Iteration 69900, lr = 4.20843e-05
    I0515 02:00:36.952450 23147 solver.cpp:337] Iteration 70000, Testing net (#0)
    I0515 02:00:41.374332 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.824166
    I0515 02:00:41.374382 23147 solver.cpp:404]     Test net output #1: loss_c = 0.548274 (* 1 = 0.548274 loss)
    I0515 02:00:41.429427 23147 solver.cpp:228] Iteration 70000, loss = 0.0458558
    I0515 02:00:41.429491 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:00:41.429519 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0458557 (* 1 = 0.0458557 loss)
    I0515 02:00:41.429545 23147 sgd_solver.cpp:106] Iteration 70000, lr = 4.20448e-05
    I0515 02:00:49.557018 23147 solver.cpp:228] Iteration 70100, loss = 0.0232183
    I0515 02:00:49.557060 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:00:49.557080 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0232182 (* 1 = 0.0232182 loss)
    I0515 02:00:49.557096 23147 sgd_solver.cpp:106] Iteration 70100, lr = 4.20054e-05
    I0515 02:00:57.669348 23147 solver.cpp:228] Iteration 70200, loss = 0.0487806
    I0515 02:00:57.669391 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:00:57.669411 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0487805 (* 1 = 0.0487805 loss)
    I0515 02:00:57.669426 23147 sgd_solver.cpp:106] Iteration 70200, lr = 4.19662e-05
    I0515 02:01:05.793902 23147 solver.cpp:228] Iteration 70300, loss = 0.0199654
    I0515 02:01:05.793949 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:01:05.793969 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0199653 (* 1 = 0.0199653 loss)
    I0515 02:01:05.793984 23147 sgd_solver.cpp:106] Iteration 70300, lr = 4.1927e-05
    I0515 02:01:13.933589 23147 solver.cpp:228] Iteration 70400, loss = 0.024044
    I0515 02:01:13.933771 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:01:13.933792 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0240439 (* 1 = 0.0240439 loss)
    I0515 02:01:13.933823 23147 sgd_solver.cpp:106] Iteration 70400, lr = 4.18878e-05
    I0515 02:01:22.072644 23147 solver.cpp:228] Iteration 70500, loss = 0.00685303
    I0515 02:01:22.072692 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:01:22.072716 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00685292 (* 1 = 0.00685292 loss)
    I0515 02:01:22.072731 23147 sgd_solver.cpp:106] Iteration 70500, lr = 4.18488e-05
    I0515 02:01:30.209401 23147 solver.cpp:228] Iteration 70600, loss = 0.0120056
    I0515 02:01:30.209457 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:01:30.209481 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0120055 (* 1 = 0.0120055 loss)
    I0515 02:01:30.209494 23147 sgd_solver.cpp:106] Iteration 70600, lr = 4.18099e-05
    I0515 02:01:38.349222 23147 solver.cpp:228] Iteration 70700, loss = 0.0115004
    I0515 02:01:38.349269 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:01:38.349288 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0115003 (* 1 = 0.0115003 loss)
    I0515 02:01:38.349303 23147 sgd_solver.cpp:106] Iteration 70700, lr = 4.1771e-05
    I0515 02:01:46.477874 23147 solver.cpp:228] Iteration 70800, loss = 0.030252
    I0515 02:01:46.478142 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:01:46.478188 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0302518 (* 1 = 0.0302518 loss)
    I0515 02:01:46.478212 23147 sgd_solver.cpp:106] Iteration 70800, lr = 4.17322e-05
    I0515 02:01:54.616447 23147 solver.cpp:228] Iteration 70900, loss = 0.0571071
    I0515 02:01:54.616494 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:01:54.616514 23147 solver.cpp:244]     Train net output #1: loss_c = 0.057107 (* 1 = 0.057107 loss)
    I0515 02:01:54.616529 23147 sgd_solver.cpp:106] Iteration 70900, lr = 4.16935e-05
    I0515 02:02:02.670481 23147 solver.cpp:337] Iteration 71000, Testing net (#0)
    I0515 02:02:07.074503 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.829167
    I0515 02:02:07.074560 23147 solver.cpp:404]     Test net output #1: loss_c = 0.544715 (* 1 = 0.544715 loss)
    I0515 02:02:07.130897 23147 solver.cpp:228] Iteration 71000, loss = 0.0113748
    I0515 02:02:07.130975 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:02:07.131006 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0113747 (* 1 = 0.0113747 loss)
    I0515 02:02:07.131031 23147 sgd_solver.cpp:106] Iteration 71000, lr = 4.16549e-05
    I0515 02:02:15.220654 23147 solver.cpp:228] Iteration 71100, loss = 0.0185073
    I0515 02:02:15.220713 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:02:15.220742 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0185071 (* 1 = 0.0185071 loss)
    I0515 02:02:15.220763 23147 sgd_solver.cpp:106] Iteration 71100, lr = 4.16164e-05
    I0515 02:02:23.307476 23147 solver.cpp:228] Iteration 71200, loss = 0.0422126
    I0515 02:02:23.307680 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:02:23.307726 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0422124 (* 1 = 0.0422124 loss)
    I0515 02:02:23.307751 23147 sgd_solver.cpp:106] Iteration 71200, lr = 4.15779e-05
    I0515 02:02:31.405150 23147 solver.cpp:228] Iteration 71300, loss = 0.0185921
    I0515 02:02:31.405196 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:02:31.405218 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0185919 (* 1 = 0.0185919 loss)
    I0515 02:02:31.405233 23147 sgd_solver.cpp:106] Iteration 71300, lr = 4.15396e-05
    I0515 02:02:39.546937 23147 solver.cpp:228] Iteration 71400, loss = 0.0065715
    I0515 02:02:39.546988 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:02:39.547009 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00657138 (* 1 = 0.00657138 loss)
    I0515 02:02:39.547024 23147 sgd_solver.cpp:106] Iteration 71400, lr = 4.15013e-05
    I0515 02:02:47.689110 23147 solver.cpp:228] Iteration 71500, loss = 0.0229266
    I0515 02:02:47.689160 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:02:47.689180 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0229264 (* 1 = 0.0229264 loss)
    I0515 02:02:47.689195 23147 sgd_solver.cpp:106] Iteration 71500, lr = 4.14631e-05
    I0515 02:02:55.830307 23147 solver.cpp:228] Iteration 71600, loss = 0.0191222
    I0515 02:02:55.830462 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:02:55.830507 23147 solver.cpp:244]     Train net output #1: loss_c = 0.019122 (* 1 = 0.019122 loss)
    I0515 02:02:55.830533 23147 sgd_solver.cpp:106] Iteration 71600, lr = 4.1425e-05
    I0515 02:03:03.967209 23147 solver.cpp:228] Iteration 71700, loss = 0.0142332
    I0515 02:03:03.967253 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:03:03.967273 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0142331 (* 1 = 0.0142331 loss)
    I0515 02:03:03.967288 23147 sgd_solver.cpp:106] Iteration 71700, lr = 4.1387e-05
    I0515 02:03:12.106866 23147 solver.cpp:228] Iteration 71800, loss = 0.021749
    I0515 02:03:12.106920 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:03:12.106943 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0217488 (* 1 = 0.0217488 loss)
    I0515 02:03:12.106958 23147 sgd_solver.cpp:106] Iteration 71800, lr = 4.1349e-05
    I0515 02:03:20.241016 23147 solver.cpp:228] Iteration 71900, loss = 0.0309395
    I0515 02:03:20.241061 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:03:20.241081 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0309394 (* 1 = 0.0309394 loss)
    I0515 02:03:20.241096 23147 sgd_solver.cpp:106] Iteration 71900, lr = 4.13111e-05
    I0515 02:03:28.301920 23147 solver.cpp:337] Iteration 72000, Testing net (#0)
    I0515 02:03:32.706164 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.811167
    I0515 02:03:32.706215 23147 solver.cpp:404]     Test net output #1: loss_c = 0.557673 (* 1 = 0.557673 loss)
    I0515 02:03:32.757524 23147 solver.cpp:228] Iteration 72000, loss = 0.00942949
    I0515 02:03:32.757560 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:03:32.757580 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00942937 (* 1 = 0.00942937 loss)
    I0515 02:03:32.757597 23147 sgd_solver.cpp:106] Iteration 72000, lr = 4.12733e-05
    I0515 02:03:40.889572 23147 solver.cpp:228] Iteration 72100, loss = 0.0139304
    I0515 02:03:40.889616 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:03:40.889637 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0139303 (* 1 = 0.0139303 loss)
    I0515 02:03:40.889652 23147 sgd_solver.cpp:106] Iteration 72100, lr = 4.12356e-05
    I0515 02:03:49.026903 23147 solver.cpp:228] Iteration 72200, loss = 0.0210748
    I0515 02:03:49.026948 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:03:49.026971 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0210747 (* 1 = 0.0210747 loss)
    I0515 02:03:49.026986 23147 sgd_solver.cpp:106] Iteration 72200, lr = 4.1198e-05
    I0515 02:03:57.159414 23147 solver.cpp:228] Iteration 72300, loss = 0.0289078
    I0515 02:03:57.159452 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:03:57.159473 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0289076 (* 1 = 0.0289076 loss)
    I0515 02:03:57.159488 23147 sgd_solver.cpp:106] Iteration 72300, lr = 4.11605e-05
    I0515 02:04:05.298735 23147 solver.cpp:228] Iteration 72400, loss = 0.0201711
    I0515 02:04:05.299007 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:04:05.299054 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0201709 (* 1 = 0.0201709 loss)
    I0515 02:04:05.299079 23147 sgd_solver.cpp:106] Iteration 72400, lr = 4.1123e-05
    I0515 02:04:13.389505 23147 solver.cpp:228] Iteration 72500, loss = 0.0109179
    I0515 02:04:13.389560 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:04:13.389590 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0109177 (* 1 = 0.0109177 loss)
    I0515 02:04:13.389611 23147 sgd_solver.cpp:106] Iteration 72500, lr = 4.10856e-05
    I0515 02:04:21.479133 23147 solver.cpp:228] Iteration 72600, loss = 0.0143842
    I0515 02:04:21.479190 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:04:21.479219 23147 solver.cpp:244]     Train net output #1: loss_c = 0.014384 (* 1 = 0.014384 loss)
    I0515 02:04:21.479243 23147 sgd_solver.cpp:106] Iteration 72600, lr = 4.10483e-05
    I0515 02:04:29.570382 23147 solver.cpp:228] Iteration 72700, loss = 0.0098664
    I0515 02:04:29.570442 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:04:29.570473 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00986626 (* 1 = 0.00986626 loss)
    I0515 02:04:29.570494 23147 sgd_solver.cpp:106] Iteration 72700, lr = 4.1011e-05
    I0515 02:04:37.660365 23147 solver.cpp:228] Iteration 72800, loss = 0.0129202
    I0515 02:04:37.660512 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:04:37.660557 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0129201 (* 1 = 0.0129201 loss)
    I0515 02:04:37.660581 23147 sgd_solver.cpp:106] Iteration 72800, lr = 4.09739e-05
    I0515 02:04:45.801097 23147 solver.cpp:228] Iteration 72900, loss = 0.0151385
    I0515 02:04:45.801142 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:04:45.801165 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0151384 (* 1 = 0.0151384 loss)
    I0515 02:04:45.801180 23147 sgd_solver.cpp:106] Iteration 72900, lr = 4.09368e-05
    I0515 02:04:53.859689 23147 solver.cpp:337] Iteration 73000, Testing net (#0)
    I0515 02:04:58.282063 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.83625
    I0515 02:04:58.282120 23147 solver.cpp:404]     Test net output #1: loss_c = 0.509545 (* 1 = 0.509545 loss)
    I0515 02:04:58.334357 23147 solver.cpp:228] Iteration 73000, loss = 0.0106763
    I0515 02:04:58.334408 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:04:58.334427 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0106761 (* 1 = 0.0106761 loss)
    I0515 02:04:58.334450 23147 sgd_solver.cpp:106] Iteration 73000, lr = 4.08998e-05
    I0515 02:05:06.478106 23147 solver.cpp:228] Iteration 73100, loss = 0.0207284
    I0515 02:05:06.478157 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:05:06.478178 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0207282 (* 1 = 0.0207282 loss)
    I0515 02:05:06.478193 23147 sgd_solver.cpp:106] Iteration 73100, lr = 4.08629e-05
    I0515 02:05:14.626442 23147 solver.cpp:228] Iteration 73200, loss = 0.0171441
    I0515 02:05:14.626600 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:05:14.626646 23147 solver.cpp:244]     Train net output #1: loss_c = 0.017144 (* 1 = 0.017144 loss)
    I0515 02:05:14.626669 23147 sgd_solver.cpp:106] Iteration 73200, lr = 4.08261e-05
    I0515 02:05:22.746587 23147 solver.cpp:228] Iteration 73300, loss = 0.0110791
    I0515 02:05:22.746644 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:05:22.746673 23147 solver.cpp:244]     Train net output #1: loss_c = 0.011079 (* 1 = 0.011079 loss)
    I0515 02:05:22.746695 23147 sgd_solver.cpp:106] Iteration 73300, lr = 4.07893e-05
    I0515 02:05:30.873504 23147 solver.cpp:228] Iteration 73400, loss = 0.0338464
    I0515 02:05:30.873560 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:05:30.873589 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0338463 (* 1 = 0.0338463 loss)
    I0515 02:05:30.873610 23147 sgd_solver.cpp:106] Iteration 73400, lr = 4.07526e-05
    I0515 02:05:38.999217 23147 solver.cpp:228] Iteration 73500, loss = 0.0269577
    I0515 02:05:38.999258 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:05:38.999277 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0269576 (* 1 = 0.0269576 loss)
    I0515 02:05:38.999291 23147 sgd_solver.cpp:106] Iteration 73500, lr = 4.0716e-05
    I0515 02:05:47.137822 23147 solver.cpp:228] Iteration 73600, loss = 0.0198119
    I0515 02:05:47.138065 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:05:47.138113 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0198117 (* 1 = 0.0198117 loss)
    I0515 02:05:47.138137 23147 sgd_solver.cpp:106] Iteration 73600, lr = 4.06795e-05
    I0515 02:05:55.263898 23147 solver.cpp:228] Iteration 73700, loss = 0.0402671
    I0515 02:05:55.263939 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:05:55.263958 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0402669 (* 1 = 0.0402669 loss)
    I0515 02:05:55.263973 23147 sgd_solver.cpp:106] Iteration 73700, lr = 4.0643e-05
    I0515 02:06:03.403709 23147 solver.cpp:228] Iteration 73800, loss = 0.0348235
    I0515 02:06:03.403748 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:06:03.403767 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0348234 (* 1 = 0.0348234 loss)
    I0515 02:06:03.403782 23147 sgd_solver.cpp:106] Iteration 73800, lr = 4.06066e-05
    I0515 02:06:11.535933 23147 solver.cpp:228] Iteration 73900, loss = 0.0533246
    I0515 02:06:11.535974 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:06:11.535994 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0533244 (* 1 = 0.0533244 loss)
    I0515 02:06:11.536008 23147 sgd_solver.cpp:106] Iteration 73900, lr = 4.05703e-05
    I0515 02:06:19.597489 23147 solver.cpp:337] Iteration 74000, Testing net (#0)
    I0515 02:06:23.973820 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.828333
    I0515 02:06:23.973878 23147 solver.cpp:404]     Test net output #1: loss_c = 0.530487 (* 1 = 0.530487 loss)
    I0515 02:06:24.025311 23147 solver.cpp:228] Iteration 74000, loss = 0.0148794
    I0515 02:06:24.025363 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:06:24.025384 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0148793 (* 1 = 0.0148793 loss)
    I0515 02:06:24.025405 23147 sgd_solver.cpp:106] Iteration 74000, lr = 4.05341e-05
    I0515 02:06:32.126652 23147 solver.cpp:228] Iteration 74100, loss = 0.0339602
    I0515 02:06:32.126709 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:06:32.126739 23147 solver.cpp:244]     Train net output #1: loss_c = 0.03396 (* 1 = 0.03396 loss)
    I0515 02:06:32.126760 23147 sgd_solver.cpp:106] Iteration 74100, lr = 4.04979e-05
    I0515 02:06:40.218210 23147 solver.cpp:228] Iteration 74200, loss = 0.0302752
    I0515 02:06:40.218269 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:06:40.218299 23147 solver.cpp:244]     Train net output #1: loss_c = 0.030275 (* 1 = 0.030275 loss)
    I0515 02:06:40.218322 23147 sgd_solver.cpp:106] Iteration 74200, lr = 4.04619e-05
    I0515 02:06:48.305032 23147 solver.cpp:228] Iteration 74300, loss = 0.00967283
    I0515 02:06:48.305095 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:06:48.305124 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00967268 (* 1 = 0.00967268 loss)
    I0515 02:06:48.305146 23147 sgd_solver.cpp:106] Iteration 74300, lr = 4.04259e-05
    I0515 02:06:56.398922 23147 solver.cpp:228] Iteration 74400, loss = 0.026301
    I0515 02:06:56.399075 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:06:56.399106 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0263008 (* 1 = 0.0263008 loss)
    I0515 02:06:56.399149 23147 sgd_solver.cpp:106] Iteration 74400, lr = 4.03899e-05
    I0515 02:07:04.488811 23147 solver.cpp:228] Iteration 74500, loss = 0.00812006
    I0515 02:07:04.488867 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:07:04.488895 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0081199 (* 1 = 0.0081199 loss)
    I0515 02:07:04.488917 23147 sgd_solver.cpp:106] Iteration 74500, lr = 4.03541e-05
    I0515 02:07:12.578883 23147 solver.cpp:228] Iteration 74600, loss = 0.0136067
    I0515 02:07:12.578941 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:07:12.578969 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0136065 (* 1 = 0.0136065 loss)
    I0515 02:07:12.578991 23147 sgd_solver.cpp:106] Iteration 74600, lr = 4.03183e-05
    I0515 02:07:20.667433 23147 solver.cpp:228] Iteration 74700, loss = 0.0425207
    I0515 02:07:20.667484 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:07:20.667517 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0425206 (* 1 = 0.0425206 loss)
    I0515 02:07:20.667539 23147 sgd_solver.cpp:106] Iteration 74700, lr = 4.02826e-05
    I0515 02:07:28.751361 23147 solver.cpp:228] Iteration 74800, loss = 0.0155273
    I0515 02:07:28.751516 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:07:28.751564 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0155272 (* 1 = 0.0155272 loss)
    I0515 02:07:28.751590 23147 sgd_solver.cpp:106] Iteration 74800, lr = 4.0247e-05
    I0515 02:07:36.889652 23147 solver.cpp:228] Iteration 74900, loss = 0.0375533
    I0515 02:07:36.889693 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:07:36.889714 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0375532 (* 1 = 0.0375532 loss)
    I0515 02:07:36.889729 23147 sgd_solver.cpp:106] Iteration 74900, lr = 4.02114e-05
    I0515 02:07:44.944564 23147 solver.cpp:337] Iteration 75000, Testing net (#0)
    I0515 02:07:49.362258 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.829667
    I0515 02:07:49.362314 23147 solver.cpp:404]     Test net output #1: loss_c = 0.545298 (* 1 = 0.545298 loss)
    I0515 02:07:49.414629 23147 solver.cpp:228] Iteration 75000, loss = 0.00948661
    I0515 02:07:49.414705 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:07:49.414731 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00948647 (* 1 = 0.00948647 loss)
    I0515 02:07:49.414747 23147 sgd_solver.cpp:106] Iteration 75000, lr = 4.01759e-05
    I0515 02:07:57.512697 23147 solver.cpp:228] Iteration 75100, loss = 0.00820701
    I0515 02:07:57.512758 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:07:57.512787 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00820687 (* 1 = 0.00820687 loss)
    I0515 02:07:57.512809 23147 sgd_solver.cpp:106] Iteration 75100, lr = 4.01405e-05
    I0515 02:08:05.600257 23147 solver.cpp:228] Iteration 75200, loss = 0.0189179
    I0515 02:08:05.600371 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:08:05.600402 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0189178 (* 1 = 0.0189178 loss)
    I0515 02:08:05.600425 23147 sgd_solver.cpp:106] Iteration 75200, lr = 4.01052e-05
    I0515 02:08:13.686606 23147 solver.cpp:228] Iteration 75300, loss = 0.0159846
    I0515 02:08:13.686667 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:08:13.686697 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0159845 (* 1 = 0.0159845 loss)
    I0515 02:08:13.686717 23147 sgd_solver.cpp:106] Iteration 75300, lr = 4.00699e-05
    I0515 02:08:21.774623 23147 solver.cpp:228] Iteration 75400, loss = 0.0160407
    I0515 02:08:21.774677 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:08:21.774706 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0160405 (* 1 = 0.0160405 loss)
    I0515 02:08:21.774727 23147 sgd_solver.cpp:106] Iteration 75400, lr = 4.00347e-05
    I0515 02:08:29.878099 23147 solver.cpp:228] Iteration 75500, loss = 0.023106
    I0515 02:08:29.878152 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:08:29.878185 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0231058 (* 1 = 0.0231058 loss)
    I0515 02:08:29.878207 23147 sgd_solver.cpp:106] Iteration 75500, lr = 3.99996e-05
    I0515 02:08:37.972246 23147 solver.cpp:228] Iteration 75600, loss = 0.0382671
    I0515 02:08:37.972501 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 02:08:37.972551 23147 solver.cpp:244]     Train net output #1: loss_c = 0.038267 (* 1 = 0.038267 loss)
    I0515 02:08:37.972576 23147 sgd_solver.cpp:106] Iteration 75600, lr = 3.99645e-05
    I0515 02:08:46.111403 23147 solver.cpp:228] Iteration 75700, loss = 0.0124159
    I0515 02:08:46.111449 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:08:46.111469 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0124158 (* 1 = 0.0124158 loss)
    I0515 02:08:46.111484 23147 sgd_solver.cpp:106] Iteration 75700, lr = 3.99295e-05
    I0515 02:08:54.250859 23147 solver.cpp:228] Iteration 75800, loss = 0.0105043
    I0515 02:08:54.250908 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:08:54.250929 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0105042 (* 1 = 0.0105042 loss)
    I0515 02:08:54.250943 23147 sgd_solver.cpp:106] Iteration 75800, lr = 3.98946e-05
    I0515 02:09:02.390730 23147 solver.cpp:228] Iteration 75900, loss = 0.0268868
    I0515 02:09:02.390784 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:09:02.390805 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0268867 (* 1 = 0.0268867 loss)
    I0515 02:09:02.390820 23147 sgd_solver.cpp:106] Iteration 75900, lr = 3.98598e-05
    I0515 02:09:10.449828 23147 solver.cpp:337] Iteration 76000, Testing net (#0)
    I0515 02:09:14.862429 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.838
    I0515 02:09:14.862479 23147 solver.cpp:404]     Test net output #1: loss_c = 0.51975 (* 1 = 0.51975 loss)
    I0515 02:09:14.917559 23147 solver.cpp:228] Iteration 76000, loss = 0.0189936
    I0515 02:09:14.917618 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:09:14.917646 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0189935 (* 1 = 0.0189935 loss)
    I0515 02:09:14.917671 23147 sgd_solver.cpp:106] Iteration 76000, lr = 3.9825e-05
    I0515 02:09:23.055899 23147 solver.cpp:228] Iteration 76100, loss = 0.0151174
    I0515 02:09:23.055951 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:09:23.055973 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0151173 (* 1 = 0.0151173 loss)
    I0515 02:09:23.055987 23147 sgd_solver.cpp:106] Iteration 76100, lr = 3.97903e-05
    I0515 02:09:31.194387 23147 solver.cpp:228] Iteration 76200, loss = 0.0118297
    I0515 02:09:31.194440 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:09:31.194460 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0118296 (* 1 = 0.0118296 loss)
    I0515 02:09:31.194474 23147 sgd_solver.cpp:106] Iteration 76200, lr = 3.97557e-05
    I0515 02:09:39.310967 23147 solver.cpp:228] Iteration 76300, loss = 0.0239773
    I0515 02:09:39.311012 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:09:39.311033 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0239772 (* 1 = 0.0239772 loss)
    I0515 02:09:39.311048 23147 sgd_solver.cpp:106] Iteration 76300, lr = 3.97212e-05
    I0515 02:09:47.443869 23147 solver.cpp:228] Iteration 76400, loss = 0.0868295
    I0515 02:09:47.444016 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 02:09:47.444062 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0868294 (* 1 = 0.0868294 loss)
    I0515 02:09:47.444087 23147 sgd_solver.cpp:106] Iteration 76400, lr = 3.96867e-05
    I0515 02:09:55.535893 23147 solver.cpp:228] Iteration 76500, loss = 0.0257719
    I0515 02:09:55.535946 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:09:55.535975 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0257718 (* 1 = 0.0257718 loss)
    I0515 02:09:55.535996 23147 sgd_solver.cpp:106] Iteration 76500, lr = 3.96523e-05
    I0515 02:10:03.622335 23147 solver.cpp:228] Iteration 76600, loss = 0.00719705
    I0515 02:10:03.622397 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:10:03.622426 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00719695 (* 1 = 0.00719695 loss)
    I0515 02:10:03.622448 23147 sgd_solver.cpp:106] Iteration 76600, lr = 3.96179e-05
    I0515 02:10:11.708204 23147 solver.cpp:228] Iteration 76700, loss = 0.0104582
    I0515 02:10:11.708263 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:10:11.708292 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0104581 (* 1 = 0.0104581 loss)
    I0515 02:10:11.708313 23147 sgd_solver.cpp:106] Iteration 76700, lr = 3.95836e-05
    I0515 02:10:19.798864 23147 solver.cpp:228] Iteration 76800, loss = 0.0259525
    I0515 02:10:19.799114 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:10:19.799162 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0259524 (* 1 = 0.0259524 loss)
    I0515 02:10:19.799187 23147 sgd_solver.cpp:106] Iteration 76800, lr = 3.95494e-05
    I0515 02:10:27.894245 23147 solver.cpp:228] Iteration 76900, loss = 0.0224117
    I0515 02:10:27.894305 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:10:27.894335 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0224116 (* 1 = 0.0224116 loss)
    I0515 02:10:27.894356 23147 sgd_solver.cpp:106] Iteration 76900, lr = 3.95153e-05
    I0515 02:10:35.901760 23147 solver.cpp:337] Iteration 77000, Testing net (#0)
    I0515 02:10:40.321199 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.82675
    I0515 02:10:40.321249 23147 solver.cpp:404]     Test net output #1: loss_c = 0.524491 (* 1 = 0.524491 loss)
    I0515 02:10:40.372597 23147 solver.cpp:228] Iteration 77000, loss = 0.0107185
    I0515 02:10:40.372633 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:10:40.372653 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0107184 (* 1 = 0.0107184 loss)
    I0515 02:10:40.372671 23147 sgd_solver.cpp:106] Iteration 77000, lr = 3.94812e-05
    I0515 02:10:48.429235 23147 solver.cpp:228] Iteration 77100, loss = 0.0129546
    I0515 02:10:48.429286 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:10:48.429311 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0129545 (* 1 = 0.0129545 loss)
    I0515 02:10:48.429325 23147 sgd_solver.cpp:106] Iteration 77100, lr = 3.94472e-05
    I0515 02:10:56.484540 23147 solver.cpp:228] Iteration 77200, loss = 0.0123667
    I0515 02:10:56.484684 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:10:56.484719 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0123666 (* 1 = 0.0123666 loss)
    I0515 02:10:56.484737 23147 sgd_solver.cpp:106] Iteration 77200, lr = 3.94133e-05
    I0515 02:11:04.624652 23147 solver.cpp:228] Iteration 77300, loss = 0.0394372
    I0515 02:11:04.624699 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:11:04.624718 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0394371 (* 1 = 0.0394371 loss)
    I0515 02:11:04.624733 23147 sgd_solver.cpp:106] Iteration 77300, lr = 3.93794e-05
    I0515 02:11:12.765617 23147 solver.cpp:228] Iteration 77400, loss = 0.0108103
    I0515 02:11:12.765663 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:11:12.765682 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0108102 (* 1 = 0.0108102 loss)
    I0515 02:11:12.765697 23147 sgd_solver.cpp:106] Iteration 77400, lr = 3.93456e-05
    I0515 02:11:20.900262 23147 solver.cpp:228] Iteration 77500, loss = 0.0104323
    I0515 02:11:20.900310 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:11:20.900329 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0104322 (* 1 = 0.0104322 loss)
    I0515 02:11:20.900343 23147 sgd_solver.cpp:106] Iteration 77500, lr = 3.93119e-05
    I0515 02:11:29.025641 23147 solver.cpp:228] Iteration 77600, loss = 0.021956
    I0515 02:11:29.025828 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:11:29.025849 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0219559 (* 1 = 0.0219559 loss)
    I0515 02:11:29.025882 23147 sgd_solver.cpp:106] Iteration 77600, lr = 3.92782e-05
    I0515 02:11:37.162180 23147 solver.cpp:228] Iteration 77700, loss = 0.0317172
    I0515 02:11:37.162225 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:11:37.162245 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0317171 (* 1 = 0.0317171 loss)
    I0515 02:11:37.162259 23147 sgd_solver.cpp:106] Iteration 77700, lr = 3.92446e-05
    I0515 02:11:45.302162 23147 solver.cpp:228] Iteration 77800, loss = 0.0347849
    I0515 02:11:45.302208 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:11:45.302228 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0347848 (* 1 = 0.0347848 loss)
    I0515 02:11:45.302243 23147 sgd_solver.cpp:106] Iteration 77800, lr = 3.92111e-05
    I0515 02:11:53.444192 23147 solver.cpp:228] Iteration 77900, loss = 0.0204722
    I0515 02:11:53.444239 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:11:53.444258 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0204721 (* 1 = 0.0204721 loss)
    I0515 02:11:53.444273 23147 sgd_solver.cpp:106] Iteration 77900, lr = 3.91776e-05
    I0515 02:12:01.494650 23147 solver.cpp:337] Iteration 78000, Testing net (#0)
    I0515 02:12:05.904709 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.818833
    I0515 02:12:05.904758 23147 solver.cpp:404]     Test net output #1: loss_c = 0.542035 (* 1 = 0.542035 loss)
    I0515 02:12:05.956118 23147 solver.cpp:228] Iteration 78000, loss = 0.017645
    I0515 02:12:05.956153 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:12:05.956173 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0176449 (* 1 = 0.0176449 loss)
    I0515 02:12:05.956192 23147 sgd_solver.cpp:106] Iteration 78000, lr = 3.91443e-05
    I0515 02:12:14.091918 23147 solver.cpp:228] Iteration 78100, loss = 0.0162617
    I0515 02:12:14.091974 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:12:14.091995 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0162616 (* 1 = 0.0162616 loss)
    I0515 02:12:14.092011 23147 sgd_solver.cpp:106] Iteration 78100, lr = 3.91109e-05
    I0515 02:12:22.225726 23147 solver.cpp:228] Iteration 78200, loss = 0.0166177
    I0515 02:12:22.225776 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:12:22.225796 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0166176 (* 1 = 0.0166176 loss)
    I0515 02:12:22.225811 23147 sgd_solver.cpp:106] Iteration 78200, lr = 3.90777e-05
    I0515 02:12:30.362661 23147 solver.cpp:228] Iteration 78300, loss = 0.0355771
    I0515 02:12:30.362707 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:12:30.362728 23147 solver.cpp:244]     Train net output #1: loss_c = 0.035577 (* 1 = 0.035577 loss)
    I0515 02:12:30.362742 23147 sgd_solver.cpp:106] Iteration 78300, lr = 3.90445e-05
    I0515 02:12:38.502356 23147 solver.cpp:228] Iteration 78400, loss = 0.0112177
    I0515 02:12:38.502454 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:12:38.502478 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0112176 (* 1 = 0.0112176 loss)
    I0515 02:12:38.502493 23147 sgd_solver.cpp:106] Iteration 78400, lr = 3.90113e-05
    I0515 02:12:46.641851 23147 solver.cpp:228] Iteration 78500, loss = 0.00774758
    I0515 02:12:46.641896 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:12:46.641917 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00774748 (* 1 = 0.00774748 loss)
    I0515 02:12:46.641932 23147 sgd_solver.cpp:106] Iteration 78500, lr = 3.89783e-05
    I0515 02:12:54.780766 23147 solver.cpp:228] Iteration 78600, loss = 0.0145445
    I0515 02:12:54.780813 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:12:54.780834 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0145444 (* 1 = 0.0145444 loss)
    I0515 02:12:54.780848 23147 sgd_solver.cpp:106] Iteration 78600, lr = 3.89453e-05
    I0515 02:13:02.918498 23147 solver.cpp:228] Iteration 78700, loss = 0.0320785
    I0515 02:13:02.918545 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:13:02.918565 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0320784 (* 1 = 0.0320784 loss)
    I0515 02:13:02.918579 23147 sgd_solver.cpp:106] Iteration 78700, lr = 3.89123e-05
    I0515 02:13:11.057193 23147 solver.cpp:228] Iteration 78800, loss = 0.0155443
    I0515 02:13:11.057368 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:13:11.057392 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0155442 (* 1 = 0.0155442 loss)
    I0515 02:13:11.057423 23147 sgd_solver.cpp:106] Iteration 78800, lr = 3.88795e-05
    I0515 02:13:19.196272 23147 solver.cpp:228] Iteration 78900, loss = 0.019062
    I0515 02:13:19.196315 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:13:19.196334 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0190619 (* 1 = 0.0190619 loss)
    I0515 02:13:19.196348 23147 sgd_solver.cpp:106] Iteration 78900, lr = 3.88467e-05
    I0515 02:13:27.258152 23147 solver.cpp:337] Iteration 79000, Testing net (#0)
    I0515 02:13:31.676735 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.807333
    I0515 02:13:31.676794 23147 solver.cpp:404]     Test net output #1: loss_c = 0.547743 (* 1 = 0.547743 loss)
    I0515 02:13:31.728332 23147 solver.cpp:228] Iteration 79000, loss = 0.00650988
    I0515 02:13:31.728421 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:13:31.728451 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00650978 (* 1 = 0.00650978 loss)
    I0515 02:13:31.728472 23147 sgd_solver.cpp:106] Iteration 79000, lr = 3.88139e-05
    I0515 02:13:39.822149 23147 solver.cpp:228] Iteration 79100, loss = 0.00957272
    I0515 02:13:39.822209 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:13:39.822239 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00957263 (* 1 = 0.00957263 loss)
    I0515 02:13:39.822260 23147 sgd_solver.cpp:106] Iteration 79100, lr = 3.87812e-05
    I0515 02:13:47.952266 23147 solver.cpp:228] Iteration 79200, loss = 0.0174867
    I0515 02:13:47.952424 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:13:47.952468 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0174866 (* 1 = 0.0174866 loss)
    I0515 02:13:47.952492 23147 sgd_solver.cpp:106] Iteration 79200, lr = 3.87486e-05
    I0515 02:13:56.082170 23147 solver.cpp:228] Iteration 79300, loss = 0.0175836
    I0515 02:13:56.082208 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:13:56.082228 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0175835 (* 1 = 0.0175835 loss)
    I0515 02:13:56.082242 23147 sgd_solver.cpp:106] Iteration 79300, lr = 3.87161e-05
    I0515 02:14:04.221951 23147 solver.cpp:228] Iteration 79400, loss = 0.012932
    I0515 02:14:04.221990 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:14:04.222010 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0129319 (* 1 = 0.0129319 loss)
    I0515 02:14:04.222025 23147 sgd_solver.cpp:106] Iteration 79400, lr = 3.86836e-05
    I0515 02:14:12.364768 23147 solver.cpp:228] Iteration 79500, loss = 0.0107314
    I0515 02:14:12.364812 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:14:12.364831 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0107313 (* 1 = 0.0107313 loss)
    I0515 02:14:12.364846 23147 sgd_solver.cpp:106] Iteration 79500, lr = 3.86512e-05
    I0515 02:14:20.503257 23147 solver.cpp:228] Iteration 79600, loss = 0.0173685
    I0515 02:14:20.503408 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:14:20.503455 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0173684 (* 1 = 0.0173684 loss)
    I0515 02:14:20.503480 23147 sgd_solver.cpp:106] Iteration 79600, lr = 3.86188e-05
    I0515 02:14:28.594837 23147 solver.cpp:228] Iteration 79700, loss = 0.0360695
    I0515 02:14:28.594898 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:14:28.594928 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0360694 (* 1 = 0.0360694 loss)
    I0515 02:14:28.594951 23147 sgd_solver.cpp:106] Iteration 79700, lr = 3.85865e-05
    I0515 02:14:36.686926 23147 solver.cpp:228] Iteration 79800, loss = 0.0304676
    I0515 02:14:36.686982 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:14:36.687011 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0304676 (* 1 = 0.0304676 loss)
    I0515 02:14:36.687033 23147 sgd_solver.cpp:106] Iteration 79800, lr = 3.85543e-05
    I0515 02:14:44.772370 23147 solver.cpp:228] Iteration 79900, loss = 0.0504928
    I0515 02:14:44.772431 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:14:44.772462 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0504927 (* 1 = 0.0504927 loss)
    I0515 02:14:44.772483 23147 sgd_solver.cpp:106] Iteration 79900, lr = 3.85221e-05
    I0515 02:14:52.828421 23147 solver.cpp:337] Iteration 80000, Testing net (#0)
    I0515 02:14:57.191311 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.82275
    I0515 02:14:57.191378 23147 solver.cpp:404]     Test net output #1: loss_c = 0.562647 (* 1 = 0.562647 loss)
    I0515 02:14:57.247561 23147 solver.cpp:228] Iteration 80000, loss = 0.0160929
    I0515 02:14:57.247597 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:14:57.247624 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0160928 (* 1 = 0.0160928 loss)
    I0515 02:14:57.247655 23147 sgd_solver.cpp:106] Iteration 80000, lr = 3.849e-05
    I0515 02:15:05.338393 23147 solver.cpp:228] Iteration 80100, loss = 0.0396383
    I0515 02:15:05.338454 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:15:05.338484 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0396382 (* 1 = 0.0396382 loss)
    I0515 02:15:05.338505 23147 sgd_solver.cpp:106] Iteration 80100, lr = 3.8458e-05
    I0515 02:15:13.433290 23147 solver.cpp:228] Iteration 80200, loss = 0.0115364
    I0515 02:15:13.433346 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:15:13.433377 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0115363 (* 1 = 0.0115363 loss)
    I0515 02:15:13.433398 23147 sgd_solver.cpp:106] Iteration 80200, lr = 3.8426e-05
    I0515 02:15:21.523844 23147 solver.cpp:228] Iteration 80300, loss = 0.0106204
    I0515 02:15:21.523905 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:15:21.523934 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0106203 (* 1 = 0.0106203 loss)
    I0515 02:15:21.523957 23147 sgd_solver.cpp:106] Iteration 80300, lr = 3.83941e-05
    I0515 02:15:29.632086 23147 solver.cpp:228] Iteration 80400, loss = 0.00444463
    I0515 02:15:29.632171 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:15:29.632192 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00444454 (* 1 = 0.00444454 loss)
    I0515 02:15:29.632207 23147 sgd_solver.cpp:106] Iteration 80400, lr = 3.83622e-05
    I0515 02:15:37.776388 23147 solver.cpp:228] Iteration 80500, loss = 0.0237959
    I0515 02:15:37.776429 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:15:37.776448 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0237958 (* 1 = 0.0237958 loss)
    I0515 02:15:37.776463 23147 sgd_solver.cpp:106] Iteration 80500, lr = 3.83304e-05
    I0515 02:15:45.916820 23147 solver.cpp:228] Iteration 80600, loss = 0.019189
    I0515 02:15:45.916863 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:15:45.916885 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0191889 (* 1 = 0.0191889 loss)
    I0515 02:15:45.916899 23147 sgd_solver.cpp:106] Iteration 80600, lr = 3.82987e-05
    I0515 02:15:54.058069 23147 solver.cpp:228] Iteration 80700, loss = 0.0194112
    I0515 02:15:54.058112 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:15:54.058135 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0194111 (* 1 = 0.0194111 loss)
    I0515 02:15:54.058149 23147 sgd_solver.cpp:106] Iteration 80700, lr = 3.8267e-05
    I0515 02:16:02.199612 23147 solver.cpp:228] Iteration 80800, loss = 0.0200736
    I0515 02:16:02.199815 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:16:02.199836 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0200735 (* 1 = 0.0200735 loss)
    I0515 02:16:02.199851 23147 sgd_solver.cpp:106] Iteration 80800, lr = 3.82354e-05
    I0515 02:16:10.341598 23147 solver.cpp:228] Iteration 80900, loss = 0.0252763
    I0515 02:16:10.341639 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:16:10.341660 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0252762 (* 1 = 0.0252762 loss)
    I0515 02:16:10.341675 23147 sgd_solver.cpp:106] Iteration 80900, lr = 3.82038e-05
    I0515 02:16:18.402158 23147 solver.cpp:337] Iteration 81000, Testing net (#0)
    I0515 02:16:22.820144 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.8205
    I0515 02:16:22.820196 23147 solver.cpp:404]     Test net output #1: loss_c = 0.538553 (* 1 = 0.538553 loss)
    I0515 02:16:22.871510 23147 solver.cpp:228] Iteration 81000, loss = 0.0167072
    I0515 02:16:22.871549 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:16:22.871568 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0167071 (* 1 = 0.0167071 loss)
    I0515 02:16:22.871587 23147 sgd_solver.cpp:106] Iteration 81000, lr = 3.81724e-05
    I0515 02:16:31.007032 23147 solver.cpp:228] Iteration 81100, loss = 0.023969
    I0515 02:16:31.007076 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:16:31.007096 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0239689 (* 1 = 0.0239689 loss)
    I0515 02:16:31.007110 23147 sgd_solver.cpp:106] Iteration 81100, lr = 3.81409e-05
    I0515 02:16:39.144065 23147 solver.cpp:228] Iteration 81200, loss = 0.0110044
    I0515 02:16:39.144165 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:16:39.144186 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0110043 (* 1 = 0.0110043 loss)
    I0515 02:16:39.144201 23147 sgd_solver.cpp:106] Iteration 81200, lr = 3.81096e-05
    I0515 02:16:47.281324 23147 solver.cpp:228] Iteration 81300, loss = 0.0224656
    I0515 02:16:47.281373 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:16:47.281393 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0224655 (* 1 = 0.0224655 loss)
    I0515 02:16:47.281406 23147 sgd_solver.cpp:106] Iteration 81300, lr = 3.80782e-05
    I0515 02:16:55.422389 23147 solver.cpp:228] Iteration 81400, loss = 0.0139266
    I0515 02:16:55.422427 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:16:55.422451 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0139265 (* 1 = 0.0139265 loss)
    I0515 02:16:55.422466 23147 sgd_solver.cpp:106] Iteration 81400, lr = 3.8047e-05
    I0515 02:17:03.560340 23147 solver.cpp:228] Iteration 81500, loss = 0.0135511
    I0515 02:17:03.560391 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:17:03.560412 23147 solver.cpp:244]     Train net output #1: loss_c = 0.013551 (* 1 = 0.013551 loss)
    I0515 02:17:03.560426 23147 sgd_solver.cpp:106] Iteration 81500, lr = 3.80158e-05
    I0515 02:17:11.702119 23147 solver.cpp:228] Iteration 81600, loss = 0.0190625
    I0515 02:17:11.702217 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:17:11.702239 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0190624 (* 1 = 0.0190624 loss)
    I0515 02:17:11.702253 23147 sgd_solver.cpp:106] Iteration 81600, lr = 3.79847e-05
    I0515 02:17:19.842298 23147 solver.cpp:228] Iteration 81700, loss = 0.0157008
    I0515 02:17:19.842351 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:17:19.842375 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0157008 (* 1 = 0.0157008 loss)
    I0515 02:17:19.842388 23147 sgd_solver.cpp:106] Iteration 81700, lr = 3.79536e-05
    I0515 02:17:27.971035 23147 solver.cpp:228] Iteration 81800, loss = 0.025957
    I0515 02:17:27.971078 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:17:27.971098 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0259569 (* 1 = 0.0259569 loss)
    I0515 02:17:27.971112 23147 sgd_solver.cpp:106] Iteration 81800, lr = 3.79226e-05
    I0515 02:17:36.107892 23147 solver.cpp:228] Iteration 81900, loss = 0.00845481
    I0515 02:17:36.107946 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:17:36.107966 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00845472 (* 1 = 0.00845472 loss)
    I0515 02:17:36.107980 23147 sgd_solver.cpp:106] Iteration 81900, lr = 3.78916e-05
    I0515 02:17:44.165743 23147 solver.cpp:337] Iteration 82000, Testing net (#0)
    I0515 02:17:48.552623 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.829417
    I0515 02:17:48.552675 23147 solver.cpp:404]     Test net output #1: loss_c = 0.497183 (* 1 = 0.497183 loss)
    I0515 02:17:48.604002 23147 solver.cpp:228] Iteration 82000, loss = 0.0184672
    I0515 02:17:48.604053 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:17:48.604071 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0184671 (* 1 = 0.0184671 loss)
    I0515 02:17:48.604090 23147 sgd_solver.cpp:106] Iteration 82000, lr = 3.78607e-05
    I0515 02:17:56.726374 23147 solver.cpp:228] Iteration 82100, loss = 0.01901
    I0515 02:17:56.726419 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:17:56.726439 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0190099 (* 1 = 0.0190099 loss)
    I0515 02:17:56.726454 23147 sgd_solver.cpp:106] Iteration 82100, lr = 3.78299e-05
    I0515 02:18:04.866061 23147 solver.cpp:228] Iteration 82200, loss = 0.025993
    I0515 02:18:04.866109 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:18:04.866129 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0259929 (* 1 = 0.0259929 loss)
    I0515 02:18:04.866144 23147 sgd_solver.cpp:106] Iteration 82200, lr = 3.77991e-05
    I0515 02:18:13.006165 23147 solver.cpp:228] Iteration 82300, loss = 0.0112037
    I0515 02:18:13.006208 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:18:13.006228 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0112036 (* 1 = 0.0112036 loss)
    I0515 02:18:13.006243 23147 sgd_solver.cpp:106] Iteration 82300, lr = 3.77684e-05
    I0515 02:18:21.145954 23147 solver.cpp:228] Iteration 82400, loss = 0.00582379
    I0515 02:18:21.146059 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:18:21.146080 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00582371 (* 1 = 0.00582371 loss)
    I0515 02:18:21.146095 23147 sgd_solver.cpp:106] Iteration 82400, lr = 3.77378e-05
    I0515 02:18:29.284946 23147 solver.cpp:228] Iteration 82500, loss = 0.026589
    I0515 02:18:29.284994 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:18:29.285014 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0265889 (* 1 = 0.0265889 loss)
    I0515 02:18:29.285029 23147 sgd_solver.cpp:106] Iteration 82500, lr = 3.77071e-05
    I0515 02:18:37.422786 23147 solver.cpp:228] Iteration 82600, loss = 0.0375462
    I0515 02:18:37.422834 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:18:37.422853 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0375462 (* 1 = 0.0375462 loss)
    I0515 02:18:37.422868 23147 sgd_solver.cpp:106] Iteration 82600, lr = 3.76766e-05
    I0515 02:18:45.563467 23147 solver.cpp:228] Iteration 82700, loss = 0.00706927
    I0515 02:18:45.563516 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:18:45.563536 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0070692 (* 1 = 0.0070692 loss)
    I0515 02:18:45.563550 23147 sgd_solver.cpp:106] Iteration 82700, lr = 3.76461e-05
    I0515 02:18:53.698443 23147 solver.cpp:228] Iteration 82800, loss = 0.0153206
    I0515 02:18:53.698688 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:18:53.698735 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0153206 (* 1 = 0.0153206 loss)
    I0515 02:18:53.698760 23147 sgd_solver.cpp:106] Iteration 82800, lr = 3.76157e-05
    I0515 02:19:01.837455 23147 solver.cpp:228] Iteration 82900, loss = 0.0259642
    I0515 02:19:01.837496 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:19:01.837517 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0259642 (* 1 = 0.0259642 loss)
    I0515 02:19:01.837532 23147 sgd_solver.cpp:106] Iteration 82900, lr = 3.75853e-05
    I0515 02:19:09.890753 23147 solver.cpp:337] Iteration 83000, Testing net (#0)
    I0515 02:19:14.251060 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.80975
    I0515 02:19:14.251111 23147 solver.cpp:404]     Test net output #1: loss_c = 0.544342 (* 1 = 0.544342 loss)
    I0515 02:19:14.302440 23147 solver.cpp:228] Iteration 83000, loss = 0.0165952
    I0515 02:19:14.302477 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:19:14.302496 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0165951 (* 1 = 0.0165951 loss)
    I0515 02:19:14.302515 23147 sgd_solver.cpp:106] Iteration 83000, lr = 3.7555e-05
    I0515 02:19:22.439627 23147 solver.cpp:228] Iteration 83100, loss = 0.0647178
    I0515 02:19:22.439673 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:19:22.439693 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0647177 (* 1 = 0.0647177 loss)
    I0515 02:19:22.439708 23147 sgd_solver.cpp:106] Iteration 83100, lr = 3.75247e-05
    I0515 02:19:30.570989 23147 solver.cpp:228] Iteration 83200, loss = 0.008538
    I0515 02:19:30.571080 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:19:30.571100 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00853794 (* 1 = 0.00853794 loss)
    I0515 02:19:30.571115 23147 sgd_solver.cpp:106] Iteration 83200, lr = 3.74945e-05
    I0515 02:19:38.712019 23147 solver.cpp:228] Iteration 83300, loss = 0.0177525
    I0515 02:19:38.712067 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:19:38.712087 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0177524 (* 1 = 0.0177524 loss)
    I0515 02:19:38.712101 23147 sgd_solver.cpp:106] Iteration 83300, lr = 3.74644e-05
    I0515 02:19:46.844580 23147 solver.cpp:228] Iteration 83400, loss = 0.0118211
    I0515 02:19:46.844626 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:19:46.844646 23147 solver.cpp:244]     Train net output #1: loss_c = 0.011821 (* 1 = 0.011821 loss)
    I0515 02:19:46.844661 23147 sgd_solver.cpp:106] Iteration 83400, lr = 3.74343e-05
    I0515 02:19:54.972519 23147 solver.cpp:228] Iteration 83500, loss = 0.0122506
    I0515 02:19:54.972564 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:19:54.972585 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0122506 (* 1 = 0.0122506 loss)
    I0515 02:19:54.972599 23147 sgd_solver.cpp:106] Iteration 83500, lr = 3.74043e-05
    I0515 02:20:03.111048 23147 solver.cpp:228] Iteration 83600, loss = 0.0217696
    I0515 02:20:03.111202 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:20:03.111248 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0217695 (* 1 = 0.0217695 loss)
    I0515 02:20:03.111273 23147 sgd_solver.cpp:106] Iteration 83600, lr = 3.73743e-05
    I0515 02:20:11.245550 23147 solver.cpp:228] Iteration 83700, loss = 0.0244954
    I0515 02:20:11.245594 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:20:11.245614 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0244953 (* 1 = 0.0244953 loss)
    I0515 02:20:11.245628 23147 sgd_solver.cpp:106] Iteration 83700, lr = 3.73444e-05
    I0515 02:20:19.383087 23147 solver.cpp:228] Iteration 83800, loss = 0.00667851
    I0515 02:20:19.383132 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:20:19.383152 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00667845 (* 1 = 0.00667845 loss)
    I0515 02:20:19.383167 23147 sgd_solver.cpp:106] Iteration 83800, lr = 3.73145e-05
    I0515 02:20:27.523942 23147 solver.cpp:228] Iteration 83900, loss = 0.0264333
    I0515 02:20:27.523988 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:20:27.524008 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0264332 (* 1 = 0.0264332 loss)
    I0515 02:20:27.524021 23147 sgd_solver.cpp:106] Iteration 83900, lr = 3.72847e-05
    I0515 02:20:35.585134 23147 solver.cpp:337] Iteration 84000, Testing net (#0)
    I0515 02:20:40.011924 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.818083
    I0515 02:20:40.011988 23147 solver.cpp:404]     Test net output #1: loss_c = 0.532545 (* 1 = 0.532545 loss)
    I0515 02:20:40.064281 23147 solver.cpp:228] Iteration 84000, loss = 0.0156704
    I0515 02:20:40.064359 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:20:40.064383 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0156703 (* 1 = 0.0156703 loss)
    I0515 02:20:40.064401 23147 sgd_solver.cpp:106] Iteration 84000, lr = 3.7255e-05
    I0515 02:20:48.199676 23147 solver.cpp:228] Iteration 84100, loss = 0.025579
    I0515 02:20:48.199723 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:20:48.199748 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0255789 (* 1 = 0.0255789 loss)
    I0515 02:20:48.199761 23147 sgd_solver.cpp:106] Iteration 84100, lr = 3.72253e-05
    I0515 02:20:56.339699 23147 solver.cpp:228] Iteration 84200, loss = 0.0130579
    I0515 02:20:56.339745 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:20:56.339768 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0130578 (* 1 = 0.0130578 loss)
    I0515 02:20:56.339782 23147 sgd_solver.cpp:106] Iteration 84200, lr = 3.71956e-05
    I0515 02:21:04.478253 23147 solver.cpp:228] Iteration 84300, loss = 0.00592979
    I0515 02:21:04.478302 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:21:04.478323 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00592972 (* 1 = 0.00592972 loss)
    I0515 02:21:04.478338 23147 sgd_solver.cpp:106] Iteration 84300, lr = 3.7166e-05
    I0515 02:21:12.614148 23147 solver.cpp:228] Iteration 84400, loss = 0.0646666
    I0515 02:21:12.614294 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:21:12.614338 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0646665 (* 1 = 0.0646665 loss)
    I0515 02:21:12.614364 23147 sgd_solver.cpp:106] Iteration 84400, lr = 3.71365e-05
    I0515 02:21:20.749606 23147 solver.cpp:228] Iteration 84500, loss = 0.0169923
    I0515 02:21:20.749653 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:21:20.749672 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0169923 (* 1 = 0.0169923 loss)
    I0515 02:21:20.749687 23147 sgd_solver.cpp:106] Iteration 84500, lr = 3.7107e-05
    I0515 02:21:28.889526 23147 solver.cpp:228] Iteration 84600, loss = 0.027611
    I0515 02:21:28.889572 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:21:28.889592 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0276109 (* 1 = 0.0276109 loss)
    I0515 02:21:28.889607 23147 sgd_solver.cpp:106] Iteration 84600, lr = 3.70776e-05
    I0515 02:21:37.029218 23147 solver.cpp:228] Iteration 84700, loss = 0.0209868
    I0515 02:21:37.029263 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:21:37.029283 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0209868 (* 1 = 0.0209868 loss)
    I0515 02:21:37.029297 23147 sgd_solver.cpp:106] Iteration 84700, lr = 3.70482e-05
    I0515 02:21:45.170565 23147 solver.cpp:228] Iteration 84800, loss = 0.0138234
    I0515 02:21:45.170714 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:21:45.170760 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0138234 (* 1 = 0.0138234 loss)
    I0515 02:21:45.170785 23147 sgd_solver.cpp:106] Iteration 84800, lr = 3.70189e-05
    I0515 02:21:53.315928 23147 solver.cpp:228] Iteration 84900, loss = 0.0181054
    I0515 02:21:53.315969 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:21:53.315991 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0181054 (* 1 = 0.0181054 loss)
    I0515 02:21:53.316005 23147 sgd_solver.cpp:106] Iteration 84900, lr = 3.69897e-05
    I0515 02:22:01.377400 23147 solver.cpp:337] Iteration 85000, Testing net (#0)
    I0515 02:22:05.799473 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.825167
    I0515 02:22:05.799527 23147 solver.cpp:404]     Test net output #1: loss_c = 0.551445 (* 1 = 0.551445 loss)
    I0515 02:22:05.850841 23147 solver.cpp:228] Iteration 85000, loss = 0.0163918
    I0515 02:22:05.850865 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:22:05.850884 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0163917 (* 1 = 0.0163917 loss)
    I0515 02:22:05.850903 23147 sgd_solver.cpp:106] Iteration 85000, lr = 3.69605e-05
    I0515 02:22:13.975410 23147 solver.cpp:228] Iteration 85100, loss = 0.0130412
    I0515 02:22:13.975458 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:22:13.975479 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0130412 (* 1 = 0.0130412 loss)
    I0515 02:22:13.975495 23147 sgd_solver.cpp:106] Iteration 85100, lr = 3.69313e-05
    I0515 02:22:22.110404 23147 solver.cpp:228] Iteration 85200, loss = 0.0122601
    I0515 02:22:22.110635 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:22:22.110682 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0122601 (* 1 = 0.0122601 loss)
    I0515 02:22:22.110707 23147 sgd_solver.cpp:106] Iteration 85200, lr = 3.69022e-05
    I0515 02:22:30.201427 23147 solver.cpp:228] Iteration 85300, loss = 0.0248256
    I0515 02:22:30.201483 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:22:30.201511 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0248255 (* 1 = 0.0248255 loss)
    I0515 02:22:30.201534 23147 sgd_solver.cpp:106] Iteration 85300, lr = 3.68732e-05
    I0515 02:22:38.288228 23147 solver.cpp:228] Iteration 85400, loss = 0.0119658
    I0515 02:22:38.288286 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:22:38.288316 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0119657 (* 1 = 0.0119657 loss)
    I0515 02:22:38.288336 23147 sgd_solver.cpp:106] Iteration 85400, lr = 3.68442e-05
    I0515 02:22:46.385066 23147 solver.cpp:228] Iteration 85500, loss = 0.0182455
    I0515 02:22:46.385128 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:22:46.385156 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0182454 (* 1 = 0.0182454 loss)
    I0515 02:22:46.385177 23147 sgd_solver.cpp:106] Iteration 85500, lr = 3.68152e-05
    I0515 02:22:54.519448 23147 solver.cpp:228] Iteration 85600, loss = 0.0211806
    I0515 02:22:54.519554 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:22:54.519575 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0211806 (* 1 = 0.0211806 loss)
    I0515 02:22:54.519590 23147 sgd_solver.cpp:106] Iteration 85600, lr = 3.67863e-05
    I0515 02:23:02.662313 23147 solver.cpp:228] Iteration 85700, loss = 0.0106857
    I0515 02:23:02.662358 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:23:02.662379 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0106856 (* 1 = 0.0106856 loss)
    I0515 02:23:02.662394 23147 sgd_solver.cpp:106] Iteration 85700, lr = 3.67575e-05
    I0515 02:23:10.804421 23147 solver.cpp:228] Iteration 85800, loss = 0.0351725
    I0515 02:23:10.804467 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:23:10.804487 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0351725 (* 1 = 0.0351725 loss)
    I0515 02:23:10.804502 23147 sgd_solver.cpp:106] Iteration 85800, lr = 3.67287e-05
    I0515 02:23:18.941074 23147 solver.cpp:228] Iteration 85900, loss = 0.0142208
    I0515 02:23:18.941120 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:23:18.941140 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0142208 (* 1 = 0.0142208 loss)
    I0515 02:23:18.941154 23147 sgd_solver.cpp:106] Iteration 85900, lr = 3.67e-05
    I0515 02:23:26.970543 23147 solver.cpp:337] Iteration 86000, Testing net (#0)
    I0515 02:23:31.394928 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.812
    I0515 02:23:31.394978 23147 solver.cpp:404]     Test net output #1: loss_c = 0.582322 (* 1 = 0.582322 loss)
    I0515 02:23:31.447168 23147 solver.cpp:228] Iteration 86000, loss = 0.021213
    I0515 02:23:31.447201 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:23:31.447218 23147 solver.cpp:244]     Train net output #1: loss_c = 0.021213 (* 1 = 0.021213 loss)
    I0515 02:23:31.447237 23147 sgd_solver.cpp:106] Iteration 86000, lr = 3.66713e-05
    I0515 02:23:39.579792 23147 solver.cpp:228] Iteration 86100, loss = 0.0297236
    I0515 02:23:39.579839 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:23:39.579859 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0297236 (* 1 = 0.0297236 loss)
    I0515 02:23:39.579875 23147 sgd_solver.cpp:106] Iteration 86100, lr = 3.66427e-05
    I0515 02:23:47.720870 23147 solver.cpp:228] Iteration 86200, loss = 0.043339
    I0515 02:23:47.720916 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:23:47.720935 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0433389 (* 1 = 0.0433389 loss)
    I0515 02:23:47.720949 23147 sgd_solver.cpp:106] Iteration 86200, lr = 3.66141e-05
    I0515 02:23:55.856849 23147 solver.cpp:228] Iteration 86300, loss = 0.0366913
    I0515 02:23:55.856896 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:23:55.856916 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0366913 (* 1 = 0.0366913 loss)
    I0515 02:23:55.856930 23147 sgd_solver.cpp:106] Iteration 86300, lr = 3.65856e-05
    I0515 02:24:03.996440 23147 solver.cpp:228] Iteration 86400, loss = 0.0331358
    I0515 02:24:03.996534 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:24:03.996554 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0331358 (* 1 = 0.0331358 loss)
    I0515 02:24:03.996569 23147 sgd_solver.cpp:106] Iteration 86400, lr = 3.65571e-05
    I0515 02:24:12.137383 23147 solver.cpp:228] Iteration 86500, loss = 0.00984709
    I0515 02:24:12.137428 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:24:12.137449 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00984706 (* 1 = 0.00984706 loss)
    I0515 02:24:12.137464 23147 sgd_solver.cpp:106] Iteration 86500, lr = 3.65287e-05
    I0515 02:24:20.276839 23147 solver.cpp:228] Iteration 86600, loss = 0.0321649
    I0515 02:24:20.276886 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:24:20.276908 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0321649 (* 1 = 0.0321649 loss)
    I0515 02:24:20.276923 23147 sgd_solver.cpp:106] Iteration 86600, lr = 3.65004e-05
    I0515 02:24:28.384938 23147 solver.cpp:228] Iteration 86700, loss = 0.0198743
    I0515 02:24:28.384981 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:24:28.385001 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0198742 (* 1 = 0.0198742 loss)
    I0515 02:24:28.385015 23147 sgd_solver.cpp:106] Iteration 86700, lr = 3.6472e-05
    I0515 02:24:36.526260 23147 solver.cpp:228] Iteration 86800, loss = 0.0127933
    I0515 02:24:36.526648 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:24:36.526671 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0127933 (* 1 = 0.0127933 loss)
    I0515 02:24:36.526690 23147 sgd_solver.cpp:106] Iteration 86800, lr = 3.64438e-05
    I0515 02:24:44.657739 23147 solver.cpp:228] Iteration 86900, loss = 0.00871812
    I0515 02:24:44.657783 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:24:44.657804 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00871808 (* 1 = 0.00871808 loss)
    I0515 02:24:44.657819 23147 sgd_solver.cpp:106] Iteration 86900, lr = 3.64156e-05
    I0515 02:24:52.712649 23147 solver.cpp:337] Iteration 87000, Testing net (#0)
    I0515 02:24:57.131048 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.825917
    I0515 02:24:57.131099 23147 solver.cpp:404]     Test net output #1: loss_c = 0.528843 (* 1 = 0.528843 loss)
    I0515 02:24:57.186080 23147 solver.cpp:228] Iteration 87000, loss = 0.0125759
    I0515 02:24:57.186138 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:24:57.186170 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0125758 (* 1 = 0.0125758 loss)
    I0515 02:24:57.186195 23147 sgd_solver.cpp:106] Iteration 87000, lr = 3.63874e-05
    I0515 02:25:05.272495 23147 solver.cpp:228] Iteration 87100, loss = 0.00804374
    I0515 02:25:05.272557 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:25:05.272585 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00804371 (* 1 = 0.00804371 loss)
    I0515 02:25:05.272608 23147 sgd_solver.cpp:106] Iteration 87100, lr = 3.63593e-05
    I0515 02:25:13.361101 23147 solver.cpp:228] Iteration 87200, loss = 0.0110814
    I0515 02:25:13.361337 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:25:13.361388 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0110814 (* 1 = 0.0110814 loss)
    I0515 02:25:13.361412 23147 sgd_solver.cpp:106] Iteration 87200, lr = 3.63312e-05
    I0515 02:25:21.446089 23147 solver.cpp:228] Iteration 87300, loss = 0.0091338
    I0515 02:25:21.446149 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:25:21.446178 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00913379 (* 1 = 0.00913379 loss)
    I0515 02:25:21.446199 23147 sgd_solver.cpp:106] Iteration 87300, lr = 3.63032e-05
    I0515 02:25:29.530505 23147 solver.cpp:228] Iteration 87400, loss = 0.0191103
    I0515 02:25:29.530562 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:25:29.530591 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0191103 (* 1 = 0.0191103 loss)
    I0515 02:25:29.530612 23147 sgd_solver.cpp:106] Iteration 87400, lr = 3.62753e-05
    I0515 02:25:37.637212 23147 solver.cpp:228] Iteration 87500, loss = 0.00814893
    I0515 02:25:37.637258 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:25:37.637279 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00814893 (* 1 = 0.00814893 loss)
    I0515 02:25:37.637295 23147 sgd_solver.cpp:106] Iteration 87500, lr = 3.62474e-05
    I0515 02:25:45.773221 23147 solver.cpp:228] Iteration 87600, loss = 0.0200138
    I0515 02:25:45.773360 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:25:45.773406 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0200138 (* 1 = 0.0200138 loss)
    I0515 02:25:45.773430 23147 sgd_solver.cpp:106] Iteration 87600, lr = 3.62195e-05
    I0515 02:25:53.909741 23147 solver.cpp:228] Iteration 87700, loss = 0.0250163
    I0515 02:25:53.909797 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:25:53.909823 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0250163 (* 1 = 0.0250163 loss)
    I0515 02:25:53.909838 23147 sgd_solver.cpp:106] Iteration 87700, lr = 3.61917e-05
    I0515 02:26:02.042419 23147 solver.cpp:228] Iteration 87800, loss = 0.0163782
    I0515 02:26:02.042465 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:26:02.042484 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0163782 (* 1 = 0.0163782 loss)
    I0515 02:26:02.042500 23147 sgd_solver.cpp:106] Iteration 87800, lr = 3.6164e-05
    I0515 02:26:10.175153 23147 solver.cpp:228] Iteration 87900, loss = 0.00993676
    I0515 02:26:10.175199 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:26:10.175220 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00993676 (* 1 = 0.00993676 loss)
    I0515 02:26:10.175235 23147 sgd_solver.cpp:106] Iteration 87900, lr = 3.61362e-05
    I0515 02:26:18.233934 23147 solver.cpp:337] Iteration 88000, Testing net (#0)
    I0515 02:26:22.649700 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.831833
    I0515 02:26:22.649751 23147 solver.cpp:404]     Test net output #1: loss_c = 0.519084 (* 1 = 0.519084 loss)
    I0515 02:26:22.705085 23147 solver.cpp:228] Iteration 88000, loss = 0.0117037
    I0515 02:26:22.705154 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:26:22.705184 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0117037 (* 1 = 0.0117037 loss)
    I0515 02:26:22.705210 23147 sgd_solver.cpp:106] Iteration 88000, lr = 3.61086e-05
    I0515 02:26:30.834617 23147 solver.cpp:228] Iteration 88100, loss = 0.021103
    I0515 02:26:30.834658 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:26:30.834681 23147 solver.cpp:244]     Train net output #1: loss_c = 0.021103 (* 1 = 0.021103 loss)
    I0515 02:26:30.834695 23147 sgd_solver.cpp:106] Iteration 88100, lr = 3.6081e-05
    I0515 02:26:38.970978 23147 solver.cpp:228] Iteration 88200, loss = 0.0151637
    I0515 02:26:38.971024 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:26:38.971043 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0151637 (* 1 = 0.0151637 loss)
    I0515 02:26:38.971058 23147 sgd_solver.cpp:106] Iteration 88200, lr = 3.60534e-05
    I0515 02:26:47.109565 23147 solver.cpp:228] Iteration 88300, loss = 0.023828
    I0515 02:26:47.109616 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:26:47.109635 23147 solver.cpp:244]     Train net output #1: loss_c = 0.023828 (* 1 = 0.023828 loss)
    I0515 02:26:47.109650 23147 sgd_solver.cpp:106] Iteration 88300, lr = 3.60259e-05
    I0515 02:26:55.248260 23147 solver.cpp:228] Iteration 88400, loss = 0.0205154
    I0515 02:26:55.248458 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:26:55.248508 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0205154 (* 1 = 0.0205154 loss)
    I0515 02:26:55.248524 23147 sgd_solver.cpp:106] Iteration 88400, lr = 3.59984e-05
    I0515 02:27:03.384575 23147 solver.cpp:228] Iteration 88500, loss = 0.00972216
    I0515 02:27:03.384625 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:27:03.384647 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00972217 (* 1 = 0.00972217 loss)
    I0515 02:27:03.384662 23147 sgd_solver.cpp:106] Iteration 88500, lr = 3.5971e-05
    I0515 02:27:11.521397 23147 solver.cpp:228] Iteration 88600, loss = 0.0514689
    I0515 02:27:11.521447 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:27:11.521468 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0514689 (* 1 = 0.0514689 loss)
    I0515 02:27:11.521483 23147 sgd_solver.cpp:106] Iteration 88600, lr = 3.59437e-05
    I0515 02:27:19.630733 23147 solver.cpp:228] Iteration 88700, loss = 0.0268013
    I0515 02:27:19.630780 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:27:19.630801 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0268013 (* 1 = 0.0268013 loss)
    I0515 02:27:19.630816 23147 sgd_solver.cpp:106] Iteration 88700, lr = 3.59163e-05
    I0515 02:27:27.769697 23147 solver.cpp:228] Iteration 88800, loss = 0.0142486
    I0515 02:27:27.769789 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:27:27.769809 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0142486 (* 1 = 0.0142486 loss)
    I0515 02:27:27.769824 23147 sgd_solver.cpp:106] Iteration 88800, lr = 3.58891e-05
    I0515 02:27:35.896255 23147 solver.cpp:228] Iteration 88900, loss = 0.0155931
    I0515 02:27:35.896296 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:27:35.896317 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0155931 (* 1 = 0.0155931 loss)
    I0515 02:27:35.896334 23147 sgd_solver.cpp:106] Iteration 88900, lr = 3.58619e-05
    I0515 02:27:43.955428 23147 solver.cpp:337] Iteration 89000, Testing net (#0)
    I0515 02:27:48.364089 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.822833
    I0515 02:27:48.364145 23147 solver.cpp:404]     Test net output #1: loss_c = 0.538153 (* 1 = 0.538153 loss)
    I0515 02:27:48.416220 23147 solver.cpp:228] Iteration 89000, loss = 0.0269284
    I0515 02:27:48.416244 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:27:48.416263 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0269284 (* 1 = 0.0269284 loss)
    I0515 02:27:48.416282 23147 sgd_solver.cpp:106] Iteration 89000, lr = 3.58347e-05
    I0515 02:27:56.558883 23147 solver.cpp:228] Iteration 89100, loss = 0.00789669
    I0515 02:27:56.558934 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:27:56.558955 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00789671 (* 1 = 0.00789671 loss)
    I0515 02:27:56.558970 23147 sgd_solver.cpp:106] Iteration 89100, lr = 3.58076e-05
    I0515 02:28:04.698992 23147 solver.cpp:228] Iteration 89200, loss = 0.0211797
    I0515 02:28:04.699152 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:28:04.699173 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0211797 (* 1 = 0.0211797 loss)
    I0515 02:28:04.699188 23147 sgd_solver.cpp:106] Iteration 89200, lr = 3.57805e-05
    I0515 02:28:12.835156 23147 solver.cpp:228] Iteration 89300, loss = 0.0179637
    I0515 02:28:12.835202 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:28:12.835222 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0179638 (* 1 = 0.0179638 loss)
    I0515 02:28:12.835237 23147 sgd_solver.cpp:106] Iteration 89300, lr = 3.57535e-05
    I0515 02:28:20.975811 23147 solver.cpp:228] Iteration 89400, loss = 0.00756621
    I0515 02:28:20.975857 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:28:20.975883 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00756623 (* 1 = 0.00756623 loss)
    I0515 02:28:20.975896 23147 sgd_solver.cpp:106] Iteration 89400, lr = 3.57265e-05
    I0515 02:28:29.107470 23147 solver.cpp:228] Iteration 89500, loss = 0.00795836
    I0515 02:28:29.107523 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:28:29.107544 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00795837 (* 1 = 0.00795837 loss)
    I0515 02:28:29.107559 23147 sgd_solver.cpp:106] Iteration 89500, lr = 3.56995e-05
    I0515 02:28:37.243155 23147 solver.cpp:228] Iteration 89600, loss = 0.0163764
    I0515 02:28:37.243306 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:28:37.243351 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0163764 (* 1 = 0.0163764 loss)
    I0515 02:28:37.243376 23147 sgd_solver.cpp:106] Iteration 89600, lr = 3.56727e-05
    I0515 02:28:45.383111 23147 solver.cpp:228] Iteration 89700, loss = 0.0228809
    I0515 02:28:45.383152 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:28:45.383170 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0228809 (* 1 = 0.0228809 loss)
    I0515 02:28:45.383184 23147 sgd_solver.cpp:106] Iteration 89700, lr = 3.56458e-05
    I0515 02:28:53.520992 23147 solver.cpp:228] Iteration 89800, loss = 0.0249436
    I0515 02:28:53.521039 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:28:53.521059 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0249436 (* 1 = 0.0249436 loss)
    I0515 02:28:53.521072 23147 sgd_solver.cpp:106] Iteration 89800, lr = 3.5619e-05
    I0515 02:29:01.661097 23147 solver.cpp:228] Iteration 89900, loss = 0.0114812
    I0515 02:29:01.661141 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:29:01.661161 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0114812 (* 1 = 0.0114812 loss)
    I0515 02:29:01.661176 23147 sgd_solver.cpp:106] Iteration 89900, lr = 3.55923e-05
    I0515 02:29:09.714238 23147 solver.cpp:337] Iteration 90000, Testing net (#0)
    I0515 02:29:14.128660 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.831333
    I0515 02:29:14.128712 23147 solver.cpp:404]     Test net output #1: loss_c = 0.536387 (* 1 = 0.536387 loss)
    I0515 02:29:14.184080 23147 solver.cpp:228] Iteration 90000, loss = 0.00982943
    I0515 02:29:14.184146 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:29:14.184177 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00982944 (* 1 = 0.00982944 loss)
    I0515 02:29:14.184204 23147 sgd_solver.cpp:106] Iteration 90000, lr = 3.55656e-05
    I0515 02:29:22.323632 23147 solver.cpp:228] Iteration 90100, loss = 0.0174435
    I0515 02:29:22.323681 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:29:22.323701 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0174435 (* 1 = 0.0174435 loss)
    I0515 02:29:22.323715 23147 sgd_solver.cpp:106] Iteration 90100, lr = 3.55389e-05
    I0515 02:29:30.461757 23147 solver.cpp:228] Iteration 90200, loss = 0.0214476
    I0515 02:29:30.461804 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:29:30.461825 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0214476 (* 1 = 0.0214476 loss)
    I0515 02:29:30.461840 23147 sgd_solver.cpp:106] Iteration 90200, lr = 3.55123e-05
    I0515 02:29:38.597702 23147 solver.cpp:228] Iteration 90300, loss = 0.0150003
    I0515 02:29:38.597754 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:29:38.597775 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0150003 (* 1 = 0.0150003 loss)
    I0515 02:29:38.597790 23147 sgd_solver.cpp:106] Iteration 90300, lr = 3.54858e-05
    I0515 02:29:46.740200 23147 solver.cpp:228] Iteration 90400, loss = 0.0125096
    I0515 02:29:46.740440 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:29:46.740486 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0125096 (* 1 = 0.0125096 loss)
    I0515 02:29:46.740511 23147 sgd_solver.cpp:106] Iteration 90400, lr = 3.54593e-05
    I0515 02:29:54.861315 23147 solver.cpp:228] Iteration 90500, loss = 0.0153347
    I0515 02:29:54.861366 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:29:54.861387 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0153347 (* 1 = 0.0153347 loss)
    I0515 02:29:54.861402 23147 sgd_solver.cpp:106] Iteration 90500, lr = 3.54328e-05
    I0515 02:30:02.996901 23147 solver.cpp:228] Iteration 90600, loss = 0.0365696
    I0515 02:30:02.996951 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:30:02.996971 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0365696 (* 1 = 0.0365696 loss)
    I0515 02:30:02.996986 23147 sgd_solver.cpp:106] Iteration 90600, lr = 3.54064e-05
    I0515 02:30:11.103705 23147 solver.cpp:228] Iteration 90700, loss = 0.00896104
    I0515 02:30:11.103765 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:30:11.103793 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00896104 (* 1 = 0.00896104 loss)
    I0515 02:30:11.103814 23147 sgd_solver.cpp:106] Iteration 90700, lr = 3.538e-05
    I0515 02:30:19.201135 23147 solver.cpp:228] Iteration 90800, loss = 0.010389
    I0515 02:30:19.201215 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:30:19.201236 23147 solver.cpp:244]     Train net output #1: loss_c = 0.010389 (* 1 = 0.010389 loss)
    I0515 02:30:19.201252 23147 sgd_solver.cpp:106] Iteration 90800, lr = 3.53537e-05
    I0515 02:30:27.311745 23147 solver.cpp:228] Iteration 90900, loss = 0.0240416
    I0515 02:30:27.311794 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:30:27.311813 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0240415 (* 1 = 0.0240415 loss)
    I0515 02:30:27.311828 23147 sgd_solver.cpp:106] Iteration 90900, lr = 3.53274e-05
    I0515 02:30:35.371913 23147 solver.cpp:337] Iteration 91000, Testing net (#0)
    I0515 02:30:39.765630 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.814333
    I0515 02:30:39.765677 23147 solver.cpp:404]     Test net output #1: loss_c = 0.527802 (* 1 = 0.527802 loss)
    I0515 02:30:39.817893 23147 solver.cpp:228] Iteration 91000, loss = 0.0276416
    I0515 02:30:39.817929 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:30:39.817950 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0276416 (* 1 = 0.0276416 loss)
    I0515 02:30:39.817970 23147 sgd_solver.cpp:106] Iteration 91000, lr = 3.53012e-05
    I0515 02:30:47.958750 23147 solver.cpp:228] Iteration 91100, loss = 0.0152572
    I0515 02:30:47.958802 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:30:47.958822 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0152572 (* 1 = 0.0152572 loss)
    I0515 02:30:47.958837 23147 sgd_solver.cpp:106] Iteration 91100, lr = 3.5275e-05
    I0515 02:30:56.098111 23147 solver.cpp:228] Iteration 91200, loss = 0.0085173
    I0515 02:30:56.098291 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:30:56.098312 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00851729 (* 1 = 0.00851729 loss)
    I0515 02:30:56.098327 23147 sgd_solver.cpp:106] Iteration 91200, lr = 3.52488e-05
    I0515 02:31:04.235986 23147 solver.cpp:228] Iteration 91300, loss = 0.0164611
    I0515 02:31:04.236034 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:31:04.236057 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0164611 (* 1 = 0.0164611 loss)
    I0515 02:31:04.236073 23147 sgd_solver.cpp:106] Iteration 91300, lr = 3.52227e-05
    I0515 02:31:12.358767 23147 solver.cpp:228] Iteration 91400, loss = 0.0475765
    I0515 02:31:12.358809 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:31:12.358829 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0475765 (* 1 = 0.0475765 loss)
    I0515 02:31:12.358844 23147 sgd_solver.cpp:106] Iteration 91400, lr = 3.51967e-05
    I0515 02:31:20.499557 23147 solver.cpp:228] Iteration 91500, loss = 0.0240488
    I0515 02:31:20.499603 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:31:20.499622 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0240488 (* 1 = 0.0240488 loss)
    I0515 02:31:20.499636 23147 sgd_solver.cpp:106] Iteration 91500, lr = 3.51707e-05
    I0515 02:31:28.624299 23147 solver.cpp:228] Iteration 91600, loss = 0.0395706
    I0515 02:31:28.624382 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:31:28.624402 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0395706 (* 1 = 0.0395706 loss)
    I0515 02:31:28.624416 23147 sgd_solver.cpp:106] Iteration 91600, lr = 3.51447e-05
    I0515 02:31:36.751111 23147 solver.cpp:228] Iteration 91700, loss = 0.0343085
    I0515 02:31:36.751152 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:31:36.751171 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0343085 (* 1 = 0.0343085 loss)
    I0515 02:31:36.751186 23147 sgd_solver.cpp:106] Iteration 91700, lr = 3.51188e-05
    I0515 02:31:44.891185 23147 solver.cpp:228] Iteration 91800, loss = 0.0248646
    I0515 02:31:44.891234 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:31:44.891258 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0248646 (* 1 = 0.0248646 loss)
    I0515 02:31:44.891273 23147 sgd_solver.cpp:106] Iteration 91800, lr = 3.50929e-05
    I0515 02:31:53.024046 23147 solver.cpp:228] Iteration 91900, loss = 0.0208518
    I0515 02:31:53.024097 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:31:53.024117 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0208518 (* 1 = 0.0208518 loss)
    I0515 02:31:53.024132 23147 sgd_solver.cpp:106] Iteration 91900, lr = 3.50671e-05
    I0515 02:32:01.073326 23147 solver.cpp:337] Iteration 92000, Testing net (#0)
    I0515 02:32:05.466359 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.830333
    I0515 02:32:05.466408 23147 solver.cpp:404]     Test net output #1: loss_c = 0.517493 (* 1 = 0.517493 loss)
    I0515 02:32:05.522436 23147 solver.cpp:228] Iteration 92000, loss = 0.0133104
    I0515 02:32:05.522498 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:32:05.522528 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0133104 (* 1 = 0.0133104 loss)
    I0515 02:32:05.522553 23147 sgd_solver.cpp:106] Iteration 92000, lr = 3.50413e-05
    I0515 02:32:13.661428 23147 solver.cpp:228] Iteration 92100, loss = 0.0476603
    I0515 02:32:13.661480 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:32:13.661500 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0476603 (* 1 = 0.0476603 loss)
    I0515 02:32:13.661515 23147 sgd_solver.cpp:106] Iteration 92100, lr = 3.50155e-05
    I0515 02:32:21.800315 23147 solver.cpp:228] Iteration 92200, loss = 0.0116953
    I0515 02:32:21.800366 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:32:21.800387 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0116953 (* 1 = 0.0116953 loss)
    I0515 02:32:21.800402 23147 sgd_solver.cpp:106] Iteration 92200, lr = 3.49898e-05
    I0515 02:32:29.938243 23147 solver.cpp:228] Iteration 92300, loss = 0.0286314
    I0515 02:32:29.938295 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:32:29.938316 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0286314 (* 1 = 0.0286314 loss)
    I0515 02:32:29.938330 23147 sgd_solver.cpp:106] Iteration 92300, lr = 3.49642e-05
    I0515 02:32:38.074656 23147 solver.cpp:228] Iteration 92400, loss = 0.0141271
    I0515 02:32:38.074882 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:32:38.074929 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0141271 (* 1 = 0.0141271 loss)
    I0515 02:32:38.074955 23147 sgd_solver.cpp:106] Iteration 92400, lr = 3.49386e-05
    I0515 02:32:46.211195 23147 solver.cpp:228] Iteration 92500, loss = 0.0142371
    I0515 02:32:46.211242 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:32:46.211262 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0142371 (* 1 = 0.0142371 loss)
    I0515 02:32:46.211277 23147 sgd_solver.cpp:106] Iteration 92500, lr = 3.4913e-05
    I0515 02:32:54.349856 23147 solver.cpp:228] Iteration 92600, loss = 0.011525
    I0515 02:32:54.349903 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:32:54.349923 23147 solver.cpp:244]     Train net output #1: loss_c = 0.011525 (* 1 = 0.011525 loss)
    I0515 02:32:54.349937 23147 sgd_solver.cpp:106] Iteration 92600, lr = 3.48875e-05
    I0515 02:33:02.487777 23147 solver.cpp:228] Iteration 92700, loss = 0.0227698
    I0515 02:33:02.487823 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:33:02.487843 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0227698 (* 1 = 0.0227698 loss)
    I0515 02:33:02.487856 23147 sgd_solver.cpp:106] Iteration 92700, lr = 3.4862e-05
    I0515 02:33:10.618398 23147 solver.cpp:228] Iteration 92800, loss = 0.00853515
    I0515 02:33:10.618553 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:33:10.618599 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00853515 (* 1 = 0.00853515 loss)
    I0515 02:33:10.618624 23147 sgd_solver.cpp:106] Iteration 92800, lr = 3.48366e-05
    I0515 02:33:18.759738 23147 solver.cpp:228] Iteration 92900, loss = 0.0151031
    I0515 02:33:18.759788 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:33:18.759809 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0151031 (* 1 = 0.0151031 loss)
    I0515 02:33:18.759822 23147 sgd_solver.cpp:106] Iteration 92900, lr = 3.48112e-05
    I0515 02:33:26.820894 23147 solver.cpp:337] Iteration 93000, Testing net (#0)
    I0515 02:33:31.252882 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.822083
    I0515 02:33:31.252933 23147 solver.cpp:404]     Test net output #1: loss_c = 0.537466 (* 1 = 0.537466 loss)
    I0515 02:33:31.308375 23147 solver.cpp:228] Iteration 93000, loss = 0.00636567
    I0515 02:33:31.308434 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:33:31.308464 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00636569 (* 1 = 0.00636569 loss)
    I0515 02:33:31.308487 23147 sgd_solver.cpp:106] Iteration 93000, lr = 3.47858e-05
    I0515 02:33:39.418207 23147 solver.cpp:228] Iteration 93100, loss = 0.0244248
    I0515 02:33:39.418261 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:33:39.418289 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0244248 (* 1 = 0.0244248 loss)
    I0515 02:33:39.418310 23147 sgd_solver.cpp:106] Iteration 93100, lr = 3.47605e-05
    I0515 02:33:47.504732 23147 solver.cpp:228] Iteration 93200, loss = 0.0156709
    I0515 02:33:47.504869 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:33:47.504914 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0156709 (* 1 = 0.0156709 loss)
    I0515 02:33:47.504940 23147 sgd_solver.cpp:106] Iteration 93200, lr = 3.47352e-05
    I0515 02:33:55.637840 23147 solver.cpp:228] Iteration 93300, loss = 0.0443881
    I0515 02:33:55.637890 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:33:55.637910 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0443881 (* 1 = 0.0443881 loss)
    I0515 02:33:55.637925 23147 sgd_solver.cpp:106] Iteration 93300, lr = 3.471e-05
    I0515 02:34:03.778064 23147 solver.cpp:228] Iteration 93400, loss = 0.0265234
    I0515 02:34:03.778111 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:34:03.778132 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0265234 (* 1 = 0.0265234 loss)
    I0515 02:34:03.778147 23147 sgd_solver.cpp:106] Iteration 93400, lr = 3.46848e-05
    I0515 02:34:11.901734 23147 solver.cpp:228] Iteration 93500, loss = 0.0218626
    I0515 02:34:11.901795 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:34:11.901825 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0218626 (* 1 = 0.0218626 loss)
    I0515 02:34:11.901847 23147 sgd_solver.cpp:106] Iteration 93500, lr = 3.46597e-05
    I0515 02:34:19.987690 23147 solver.cpp:228] Iteration 93600, loss = 0.0334887
    I0515 02:34:19.987921 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:34:19.987970 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0334888 (* 1 = 0.0334888 loss)
    I0515 02:34:19.987994 23147 sgd_solver.cpp:106] Iteration 93600, lr = 3.46346e-05
    I0515 02:34:28.123692 23147 solver.cpp:228] Iteration 93700, loss = 0.0135078
    I0515 02:34:28.123744 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:34:28.123764 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0135078 (* 1 = 0.0135078 loss)
    I0515 02:34:28.123778 23147 sgd_solver.cpp:106] Iteration 93700, lr = 3.46095e-05
    I0515 02:34:36.264000 23147 solver.cpp:228] Iteration 93800, loss = 0.0141454
    I0515 02:34:36.264050 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:34:36.264071 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0141454 (* 1 = 0.0141454 loss)
    I0515 02:34:36.264086 23147 sgd_solver.cpp:106] Iteration 93800, lr = 3.45845e-05
    I0515 02:34:44.404952 23147 solver.cpp:228] Iteration 93900, loss = 0.0103584
    I0515 02:34:44.404996 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:34:44.405020 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0103584 (* 1 = 0.0103584 loss)
    I0515 02:34:44.405035 23147 sgd_solver.cpp:106] Iteration 93900, lr = 3.45596e-05
    I0515 02:34:52.457512 23147 solver.cpp:337] Iteration 94000, Testing net (#0)
    I0515 02:34:56.833411 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.83025
    I0515 02:34:56.833478 23147 solver.cpp:404]     Test net output #1: loss_c = 0.530665 (* 1 = 0.530665 loss)
    I0515 02:34:56.888831 23147 solver.cpp:228] Iteration 94000, loss = 0.0186753
    I0515 02:34:56.888911 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:34:56.888939 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0186753 (* 1 = 0.0186753 loss)
    I0515 02:34:56.888964 23147 sgd_solver.cpp:106] Iteration 94000, lr = 3.45346e-05
    I0515 02:35:04.984228 23147 solver.cpp:228] Iteration 94100, loss = 0.0128703
    I0515 02:35:04.984287 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:35:04.984316 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0128703 (* 1 = 0.0128703 loss)
    I0515 02:35:04.984338 23147 sgd_solver.cpp:106] Iteration 94100, lr = 3.45098e-05
    I0515 02:35:13.076589 23147 solver.cpp:228] Iteration 94200, loss = 0.0100596
    I0515 02:35:13.076645 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:35:13.076674 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0100596 (* 1 = 0.0100596 loss)
    I0515 02:35:13.076695 23147 sgd_solver.cpp:106] Iteration 94200, lr = 3.44849e-05
    I0515 02:35:21.168990 23147 solver.cpp:228] Iteration 94300, loss = 0.016726
    I0515 02:35:21.169049 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:35:21.169080 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0167261 (* 1 = 0.0167261 loss)
    I0515 02:35:21.169100 23147 sgd_solver.cpp:106] Iteration 94300, lr = 3.44601e-05
    I0515 02:35:29.253943 23147 solver.cpp:228] Iteration 94400, loss = 0.0215389
    I0515 02:35:29.254163 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:35:29.254194 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0215389 (* 1 = 0.0215389 loss)
    I0515 02:35:29.254237 23147 sgd_solver.cpp:106] Iteration 94400, lr = 3.44354e-05
    I0515 02:35:37.341212 23147 solver.cpp:228] Iteration 94500, loss = 0.00585638
    I0515 02:35:37.341272 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:35:37.341302 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00585641 (* 1 = 0.00585641 loss)
    I0515 02:35:37.341325 23147 sgd_solver.cpp:106] Iteration 94500, lr = 3.44106e-05
    I0515 02:35:45.428387 23147 solver.cpp:228] Iteration 94600, loss = 0.0145615
    I0515 02:35:45.428449 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:35:45.428479 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0145615 (* 1 = 0.0145615 loss)
    I0515 02:35:45.428500 23147 sgd_solver.cpp:106] Iteration 94600, lr = 3.4386e-05
    I0515 02:35:53.521508 23147 solver.cpp:228] Iteration 94700, loss = 0.0711916
    I0515 02:35:53.521565 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 02:35:53.521595 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0711916 (* 1 = 0.0711916 loss)
    I0515 02:35:53.521615 23147 sgd_solver.cpp:106] Iteration 94700, lr = 3.43613e-05
    I0515 02:36:01.616675 23147 solver.cpp:228] Iteration 94800, loss = 0.0136231
    I0515 02:36:01.616794 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:36:01.616824 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0136231 (* 1 = 0.0136231 loss)
    I0515 02:36:01.616845 23147 sgd_solver.cpp:106] Iteration 94800, lr = 3.43367e-05
    I0515 02:36:09.702431 23147 solver.cpp:228] Iteration 94900, loss = 0.0172051
    I0515 02:36:09.702489 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:36:09.702518 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0172051 (* 1 = 0.0172051 loss)
    I0515 02:36:09.702540 23147 sgd_solver.cpp:106] Iteration 94900, lr = 3.43122e-05
    I0515 02:36:17.712218 23147 solver.cpp:337] Iteration 95000, Testing net (#0)
    I0515 02:36:22.070546 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.81675
    I0515 02:36:22.070632 23147 solver.cpp:404]     Test net output #1: loss_c = 0.555885 (* 1 = 0.555885 loss)
    I0515 02:36:22.127018 23147 solver.cpp:228] Iteration 95000, loss = 0.0124889
    I0515 02:36:22.127112 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:36:22.127146 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0124889 (* 1 = 0.0124889 loss)
    I0515 02:36:22.127174 23147 sgd_solver.cpp:106] Iteration 95000, lr = 3.42877e-05
    I0515 02:36:30.212623 23147 solver.cpp:228] Iteration 95100, loss = 0.0107989
    I0515 02:36:30.212683 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:36:30.212713 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0107989 (* 1 = 0.0107989 loss)
    I0515 02:36:30.212734 23147 sgd_solver.cpp:106] Iteration 95100, lr = 3.42632e-05
    I0515 02:36:38.300832 23147 solver.cpp:228] Iteration 95200, loss = 0.0208873
    I0515 02:36:38.300974 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:36:38.301019 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0208873 (* 1 = 0.0208873 loss)
    I0515 02:36:38.301044 23147 sgd_solver.cpp:106] Iteration 95200, lr = 3.42388e-05
    I0515 02:36:46.440732 23147 solver.cpp:228] Iteration 95300, loss = 0.0173421
    I0515 02:36:46.440773 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:36:46.440793 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0173421 (* 1 = 0.0173421 loss)
    I0515 02:36:46.440809 23147 sgd_solver.cpp:106] Iteration 95300, lr = 3.42144e-05
    I0515 02:36:54.577620 23147 solver.cpp:228] Iteration 95400, loss = 0.00878808
    I0515 02:36:54.577666 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:36:54.577685 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00878809 (* 1 = 0.00878809 loss)
    I0515 02:36:54.577700 23147 sgd_solver.cpp:106] Iteration 95400, lr = 3.419e-05
    I0515 02:37:02.717877 23147 solver.cpp:228] Iteration 95500, loss = 0.0476435
    I0515 02:37:02.717921 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:37:02.717939 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0476435 (* 1 = 0.0476435 loss)
    I0515 02:37:02.717954 23147 sgd_solver.cpp:106] Iteration 95500, lr = 3.41657e-05
    I0515 02:37:10.855164 23147 solver.cpp:228] Iteration 95600, loss = 0.0305892
    I0515 02:37:10.855357 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:37:10.855377 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0305892 (* 1 = 0.0305892 loss)
    I0515 02:37:10.855392 23147 sgd_solver.cpp:106] Iteration 95600, lr = 3.41415e-05
    I0515 02:37:18.996462 23147 solver.cpp:228] Iteration 95700, loss = 0.0212911
    I0515 02:37:18.996505 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:37:18.996525 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0212911 (* 1 = 0.0212911 loss)
    I0515 02:37:18.996538 23147 sgd_solver.cpp:106] Iteration 95700, lr = 3.41172e-05
    I0515 02:37:27.135731 23147 solver.cpp:228] Iteration 95800, loss = 0.0193432
    I0515 02:37:27.135774 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:37:27.135794 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0193432 (* 1 = 0.0193432 loss)
    I0515 02:37:27.135808 23147 sgd_solver.cpp:106] Iteration 95800, lr = 3.4093e-05
    I0515 02:37:35.276983 23147 solver.cpp:228] Iteration 95900, loss = 0.0254706
    I0515 02:37:35.277029 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:37:35.277048 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0254706 (* 1 = 0.0254706 loss)
    I0515 02:37:35.277063 23147 sgd_solver.cpp:106] Iteration 95900, lr = 3.40689e-05
    I0515 02:37:43.338311 23147 solver.cpp:337] Iteration 96000, Testing net (#0)
    I0515 02:37:47.741291 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.82125
    I0515 02:37:47.741340 23147 solver.cpp:404]     Test net output #1: loss_c = 0.544575 (* 1 = 0.544575 loss)
    I0515 02:37:47.792691 23147 solver.cpp:228] Iteration 96000, loss = 0.0348082
    I0515 02:37:47.792728 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:37:47.792747 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0348082 (* 1 = 0.0348082 loss)
    I0515 02:37:47.792767 23147 sgd_solver.cpp:106] Iteration 96000, lr = 3.40448e-05
    I0515 02:37:55.920488 23147 solver.cpp:228] Iteration 96100, loss = 0.00869356
    I0515 02:37:55.920536 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:37:55.920554 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00869356 (* 1 = 0.00869356 loss)
    I0515 02:37:55.920569 23147 sgd_solver.cpp:106] Iteration 96100, lr = 3.40207e-05
    I0515 02:38:04.060614 23147 solver.cpp:228] Iteration 96200, loss = 0.0224453
    I0515 02:38:04.060667 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:38:04.060688 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0224453 (* 1 = 0.0224453 loss)
    I0515 02:38:04.060703 23147 sgd_solver.cpp:106] Iteration 96200, lr = 3.39967e-05
    I0515 02:38:12.198282 23147 solver.cpp:228] Iteration 96300, loss = 0.0388605
    I0515 02:38:12.198333 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:38:12.198354 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0388605 (* 1 = 0.0388605 loss)
    I0515 02:38:12.198369 23147 sgd_solver.cpp:106] Iteration 96300, lr = 3.39727e-05
    I0515 02:38:20.337398 23147 solver.cpp:228] Iteration 96400, loss = 0.00729128
    I0515 02:38:20.337476 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:38:20.337499 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00729127 (* 1 = 0.00729127 loss)
    I0515 02:38:20.337513 23147 sgd_solver.cpp:106] Iteration 96400, lr = 3.39487e-05
    I0515 02:38:28.471387 23147 solver.cpp:228] Iteration 96500, loss = 0.0314819
    I0515 02:38:28.471439 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:38:28.471460 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0314819 (* 1 = 0.0314819 loss)
    I0515 02:38:28.471474 23147 sgd_solver.cpp:106] Iteration 96500, lr = 3.39248e-05
    I0515 02:38:36.563323 23147 solver.cpp:228] Iteration 96600, loss = 0.0360288
    I0515 02:38:36.563383 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:38:36.563413 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0360288 (* 1 = 0.0360288 loss)
    I0515 02:38:36.563436 23147 sgd_solver.cpp:106] Iteration 96600, lr = 3.3901e-05
    I0515 02:38:44.654165 23147 solver.cpp:228] Iteration 96700, loss = 0.0198728
    I0515 02:38:44.654225 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:38:44.654255 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0198728 (* 1 = 0.0198728 loss)
    I0515 02:38:44.654276 23147 sgd_solver.cpp:106] Iteration 96700, lr = 3.38771e-05
    I0515 02:38:52.743036 23147 solver.cpp:228] Iteration 96800, loss = 0.0237911
    I0515 02:38:52.743288 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:38:52.743332 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0237911 (* 1 = 0.0237911 loss)
    I0515 02:38:52.743356 23147 sgd_solver.cpp:106] Iteration 96800, lr = 3.38533e-05
    I0515 02:39:00.891330 23147 solver.cpp:228] Iteration 96900, loss = 0.0153803
    I0515 02:39:00.891379 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:39:00.891402 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0153803 (* 1 = 0.0153803 loss)
    I0515 02:39:00.891417 23147 sgd_solver.cpp:106] Iteration 96900, lr = 3.38296e-05
    I0515 02:39:08.945180 23147 solver.cpp:337] Iteration 97000, Testing net (#0)
    I0515 02:39:13.286459 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.8305
    I0515 02:39:13.286511 23147 solver.cpp:404]     Test net output #1: loss_c = 0.521749 (* 1 = 0.521749 loss)
    I0515 02:39:13.338742 23147 solver.cpp:228] Iteration 97000, loss = 0.0413666
    I0515 02:39:13.338779 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 02:39:13.338798 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0413666 (* 1 = 0.0413666 loss)
    I0515 02:39:13.338817 23147 sgd_solver.cpp:106] Iteration 97000, lr = 3.38059e-05
    I0515 02:39:21.414528 23147 solver.cpp:228] Iteration 97100, loss = 0.0173233
    I0515 02:39:21.414582 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:39:21.414607 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0173233 (* 1 = 0.0173233 loss)
    I0515 02:39:21.414621 23147 sgd_solver.cpp:106] Iteration 97100, lr = 3.37822e-05
    I0515 02:39:29.535300 23147 solver.cpp:228] Iteration 97200, loss = 0.0130147
    I0515 02:39:29.535584 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:39:29.535632 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0130147 (* 1 = 0.0130147 loss)
    I0515 02:39:29.535657 23147 sgd_solver.cpp:106] Iteration 97200, lr = 3.37586e-05
    I0515 02:39:37.679559 23147 solver.cpp:228] Iteration 97300, loss = 0.0188377
    I0515 02:39:37.679611 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:39:37.679631 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0188376 (* 1 = 0.0188376 loss)
    I0515 02:39:37.679646 23147 sgd_solver.cpp:106] Iteration 97300, lr = 3.3735e-05
    I0515 02:39:45.815349 23147 solver.cpp:228] Iteration 97400, loss = 0.0174859
    I0515 02:39:45.815397 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:39:45.815418 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0174859 (* 1 = 0.0174859 loss)
    I0515 02:39:45.815433 23147 sgd_solver.cpp:106] Iteration 97400, lr = 3.37114e-05
    I0515 02:39:53.955770 23147 solver.cpp:228] Iteration 97500, loss = 0.0133316
    I0515 02:39:53.955821 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:39:53.955840 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0133316 (* 1 = 0.0133316 loss)
    I0515 02:39:53.955855 23147 sgd_solver.cpp:106] Iteration 97500, lr = 3.36879e-05
    I0515 02:40:02.094171 23147 solver.cpp:228] Iteration 97600, loss = 0.0118529
    I0515 02:40:02.094393 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:40:02.094426 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0118528 (* 1 = 0.0118528 loss)
    I0515 02:40:02.094449 23147 sgd_solver.cpp:106] Iteration 97600, lr = 3.36644e-05
    I0515 02:40:10.236194 23147 solver.cpp:228] Iteration 97700, loss = 0.00638921
    I0515 02:40:10.236248 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:40:10.236268 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00638919 (* 1 = 0.00638919 loss)
    I0515 02:40:10.236282 23147 sgd_solver.cpp:106] Iteration 97700, lr = 3.36409e-05
    I0515 02:40:18.371969 23147 solver.cpp:228] Iteration 97800, loss = 0.0381831
    I0515 02:40:18.372020 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:40:18.372041 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0381831 (* 1 = 0.0381831 loss)
    I0515 02:40:18.372056 23147 sgd_solver.cpp:106] Iteration 97800, lr = 3.36175e-05
    I0515 02:40:26.490128 23147 solver.cpp:228] Iteration 97900, loss = 0.0186128
    I0515 02:40:26.490187 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:40:26.490216 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0186128 (* 1 = 0.0186128 loss)
    I0515 02:40:26.490238 23147 sgd_solver.cpp:106] Iteration 97900, lr = 3.35942e-05
    I0515 02:40:34.497143 23147 solver.cpp:337] Iteration 98000, Testing net (#0)
    I0515 02:40:38.904675 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.832417
    I0515 02:40:38.904731 23147 solver.cpp:404]     Test net output #1: loss_c = 0.540457 (* 1 = 0.540457 loss)
    I0515 02:40:38.959856 23147 solver.cpp:228] Iteration 98000, loss = 0.0463711
    I0515 02:40:38.959913 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:40:38.959944 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0463711 (* 1 = 0.0463711 loss)
    I0515 02:40:38.959969 23147 sgd_solver.cpp:106] Iteration 98000, lr = 3.35708e-05
    I0515 02:40:47.047746 23147 solver.cpp:228] Iteration 98100, loss = 0.0137548
    I0515 02:40:47.047797 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:40:47.047827 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0137548 (* 1 = 0.0137548 loss)
    I0515 02:40:47.047850 23147 sgd_solver.cpp:106] Iteration 98100, lr = 3.35475e-05
    I0515 02:40:55.135169 23147 solver.cpp:228] Iteration 98200, loss = 0.0178463
    I0515 02:40:55.135226 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:40:55.135257 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0178463 (* 1 = 0.0178463 loss)
    I0515 02:40:55.135277 23147 sgd_solver.cpp:106] Iteration 98200, lr = 3.35243e-05
    I0515 02:41:03.224784 23147 solver.cpp:228] Iteration 98300, loss = 0.0147597
    I0515 02:41:03.224833 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:41:03.224863 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0147596 (* 1 = 0.0147596 loss)
    I0515 02:41:03.224884 23147 sgd_solver.cpp:106] Iteration 98300, lr = 3.35011e-05
    I0515 02:41:11.311790 23147 solver.cpp:228] Iteration 98400, loss = 0.0131789
    I0515 02:41:11.311939 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:41:11.311985 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0131788 (* 1 = 0.0131788 loss)
    I0515 02:41:11.312010 23147 sgd_solver.cpp:106] Iteration 98400, lr = 3.34779e-05
    I0515 02:41:19.435654 23147 solver.cpp:228] Iteration 98500, loss = 0.0295409
    I0515 02:41:19.435700 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:41:19.435721 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0295409 (* 1 = 0.0295409 loss)
    I0515 02:41:19.435735 23147 sgd_solver.cpp:106] Iteration 98500, lr = 3.34547e-05
    I0515 02:41:27.573912 23147 solver.cpp:228] Iteration 98600, loss = 0.0350623
    I0515 02:41:27.573962 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:41:27.573982 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0350623 (* 1 = 0.0350623 loss)
    I0515 02:41:27.573997 23147 sgd_solver.cpp:106] Iteration 98600, lr = 3.34316e-05
    I0515 02:41:35.711838 23147 solver.cpp:228] Iteration 98700, loss = 0.00937248
    I0515 02:41:35.711887 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:41:35.711908 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00937246 (* 1 = 0.00937246 loss)
    I0515 02:41:35.711922 23147 sgd_solver.cpp:106] Iteration 98700, lr = 3.34086e-05
    I0515 02:41:43.853579 23147 solver.cpp:228] Iteration 98800, loss = 0.0191187
    I0515 02:41:43.853828 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:41:43.853873 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0191187 (* 1 = 0.0191187 loss)
    I0515 02:41:43.853899 23147 sgd_solver.cpp:106] Iteration 98800, lr = 3.33855e-05
    I0515 02:41:51.966568 23147 solver.cpp:228] Iteration 98900, loss = 0.0317627
    I0515 02:41:51.966619 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:41:51.966639 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0317627 (* 1 = 0.0317627 loss)
    I0515 02:41:51.966653 23147 sgd_solver.cpp:106] Iteration 98900, lr = 3.33625e-05
    I0515 02:42:00.027782 23147 solver.cpp:337] Iteration 99000, Testing net (#0)
    I0515 02:42:04.436748 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.82525
    I0515 02:42:04.436807 23147 solver.cpp:404]     Test net output #1: loss_c = 0.54374 (* 1 = 0.54374 loss)
    I0515 02:42:04.492041 23147 solver.cpp:228] Iteration 99000, loss = 0.0131331
    I0515 02:42:04.492106 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:42:04.492136 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0131331 (* 1 = 0.0131331 loss)
    I0515 02:42:04.492162 23147 sgd_solver.cpp:106] Iteration 99000, lr = 3.33396e-05
    I0515 02:42:12.619792 23147 solver.cpp:228] Iteration 99100, loss = 0.030012
    I0515 02:42:12.619843 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:42:12.619868 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0300119 (* 1 = 0.0300119 loss)
    I0515 02:42:12.619882 23147 sgd_solver.cpp:106] Iteration 99100, lr = 3.33167e-05
    I0515 02:42:20.762187 23147 solver.cpp:228] Iteration 99200, loss = 0.0146314
    I0515 02:42:20.762333 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:42:20.762378 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0146314 (* 1 = 0.0146314 loss)
    I0515 02:42:20.762403 23147 sgd_solver.cpp:106] Iteration 99200, lr = 3.32938e-05
    I0515 02:42:28.894722 23147 solver.cpp:228] Iteration 99300, loss = 0.0219676
    I0515 02:42:28.894763 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:42:28.894783 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0219676 (* 1 = 0.0219676 loss)
    I0515 02:42:28.894798 23147 sgd_solver.cpp:106] Iteration 99300, lr = 3.32709e-05
    I0515 02:42:37.033221 23147 solver.cpp:228] Iteration 99400, loss = 0.0319352
    I0515 02:42:37.033273 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:42:37.033293 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0319351 (* 1 = 0.0319351 loss)
    I0515 02:42:37.033308 23147 sgd_solver.cpp:106] Iteration 99400, lr = 3.32481e-05
    I0515 02:42:45.171681 23147 solver.cpp:228] Iteration 99500, loss = 0.0138647
    I0515 02:42:45.171731 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:42:45.171753 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0138647 (* 1 = 0.0138647 loss)
    I0515 02:42:45.171768 23147 sgd_solver.cpp:106] Iteration 99500, lr = 3.32253e-05
    I0515 02:42:53.310325 23147 solver.cpp:228] Iteration 99600, loss = 0.0201515
    I0515 02:42:53.310420 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:42:53.310441 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0201515 (* 1 = 0.0201515 loss)
    I0515 02:42:53.310456 23147 sgd_solver.cpp:106] Iteration 99600, lr = 3.32026e-05
    I0515 02:43:01.430932 23147 solver.cpp:228] Iteration 99700, loss = 0.019378
    I0515 02:43:01.430985 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:43:01.431005 23147 solver.cpp:244]     Train net output #1: loss_c = 0.019378 (* 1 = 0.019378 loss)
    I0515 02:43:01.431020 23147 sgd_solver.cpp:106] Iteration 99700, lr = 3.31799e-05
    I0515 02:43:09.570130 23147 solver.cpp:228] Iteration 99800, loss = 0.0147584
    I0515 02:43:09.570176 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:43:09.570196 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0147583 (* 1 = 0.0147583 loss)
    I0515 02:43:09.570211 23147 sgd_solver.cpp:106] Iteration 99800, lr = 3.31572e-05
    I0515 02:43:17.699565 23147 solver.cpp:228] Iteration 99900, loss = 0.0412929
    I0515 02:43:17.699610 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:43:17.699630 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0412929 (* 1 = 0.0412929 loss)
    I0515 02:43:17.699646 23147 sgd_solver.cpp:106] Iteration 99900, lr = 3.31346e-05
    I0515 02:43:25.753518 23147 solver.cpp:454] Snapshotting to binary proto file dvia_train_iter_100000.caffemodel
    I0515 02:43:25.784507 23147 sgd_solver.cpp:273] Snapshotting solver state to binary proto file dvia_train_iter_100000.solverstate
    I0515 02:43:25.785764 23147 solver.cpp:337] Iteration 100000, Testing net (#0)
    I0515 02:43:30.123265 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.823083
    I0515 02:43:30.123324 23147 solver.cpp:404]     Test net output #1: loss_c = 0.549608 (* 1 = 0.549608 loss)
    I0515 02:43:30.174820 23147 solver.cpp:228] Iteration 100000, loss = 0.0058229
    I0515 02:43:30.174893 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:43:30.174916 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00582287 (* 1 = 0.00582287 loss)
    I0515 02:43:30.174935 23147 sgd_solver.cpp:106] Iteration 100000, lr = 3.3112e-05
    I0515 02:43:38.267889 23147 solver.cpp:228] Iteration 100100, loss = 0.0171941
    I0515 02:43:38.267946 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:43:38.267976 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0171941 (* 1 = 0.0171941 loss)
    I0515 02:43:38.267997 23147 sgd_solver.cpp:106] Iteration 100100, lr = 3.30894e-05
    I0515 02:43:46.357456 23147 solver.cpp:228] Iteration 100200, loss = 0.00868941
    I0515 02:43:46.357517 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:43:46.357545 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00868939 (* 1 = 0.00868939 loss)
    I0515 02:43:46.357568 23147 sgd_solver.cpp:106] Iteration 100200, lr = 3.30669e-05
    I0515 02:43:54.449923 23147 solver.cpp:228] Iteration 100300, loss = 0.0197221
    I0515 02:43:54.449981 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:43:54.450012 23147 solver.cpp:244]     Train net output #1: loss_c = 0.019722 (* 1 = 0.019722 loss)
    I0515 02:43:54.450033 23147 sgd_solver.cpp:106] Iteration 100300, lr = 3.30444e-05
    I0515 02:44:02.538595 23147 solver.cpp:228] Iteration 100400, loss = 0.0108998
    I0515 02:44:02.538866 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:44:02.538911 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0108997 (* 1 = 0.0108997 loss)
    I0515 02:44:02.538938 23147 sgd_solver.cpp:106] Iteration 100400, lr = 3.3022e-05
    I0515 02:44:10.631198 23147 solver.cpp:228] Iteration 100500, loss = 0.00845929
    I0515 02:44:10.631250 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:44:10.631279 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00845927 (* 1 = 0.00845927 loss)
    I0515 02:44:10.631300 23147 sgd_solver.cpp:106] Iteration 100500, lr = 3.29996e-05
    I0515 02:44:18.718268 23147 solver.cpp:228] Iteration 100600, loss = 0.0133894
    I0515 02:44:18.718322 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:44:18.718349 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0133894 (* 1 = 0.0133894 loss)
    I0515 02:44:18.718371 23147 sgd_solver.cpp:106] Iteration 100600, lr = 3.29772e-05
    I0515 02:44:26.803689 23147 solver.cpp:228] Iteration 100700, loss = 0.0223366
    I0515 02:44:26.803745 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:44:26.803773 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0223366 (* 1 = 0.0223366 loss)
    I0515 02:44:26.803796 23147 sgd_solver.cpp:106] Iteration 100700, lr = 3.29548e-05
    I0515 02:44:34.887683 23147 solver.cpp:228] Iteration 100800, loss = 0.00706523
    I0515 02:44:34.887881 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:44:34.887928 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0070652 (* 1 = 0.0070652 loss)
    I0515 02:44:34.887951 23147 sgd_solver.cpp:106] Iteration 100800, lr = 3.29325e-05
    I0515 02:44:42.973197 23147 solver.cpp:228] Iteration 100900, loss = 0.00739891
    I0515 02:44:42.973253 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:44:42.973283 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00739889 (* 1 = 0.00739889 loss)
    I0515 02:44:42.973304 23147 sgd_solver.cpp:106] Iteration 100900, lr = 3.29103e-05
    I0515 02:44:50.977762 23147 solver.cpp:337] Iteration 101000, Testing net (#0)
    I0515 02:44:55.339599 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.824583
    I0515 02:44:55.339656 23147 solver.cpp:404]     Test net output #1: loss_c = 0.545721 (* 1 = 0.545721 loss)
    I0515 02:44:55.392482 23147 solver.cpp:228] Iteration 101000, loss = 0.012623
    I0515 02:44:55.392540 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:44:55.392562 23147 solver.cpp:244]     Train net output #1: loss_c = 0.012623 (* 1 = 0.012623 loss)
    I0515 02:44:55.392580 23147 sgd_solver.cpp:106] Iteration 101000, lr = 3.2888e-05
    I0515 02:45:03.487767 23147 solver.cpp:228] Iteration 101100, loss = 0.0264708
    I0515 02:45:03.487821 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:45:03.487854 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0264708 (* 1 = 0.0264708 loss)
    I0515 02:45:03.487876 23147 sgd_solver.cpp:106] Iteration 101100, lr = 3.28658e-05
    I0515 02:45:11.574424 23147 solver.cpp:228] Iteration 101200, loss = 0.049367
    I0515 02:45:11.574548 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:45:11.574578 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0493669 (* 1 = 0.0493669 loss)
    I0515 02:45:11.574599 23147 sgd_solver.cpp:106] Iteration 101200, lr = 3.28436e-05
    I0515 02:45:19.666419 23147 solver.cpp:228] Iteration 101300, loss = 0.01058
    I0515 02:45:19.666479 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:45:19.666508 23147 solver.cpp:244]     Train net output #1: loss_c = 0.01058 (* 1 = 0.01058 loss)
    I0515 02:45:19.666529 23147 sgd_solver.cpp:106] Iteration 101300, lr = 3.28215e-05
    I0515 02:45:27.761761 23147 solver.cpp:228] Iteration 101400, loss = 0.019278
    I0515 02:45:27.761816 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:45:27.761844 23147 solver.cpp:244]     Train net output #1: loss_c = 0.019278 (* 1 = 0.019278 loss)
    I0515 02:45:27.761868 23147 sgd_solver.cpp:106] Iteration 101400, lr = 3.27994e-05
    I0515 02:45:35.850539 23147 solver.cpp:228] Iteration 101500, loss = 0.0908286
    I0515 02:45:35.850590 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:45:35.850620 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0908286 (* 1 = 0.0908286 loss)
    I0515 02:45:35.850642 23147 sgd_solver.cpp:106] Iteration 101500, lr = 3.27774e-05
    I0515 02:45:43.938992 23147 solver.cpp:228] Iteration 101600, loss = 0.0289226
    I0515 02:45:43.939123 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:45:43.939168 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0289226 (* 1 = 0.0289226 loss)
    I0515 02:45:43.939193 23147 sgd_solver.cpp:106] Iteration 101600, lr = 3.27553e-05
    I0515 02:45:52.058063 23147 solver.cpp:228] Iteration 101700, loss = 0.0273584
    I0515 02:45:52.058110 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:45:52.058133 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0273583 (* 1 = 0.0273583 loss)
    I0515 02:45:52.058148 23147 sgd_solver.cpp:106] Iteration 101700, lr = 3.27333e-05
    I0515 02:46:00.194169 23147 solver.cpp:228] Iteration 101800, loss = 0.0400808
    I0515 02:46:00.194214 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:46:00.194234 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0400808 (* 1 = 0.0400808 loss)
    I0515 02:46:00.194249 23147 sgd_solver.cpp:106] Iteration 101800, lr = 3.27114e-05
    I0515 02:46:08.331092 23147 solver.cpp:228] Iteration 101900, loss = 0.00763851
    I0515 02:46:08.331148 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:46:08.331177 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00763846 (* 1 = 0.00763846 loss)
    I0515 02:46:08.331199 23147 sgd_solver.cpp:106] Iteration 101900, lr = 3.26894e-05
    I0515 02:46:16.339237 23147 solver.cpp:337] Iteration 102000, Testing net (#0)
    I0515 02:46:20.753875 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.814083
    I0515 02:46:20.753932 23147 solver.cpp:404]     Test net output #1: loss_c = 0.549429 (* 1 = 0.549429 loss)
    I0515 02:46:20.807381 23147 solver.cpp:228] Iteration 102000, loss = 0.0148439
    I0515 02:46:20.807406 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:46:20.807425 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0148438 (* 1 = 0.0148438 loss)
    I0515 02:46:20.807443 23147 sgd_solver.cpp:106] Iteration 102000, lr = 3.26675e-05
    I0515 02:46:28.941606 23147 solver.cpp:228] Iteration 102100, loss = 0.00754085
    I0515 02:46:28.941658 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:46:28.941679 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0075408 (* 1 = 0.0075408 loss)
    I0515 02:46:28.941694 23147 sgd_solver.cpp:106] Iteration 102100, lr = 3.26457e-05
    I0515 02:46:37.079958 23147 solver.cpp:228] Iteration 102200, loss = 0.0371723
    I0515 02:46:37.080008 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:46:37.080029 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0371723 (* 1 = 0.0371723 loss)
    I0515 02:46:37.080044 23147 sgd_solver.cpp:106] Iteration 102200, lr = 3.26239e-05
    I0515 02:46:45.218121 23147 solver.cpp:228] Iteration 102300, loss = 0.0111932
    I0515 02:46:45.218159 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:46:45.218178 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0111932 (* 1 = 0.0111932 loss)
    I0515 02:46:45.218194 23147 sgd_solver.cpp:106] Iteration 102300, lr = 3.26021e-05
    I0515 02:46:53.358070 23147 solver.cpp:228] Iteration 102400, loss = 0.00654183
    I0515 02:46:53.358222 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:46:53.358266 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00654177 (* 1 = 0.00654177 loss)
    I0515 02:46:53.358291 23147 sgd_solver.cpp:106] Iteration 102400, lr = 3.25803e-05
    I0515 02:47:01.449111 23147 solver.cpp:228] Iteration 102500, loss = 0.0120025
    I0515 02:47:01.449165 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:47:01.449194 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0120025 (* 1 = 0.0120025 loss)
    I0515 02:47:01.449215 23147 sgd_solver.cpp:106] Iteration 102500, lr = 3.25586e-05
    I0515 02:47:09.540194 23147 solver.cpp:228] Iteration 102600, loss = 0.0156897
    I0515 02:47:09.540243 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:47:09.540272 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0156896 (* 1 = 0.0156896 loss)
    I0515 02:47:09.540294 23147 sgd_solver.cpp:106] Iteration 102600, lr = 3.25369e-05
    I0515 02:47:17.629215 23147 solver.cpp:228] Iteration 102700, loss = 0.0112737
    I0515 02:47:17.629273 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:47:17.629302 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0112736 (* 1 = 0.0112736 loss)
    I0515 02:47:17.629323 23147 sgd_solver.cpp:106] Iteration 102700, lr = 3.25152e-05
    I0515 02:47:25.741755 23147 solver.cpp:228] Iteration 102800, loss = 0.0260366
    I0515 02:47:25.741955 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:47:25.741986 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0260365 (* 1 = 0.0260365 loss)
    I0515 02:47:25.742007 23147 sgd_solver.cpp:106] Iteration 102800, lr = 3.24936e-05
    I0515 02:47:33.879232 23147 solver.cpp:228] Iteration 102900, loss = 0.0109599
    I0515 02:47:33.879284 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:47:33.879304 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0109598 (* 1 = 0.0109598 loss)
    I0515 02:47:33.879319 23147 sgd_solver.cpp:106] Iteration 102900, lr = 3.2472e-05
    I0515 02:47:41.931598 23147 solver.cpp:337] Iteration 103000, Testing net (#0)
    I0515 02:47:46.353981 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.82825
    I0515 02:47:46.354039 23147 solver.cpp:404]     Test net output #1: loss_c = 0.545581 (* 1 = 0.545581 loss)
    I0515 02:47:46.406436 23147 solver.cpp:228] Iteration 103000, loss = 0.0146902
    I0515 02:47:46.406496 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:47:46.406517 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0146901 (* 1 = 0.0146901 loss)
    I0515 02:47:46.406538 23147 sgd_solver.cpp:106] Iteration 103000, lr = 3.24505e-05
    I0515 02:47:54.542228 23147 solver.cpp:228] Iteration 103100, loss = 0.015748
    I0515 02:47:54.542279 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:47:54.542299 23147 solver.cpp:244]     Train net output #1: loss_c = 0.015748 (* 1 = 0.015748 loss)
    I0515 02:47:54.542312 23147 sgd_solver.cpp:106] Iteration 103100, lr = 3.2429e-05
    I0515 02:48:02.597301 23147 solver.cpp:228] Iteration 103200, loss = 0.0380103
    I0515 02:48:02.597424 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:48:02.597450 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0380103 (* 1 = 0.0380103 loss)
    I0515 02:48:02.597465 23147 sgd_solver.cpp:106] Iteration 103200, lr = 3.24075e-05
    I0515 02:48:10.651262 23147 solver.cpp:228] Iteration 103300, loss = 0.0171472
    I0515 02:48:10.651317 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:48:10.651343 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0171472 (* 1 = 0.0171472 loss)
    I0515 02:48:10.651358 23147 sgd_solver.cpp:106] Iteration 103300, lr = 3.2386e-05
    I0515 02:48:18.707439 23147 solver.cpp:228] Iteration 103400, loss = 0.0188847
    I0515 02:48:18.707495 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:48:18.707526 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0188846 (* 1 = 0.0188846 loss)
    I0515 02:48:18.707541 23147 sgd_solver.cpp:106] Iteration 103400, lr = 3.23646e-05
    I0515 02:48:26.760622 23147 solver.cpp:228] Iteration 103500, loss = 0.00812697
    I0515 02:48:26.760679 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:48:26.760704 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00812692 (* 1 = 0.00812692 loss)
    I0515 02:48:26.760720 23147 sgd_solver.cpp:106] Iteration 103500, lr = 3.23432e-05
    I0515 02:48:34.824626 23147 solver.cpp:228] Iteration 103600, loss = 0.0141087
    I0515 02:48:34.824748 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:48:34.824784 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0141087 (* 1 = 0.0141087 loss)
    I0515 02:48:34.824801 23147 sgd_solver.cpp:106] Iteration 103600, lr = 3.23219e-05
    I0515 02:48:42.950486 23147 solver.cpp:228] Iteration 103700, loss = 0.0267979
    I0515 02:48:42.950532 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:48:42.950551 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0267979 (* 1 = 0.0267979 loss)
    I0515 02:48:42.950567 23147 sgd_solver.cpp:106] Iteration 103700, lr = 3.23005e-05
    I0515 02:48:51.089198 23147 solver.cpp:228] Iteration 103800, loss = 0.00666117
    I0515 02:48:51.089246 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:48:51.089265 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00666111 (* 1 = 0.00666111 loss)
    I0515 02:48:51.089279 23147 sgd_solver.cpp:106] Iteration 103800, lr = 3.22792e-05
    I0515 02:48:59.228494 23147 solver.cpp:228] Iteration 103900, loss = 0.00978407
    I0515 02:48:59.228534 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:48:59.228554 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00978402 (* 1 = 0.00978402 loss)
    I0515 02:48:59.228569 23147 sgd_solver.cpp:106] Iteration 103900, lr = 3.2258e-05
    I0515 02:49:07.287292 23147 solver.cpp:337] Iteration 104000, Testing net (#0)
    I0515 02:49:11.708894 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.822583
    I0515 02:49:11.708950 23147 solver.cpp:404]     Test net output #1: loss_c = 0.573031 (* 1 = 0.573031 loss)
    I0515 02:49:11.760278 23147 solver.cpp:228] Iteration 104000, loss = 0.012197
    I0515 02:49:11.760344 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:49:11.760367 23147 solver.cpp:244]     Train net output #1: loss_c = 0.012197 (* 1 = 0.012197 loss)
    I0515 02:49:11.760386 23147 sgd_solver.cpp:106] Iteration 104000, lr = 3.22368e-05
    I0515 02:49:19.901583 23147 solver.cpp:228] Iteration 104100, loss = 0.0239923
    I0515 02:49:19.901633 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:49:19.901653 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0239923 (* 1 = 0.0239923 loss)
    I0515 02:49:19.901667 23147 sgd_solver.cpp:106] Iteration 104100, lr = 3.22156e-05
    I0515 02:49:28.042472 23147 solver.cpp:228] Iteration 104200, loss = 0.0143847
    I0515 02:49:28.042524 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:49:28.042546 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0143847 (* 1 = 0.0143847 loss)
    I0515 02:49:28.042559 23147 sgd_solver.cpp:106] Iteration 104200, lr = 3.21944e-05
    I0515 02:49:36.178995 23147 solver.cpp:228] Iteration 104300, loss = 0.00402573
    I0515 02:49:36.179044 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:49:36.179065 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00402569 (* 1 = 0.00402569 loss)
    I0515 02:49:36.179081 23147 sgd_solver.cpp:106] Iteration 104300, lr = 3.21733e-05
    I0515 02:49:44.316794 23147 solver.cpp:228] Iteration 104400, loss = 0.0232794
    I0515 02:49:44.316956 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:49:44.317001 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0232794 (* 1 = 0.0232794 loss)
    I0515 02:49:44.317026 23147 sgd_solver.cpp:106] Iteration 104400, lr = 3.21522e-05
    I0515 02:49:52.386304 23147 solver.cpp:228] Iteration 104500, loss = 0.0087161
    I0515 02:49:52.386353 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:49:52.386375 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00871605 (* 1 = 0.00871605 loss)
    I0515 02:49:52.386390 23147 sgd_solver.cpp:106] Iteration 104500, lr = 3.21311e-05
    I0515 02:50:00.524827 23147 solver.cpp:228] Iteration 104600, loss = 0.0777943
    I0515 02:50:00.524876 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:50:00.524896 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0777942 (* 1 = 0.0777942 loss)
    I0515 02:50:00.524910 23147 sgd_solver.cpp:106] Iteration 104600, lr = 3.21101e-05
    I0515 02:50:08.662981 23147 solver.cpp:228] Iteration 104700, loss = 0.0169391
    I0515 02:50:08.663031 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:50:08.663051 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0169391 (* 1 = 0.0169391 loss)
    I0515 02:50:08.663067 23147 sgd_solver.cpp:106] Iteration 104700, lr = 3.20891e-05
    I0515 02:50:16.792557 23147 solver.cpp:228] Iteration 104800, loss = 0.0381723
    I0515 02:50:16.792654 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:50:16.792675 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0381723 (* 1 = 0.0381723 loss)
    I0515 02:50:16.792690 23147 sgd_solver.cpp:106] Iteration 104800, lr = 3.20681e-05
    I0515 02:50:24.934350 23147 solver.cpp:228] Iteration 104900, loss = 0.0375498
    I0515 02:50:24.934396 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:50:24.934414 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0375497 (* 1 = 0.0375497 loss)
    I0515 02:50:24.934429 23147 sgd_solver.cpp:106] Iteration 104900, lr = 3.20472e-05
    I0515 02:50:32.997534 23147 solver.cpp:337] Iteration 105000, Testing net (#0)
    I0515 02:50:37.413735 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.816083
    I0515 02:50:37.413794 23147 solver.cpp:404]     Test net output #1: loss_c = 0.549471 (* 1 = 0.549471 loss)
    I0515 02:50:37.465121 23147 solver.cpp:228] Iteration 105000, loss = 0.0241401
    I0515 02:50:37.465154 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:50:37.465174 23147 solver.cpp:244]     Train net output #1: loss_c = 0.02414 (* 1 = 0.02414 loss)
    I0515 02:50:37.465193 23147 sgd_solver.cpp:106] Iteration 105000, lr = 3.20263e-05
    I0515 02:50:45.602998 23147 solver.cpp:228] Iteration 105100, loss = 0.0207129
    I0515 02:50:45.603044 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:50:45.603063 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0207128 (* 1 = 0.0207128 loss)
    I0515 02:50:45.603078 23147 sgd_solver.cpp:106] Iteration 105100, lr = 3.20054e-05
    I0515 02:50:53.741746 23147 solver.cpp:228] Iteration 105200, loss = 0.0580646
    I0515 02:50:53.741986 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:50:53.742036 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0580645 (* 1 = 0.0580645 loss)
    I0515 02:50:53.742061 23147 sgd_solver.cpp:106] Iteration 105200, lr = 3.19846e-05
    I0515 02:51:01.799026 23147 solver.cpp:228] Iteration 105300, loss = 0.00825226
    I0515 02:51:01.799075 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:51:01.799101 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0082522 (* 1 = 0.0082522 loss)
    I0515 02:51:01.799115 23147 sgd_solver.cpp:106] Iteration 105300, lr = 3.19638e-05
    I0515 02:51:09.868814 23147 solver.cpp:228] Iteration 105400, loss = 0.0243639
    I0515 02:51:09.868875 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:51:09.868903 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0243638 (* 1 = 0.0243638 loss)
    I0515 02:51:09.868926 23147 sgd_solver.cpp:106] Iteration 105400, lr = 3.1943e-05
    I0515 02:51:17.957713 23147 solver.cpp:228] Iteration 105500, loss = 0.0245074
    I0515 02:51:17.957767 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:51:17.957794 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0245074 (* 1 = 0.0245074 loss)
    I0515 02:51:17.957816 23147 sgd_solver.cpp:106] Iteration 105500, lr = 3.19222e-05
    I0515 02:51:26.046609 23147 solver.cpp:228] Iteration 105600, loss = 0.0251377
    I0515 02:51:26.046711 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:51:26.046741 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0251377 (* 1 = 0.0251377 loss)
    I0515 02:51:26.046762 23147 sgd_solver.cpp:106] Iteration 105600, lr = 3.19015e-05
    I0515 02:51:34.131924 23147 solver.cpp:228] Iteration 105700, loss = 0.00885159
    I0515 02:51:34.131979 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:51:34.132007 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00885153 (* 1 = 0.00885153 loss)
    I0515 02:51:34.132028 23147 sgd_solver.cpp:106] Iteration 105700, lr = 3.18809e-05
    I0515 02:51:42.220917 23147 solver.cpp:228] Iteration 105800, loss = 0.0113795
    I0515 02:51:42.220971 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:51:42.220999 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0113794 (* 1 = 0.0113794 loss)
    I0515 02:51:42.221020 23147 sgd_solver.cpp:106] Iteration 105800, lr = 3.18602e-05
    I0515 02:51:50.307333 23147 solver.cpp:228] Iteration 105900, loss = 0.0581393
    I0515 02:51:50.307389 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:51:50.307417 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0581393 (* 1 = 0.0581393 loss)
    I0515 02:51:50.307438 23147 sgd_solver.cpp:106] Iteration 105900, lr = 3.18396e-05
    I0515 02:51:58.313431 23147 solver.cpp:337] Iteration 106000, Testing net (#0)
    I0515 02:52:02.728425 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.8145
    I0515 02:52:02.728487 23147 solver.cpp:404]     Test net output #1: loss_c = 0.54205 (* 1 = 0.54205 loss)
    I0515 02:52:02.780953 23147 solver.cpp:228] Iteration 106000, loss = 0.0185119
    I0515 02:52:02.781018 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:52:02.781043 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0185118 (* 1 = 0.0185118 loss)
    I0515 02:52:02.781060 23147 sgd_solver.cpp:106] Iteration 106000, lr = 3.1819e-05
    I0515 02:52:10.920584 23147 solver.cpp:228] Iteration 106100, loss = 0.00817451
    I0515 02:52:10.920634 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:52:10.920655 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00817447 (* 1 = 0.00817447 loss)
    I0515 02:52:10.920670 23147 sgd_solver.cpp:106] Iteration 106100, lr = 3.17984e-05
    I0515 02:52:19.060184 23147 solver.cpp:228] Iteration 106200, loss = 0.033076
    I0515 02:52:19.060227 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:52:19.060247 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0330759 (* 1 = 0.0330759 loss)
    I0515 02:52:19.060261 23147 sgd_solver.cpp:106] Iteration 106200, lr = 3.17779e-05
    I0515 02:52:27.200266 23147 solver.cpp:228] Iteration 106300, loss = 0.0319509
    I0515 02:52:27.200312 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:52:27.200335 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0319508 (* 1 = 0.0319508 loss)
    I0515 02:52:27.200350 23147 sgd_solver.cpp:106] Iteration 106300, lr = 3.17574e-05
    I0515 02:52:35.340911 23147 solver.cpp:228] Iteration 106400, loss = 0.0172454
    I0515 02:52:35.341022 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:52:35.341043 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0172453 (* 1 = 0.0172453 loss)
    I0515 02:52:35.341058 23147 sgd_solver.cpp:106] Iteration 106400, lr = 3.1737e-05
    I0515 02:52:43.481747 23147 solver.cpp:228] Iteration 106500, loss = 0.00747621
    I0515 02:52:43.481796 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:52:43.481815 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00747615 (* 1 = 0.00747615 loss)
    I0515 02:52:43.481830 23147 sgd_solver.cpp:106] Iteration 106500, lr = 3.17165e-05
    I0515 02:52:51.571076 23147 solver.cpp:228] Iteration 106600, loss = 0.0202625
    I0515 02:52:51.571131 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:52:51.571161 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0202625 (* 1 = 0.0202625 loss)
    I0515 02:52:51.571182 23147 sgd_solver.cpp:106] Iteration 106600, lr = 3.16961e-05
    I0515 02:52:59.658236 23147 solver.cpp:228] Iteration 106700, loss = 0.0123192
    I0515 02:52:59.658293 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:52:59.658321 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0123191 (* 1 = 0.0123191 loss)
    I0515 02:52:59.658344 23147 sgd_solver.cpp:106] Iteration 106700, lr = 3.16757e-05
    I0515 02:53:07.744792 23147 solver.cpp:228] Iteration 106800, loss = 0.00900316
    I0515 02:53:07.745409 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:53:07.745440 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0090031 (* 1 = 0.0090031 loss)
    I0515 02:53:07.745460 23147 sgd_solver.cpp:106] Iteration 106800, lr = 3.16554e-05
    I0515 02:53:15.830343 23147 solver.cpp:228] Iteration 106900, loss = 0.0190142
    I0515 02:53:15.830400 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:53:15.830428 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0190141 (* 1 = 0.0190141 loss)
    I0515 02:53:15.830449 23147 sgd_solver.cpp:106] Iteration 106900, lr = 3.16351e-05
    I0515 02:53:23.836484 23147 solver.cpp:337] Iteration 107000, Testing net (#0)
    I0515 02:53:28.202117 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.81575
    I0515 02:53:28.202175 23147 solver.cpp:404]     Test net output #1: loss_c = 0.528816 (* 1 = 0.528816 loss)
    I0515 02:53:28.253612 23147 solver.cpp:228] Iteration 107000, loss = 0.071582
    I0515 02:53:28.253684 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.96
    I0515 02:53:28.253705 23147 solver.cpp:244]     Train net output #1: loss_c = 0.071582 (* 1 = 0.071582 loss)
    I0515 02:53:28.253722 23147 sgd_solver.cpp:106] Iteration 107000, lr = 3.16148e-05
    I0515 02:53:36.347617 23147 solver.cpp:228] Iteration 107100, loss = 0.00864961
    I0515 02:53:36.347669 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:53:36.347699 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00864954 (* 1 = 0.00864954 loss)
    I0515 02:53:36.347721 23147 sgd_solver.cpp:106] Iteration 107100, lr = 3.15946e-05
    I0515 02:53:44.472129 23147 solver.cpp:228] Iteration 107200, loss = 0.0144548
    I0515 02:53:44.472314 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:53:44.472335 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0144548 (* 1 = 0.0144548 loss)
    I0515 02:53:44.472350 23147 sgd_solver.cpp:106] Iteration 107200, lr = 3.15743e-05
    I0515 02:53:52.603014 23147 solver.cpp:228] Iteration 107300, loss = 0.0881623
    I0515 02:53:52.603061 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:53:52.603081 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0881623 (* 1 = 0.0881623 loss)
    I0515 02:53:52.603096 23147 sgd_solver.cpp:106] Iteration 107300, lr = 3.15542e-05
    I0515 02:54:00.730322 23147 solver.cpp:228] Iteration 107400, loss = 0.0286394
    I0515 02:54:00.730368 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:54:00.730387 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0286393 (* 1 = 0.0286393 loss)
    I0515 02:54:00.730402 23147 sgd_solver.cpp:106] Iteration 107400, lr = 3.1534e-05
    I0515 02:54:08.868134 23147 solver.cpp:228] Iteration 107500, loss = 0.010977
    I0515 02:54:08.868181 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:54:08.868201 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0109769 (* 1 = 0.0109769 loss)
    I0515 02:54:08.868216 23147 sgd_solver.cpp:106] Iteration 107500, lr = 3.15139e-05
    I0515 02:54:16.996850 23147 solver.cpp:228] Iteration 107600, loss = 0.023093
    I0515 02:54:16.996997 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:54:16.997045 23147 solver.cpp:244]     Train net output #1: loss_c = 0.023093 (* 1 = 0.023093 loss)
    I0515 02:54:16.997069 23147 sgd_solver.cpp:106] Iteration 107600, lr = 3.14938e-05
    I0515 02:54:25.134102 23147 solver.cpp:228] Iteration 107700, loss = 0.0695528
    I0515 02:54:25.134152 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:54:25.134172 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0695527 (* 1 = 0.0695527 loss)
    I0515 02:54:25.134187 23147 sgd_solver.cpp:106] Iteration 107700, lr = 3.14737e-05
    I0515 02:54:33.274274 23147 solver.cpp:228] Iteration 107800, loss = 0.00578764
    I0515 02:54:33.274323 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:54:33.274344 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00578758 (* 1 = 0.00578758 loss)
    I0515 02:54:33.274358 23147 sgd_solver.cpp:106] Iteration 107800, lr = 3.14536e-05
    I0515 02:54:41.363970 23147 solver.cpp:228] Iteration 107900, loss = 0.0102272
    I0515 02:54:41.364030 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:54:41.364059 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0102272 (* 1 = 0.0102272 loss)
    I0515 02:54:41.364080 23147 sgd_solver.cpp:106] Iteration 107900, lr = 3.14336e-05
    I0515 02:54:49.377106 23147 solver.cpp:337] Iteration 108000, Testing net (#0)
    I0515 02:54:53.791587 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.823416
    I0515 02:54:53.791637 23147 solver.cpp:404]     Test net output #1: loss_c = 0.551039 (* 1 = 0.551039 loss)
    I0515 02:54:53.843981 23147 solver.cpp:228] Iteration 108000, loss = 0.0270759
    I0515 02:54:53.844039 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:54:53.844063 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0270758 (* 1 = 0.0270758 loss)
    I0515 02:54:53.844081 23147 sgd_solver.cpp:106] Iteration 108000, lr = 3.14137e-05
    I0515 02:55:01.984341 23147 solver.cpp:228] Iteration 108100, loss = 0.0193589
    I0515 02:55:01.984388 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:55:01.984411 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0193588 (* 1 = 0.0193588 loss)
    I0515 02:55:01.984426 23147 sgd_solver.cpp:106] Iteration 108100, lr = 3.13937e-05
    I0515 02:55:10.121358 23147 solver.cpp:228] Iteration 108200, loss = 0.0107813
    I0515 02:55:10.121408 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:55:10.121428 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0107812 (* 1 = 0.0107812 loss)
    I0515 02:55:10.121443 23147 sgd_solver.cpp:106] Iteration 108200, lr = 3.13738e-05
    I0515 02:55:18.261262 23147 solver.cpp:228] Iteration 108300, loss = 0.00956612
    I0515 02:55:18.261308 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:55:18.261328 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00956607 (* 1 = 0.00956607 loss)
    I0515 02:55:18.261343 23147 sgd_solver.cpp:106] Iteration 108300, lr = 3.13539e-05
    I0515 02:55:26.401856 23147 solver.cpp:228] Iteration 108400, loss = 0.00863991
    I0515 02:55:26.402048 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:55:26.402070 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00863985 (* 1 = 0.00863985 loss)
    I0515 02:55:26.402084 23147 sgd_solver.cpp:106] Iteration 108400, lr = 3.1334e-05
    I0515 02:55:34.543653 23147 solver.cpp:228] Iteration 108500, loss = 0.0084844
    I0515 02:55:34.543699 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:55:34.543721 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00848435 (* 1 = 0.00848435 loss)
    I0515 02:55:34.543735 23147 sgd_solver.cpp:106] Iteration 108500, lr = 3.13142e-05
    I0515 02:55:42.683434 23147 solver.cpp:228] Iteration 108600, loss = 0.00650559
    I0515 02:55:42.683493 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:55:42.683521 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00650554 (* 1 = 0.00650554 loss)
    I0515 02:55:42.683536 23147 sgd_solver.cpp:106] Iteration 108600, lr = 3.12944e-05
    I0515 02:55:50.823791 23147 solver.cpp:228] Iteration 108700, loss = 0.0307474
    I0515 02:55:50.823838 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:55:50.823859 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0307473 (* 1 = 0.0307473 loss)
    I0515 02:55:50.823874 23147 sgd_solver.cpp:106] Iteration 108700, lr = 3.12746e-05
    I0515 02:55:58.963135 23147 solver.cpp:228] Iteration 108800, loss = 0.00829085
    I0515 02:55:58.963243 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:55:58.963264 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00829081 (* 1 = 0.00829081 loss)
    I0515 02:55:58.963279 23147 sgd_solver.cpp:106] Iteration 108800, lr = 3.12549e-05
    I0515 02:56:07.101676 23147 solver.cpp:228] Iteration 108900, loss = 0.0206409
    I0515 02:56:07.101722 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:56:07.101742 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0206409 (* 1 = 0.0206409 loss)
    I0515 02:56:07.101758 23147 sgd_solver.cpp:106] Iteration 108900, lr = 3.12352e-05
    I0515 02:56:15.160274 23147 solver.cpp:337] Iteration 109000, Testing net (#0)
    I0515 02:56:19.544435 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.81825
    I0515 02:56:19.544495 23147 solver.cpp:404]     Test net output #1: loss_c = 0.552576 (* 1 = 0.552576 loss)
    I0515 02:56:19.600642 23147 solver.cpp:228] Iteration 109000, loss = 0.00818252
    I0515 02:56:19.600694 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:56:19.600723 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00818247 (* 1 = 0.00818247 loss)
    I0515 02:56:19.600750 23147 sgd_solver.cpp:106] Iteration 109000, lr = 3.12155e-05
    I0515 02:56:27.687034 23147 solver.cpp:228] Iteration 109100, loss = 0.0106028
    I0515 02:56:27.687095 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:56:27.687124 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0106027 (* 1 = 0.0106027 loss)
    I0515 02:56:27.687146 23147 sgd_solver.cpp:106] Iteration 109100, lr = 3.11958e-05
    I0515 02:56:35.774368 23147 solver.cpp:228] Iteration 109200, loss = 0.0168239
    I0515 02:56:35.774585 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:56:35.774637 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0168238 (* 1 = 0.0168238 loss)
    I0515 02:56:35.774659 23147 sgd_solver.cpp:106] Iteration 109200, lr = 3.11762e-05
    I0515 02:56:43.884439 23147 solver.cpp:228] Iteration 109300, loss = 0.0271417
    I0515 02:56:43.884490 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:56:43.884511 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0271416 (* 1 = 0.0271416 loss)
    I0515 02:56:43.884526 23147 sgd_solver.cpp:106] Iteration 109300, lr = 3.11566e-05
    I0515 02:56:52.021975 23147 solver.cpp:228] Iteration 109400, loss = 0.0143771
    I0515 02:56:52.022027 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:56:52.022047 23147 solver.cpp:244]     Train net output #1: loss_c = 0.014377 (* 1 = 0.014377 loss)
    I0515 02:56:52.022061 23147 sgd_solver.cpp:106] Iteration 109400, lr = 3.1137e-05
    I0515 02:57:00.161846 23147 solver.cpp:228] Iteration 109500, loss = 0.00650165
    I0515 02:57:00.161898 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:57:00.161919 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00650161 (* 1 = 0.00650161 loss)
    I0515 02:57:00.161933 23147 sgd_solver.cpp:106] Iteration 109500, lr = 3.11175e-05
    I0515 02:57:08.297047 23147 solver.cpp:228] Iteration 109600, loss = 0.0409774
    I0515 02:57:08.297147 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:57:08.297168 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0409774 (* 1 = 0.0409774 loss)
    I0515 02:57:08.297183 23147 sgd_solver.cpp:106] Iteration 109600, lr = 3.10979e-05
    I0515 02:57:16.436740 23147 solver.cpp:228] Iteration 109700, loss = 0.0078846
    I0515 02:57:16.436791 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:57:16.436811 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00788455 (* 1 = 0.00788455 loss)
    I0515 02:57:16.436826 23147 sgd_solver.cpp:106] Iteration 109700, lr = 3.10785e-05
    I0515 02:57:24.571373 23147 solver.cpp:228] Iteration 109800, loss = 0.0138928
    I0515 02:57:24.571416 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:57:24.571436 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0138927 (* 1 = 0.0138927 loss)
    I0515 02:57:24.571450 23147 sgd_solver.cpp:106] Iteration 109800, lr = 3.1059e-05
    I0515 02:57:32.708328 23147 solver.cpp:228] Iteration 109900, loss = 0.0157075
    I0515 02:57:32.708374 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:57:32.708392 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0157074 (* 1 = 0.0157074 loss)
    I0515 02:57:32.708407 23147 sgd_solver.cpp:106] Iteration 109900, lr = 3.10396e-05
    I0515 02:57:40.768872 23147 solver.cpp:337] Iteration 110000, Testing net (#0)
    I0515 02:57:45.192246 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.818083
    I0515 02:57:45.192293 23147 solver.cpp:404]     Test net output #1: loss_c = 0.542932 (* 1 = 0.542932 loss)
    I0515 02:57:45.247376 23147 solver.cpp:228] Iteration 110000, loss = 0.0380286
    I0515 02:57:45.247439 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:57:45.247469 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0380286 (* 1 = 0.0380286 loss)
    I0515 02:57:45.247493 23147 sgd_solver.cpp:106] Iteration 110000, lr = 3.10202e-05
    I0515 02:57:53.381448 23147 solver.cpp:228] Iteration 110100, loss = 0.0749848
    I0515 02:57:53.381495 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 02:57:53.381518 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0749847 (* 1 = 0.0749847 loss)
    I0515 02:57:53.381533 23147 sgd_solver.cpp:106] Iteration 110100, lr = 3.10008e-05
    I0515 02:58:01.518158 23147 solver.cpp:228] Iteration 110200, loss = 0.0103886
    I0515 02:58:01.518206 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:58:01.518226 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0103885 (* 1 = 0.0103885 loss)
    I0515 02:58:01.518240 23147 sgd_solver.cpp:106] Iteration 110200, lr = 3.09814e-05
    I0515 02:58:09.653416 23147 solver.cpp:228] Iteration 110300, loss = 0.00947146
    I0515 02:58:09.653467 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:58:09.653491 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0094714 (* 1 = 0.0094714 loss)
    I0515 02:58:09.653506 23147 sgd_solver.cpp:106] Iteration 110300, lr = 3.09621e-05
    I0515 02:58:17.793292 23147 solver.cpp:228] Iteration 110400, loss = 0.0116102
    I0515 02:58:17.793485 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:58:17.793508 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0116101 (* 1 = 0.0116101 loss)
    I0515 02:58:17.793550 23147 sgd_solver.cpp:106] Iteration 110400, lr = 3.09428e-05
    I0515 02:58:25.930171 23147 solver.cpp:228] Iteration 110500, loss = 0.0254484
    I0515 02:58:25.930212 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:58:25.930233 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0254484 (* 1 = 0.0254484 loss)
    I0515 02:58:25.930249 23147 sgd_solver.cpp:106] Iteration 110500, lr = 3.09236e-05
    I0515 02:58:34.069299 23147 solver.cpp:228] Iteration 110600, loss = 0.0103426
    I0515 02:58:34.069353 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:58:34.069373 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0103426 (* 1 = 0.0103426 loss)
    I0515 02:58:34.069389 23147 sgd_solver.cpp:106] Iteration 110600, lr = 3.09043e-05
    I0515 02:58:42.207195 23147 solver.cpp:228] Iteration 110700, loss = 0.0349667
    I0515 02:58:42.207247 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:58:42.207267 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0349666 (* 1 = 0.0349666 loss)
    I0515 02:58:42.207280 23147 sgd_solver.cpp:106] Iteration 110700, lr = 3.08851e-05
    I0515 02:58:50.344105 23147 solver.cpp:228] Iteration 110800, loss = 0.00649762
    I0515 02:58:50.344199 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:58:50.344223 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00649758 (* 1 = 0.00649758 loss)
    I0515 02:58:50.344238 23147 sgd_solver.cpp:106] Iteration 110800, lr = 3.0866e-05
    I0515 02:58:58.481048 23147 solver.cpp:228] Iteration 110900, loss = 0.0350726
    I0515 02:58:58.481096 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:58:58.481117 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0350726 (* 1 = 0.0350726 loss)
    I0515 02:58:58.481132 23147 sgd_solver.cpp:106] Iteration 110900, lr = 3.08468e-05
    I0515 02:59:06.538822 23147 solver.cpp:337] Iteration 111000, Testing net (#0)
    I0515 02:59:10.958763 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.820666
    I0515 02:59:10.958822 23147 solver.cpp:404]     Test net output #1: loss_c = 0.568383 (* 1 = 0.568383 loss)
    I0515 02:59:11.011129 23147 solver.cpp:228] Iteration 111000, loss = 0.0332161
    I0515 02:59:11.011155 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 02:59:11.011175 23147 solver.cpp:244]     Train net output #1: loss_c = 0.033216 (* 1 = 0.033216 loss)
    I0515 02:59:11.011195 23147 sgd_solver.cpp:106] Iteration 111000, lr = 3.08277e-05
    I0515 02:59:19.150702 23147 solver.cpp:228] Iteration 111100, loss = 0.0241089
    I0515 02:59:19.150753 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:59:19.150776 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0241089 (* 1 = 0.0241089 loss)
    I0515 02:59:19.150791 23147 sgd_solver.cpp:106] Iteration 111100, lr = 3.08086e-05
    I0515 02:59:27.291018 23147 solver.cpp:228] Iteration 111200, loss = 0.0123743
    I0515 02:59:27.291210 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:59:27.291231 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0123742 (* 1 = 0.0123742 loss)
    I0515 02:59:27.291246 23147 sgd_solver.cpp:106] Iteration 111200, lr = 3.07895e-05
    I0515 02:59:35.423876 23147 solver.cpp:228] Iteration 111300, loss = 0.0154355
    I0515 02:59:35.423923 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:59:35.423944 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0154355 (* 1 = 0.0154355 loss)
    I0515 02:59:35.423959 23147 sgd_solver.cpp:106] Iteration 111300, lr = 3.07705e-05
    I0515 02:59:43.564282 23147 solver.cpp:228] Iteration 111400, loss = 0.00989743
    I0515 02:59:43.564332 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:59:43.564353 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00989738 (* 1 = 0.00989738 loss)
    I0515 02:59:43.564368 23147 sgd_solver.cpp:106] Iteration 111400, lr = 3.07515e-05
    I0515 02:59:51.696655 23147 solver.cpp:228] Iteration 111500, loss = 0.0126838
    I0515 02:59:51.696707 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 02:59:51.696727 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0126838 (* 1 = 0.0126838 loss)
    I0515 02:59:51.696743 23147 sgd_solver.cpp:106] Iteration 111500, lr = 3.07325e-05
    I0515 02:59:59.836277 23147 solver.cpp:228] Iteration 111600, loss = 0.0189822
    I0515 02:59:59.836437 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 02:59:59.836483 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0189822 (* 1 = 0.0189822 loss)
    I0515 02:59:59.836506 23147 sgd_solver.cpp:106] Iteration 111600, lr = 3.07135e-05
    I0515 03:00:07.927652 23147 solver.cpp:228] Iteration 111700, loss = 0.0198967
    I0515 03:00:07.927708 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:00:07.927737 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0198966 (* 1 = 0.0198966 loss)
    I0515 03:00:07.927759 23147 sgd_solver.cpp:106] Iteration 111700, lr = 3.06946e-05
    I0515 03:00:16.017858 23147 solver.cpp:228] Iteration 111800, loss = 0.0084369
    I0515 03:00:16.017915 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:00:16.017946 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00843685 (* 1 = 0.00843685 loss)
    I0515 03:00:16.017966 23147 sgd_solver.cpp:106] Iteration 111800, lr = 3.06757e-05
    I0515 03:00:24.106050 23147 solver.cpp:228] Iteration 111900, loss = 0.0123314
    I0515 03:00:24.106106 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:00:24.106135 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0123313 (* 1 = 0.0123313 loss)
    I0515 03:00:24.106158 23147 sgd_solver.cpp:106] Iteration 111900, lr = 3.06568e-05
    I0515 03:00:32.115486 23147 solver.cpp:337] Iteration 112000, Testing net (#0)
    I0515 03:00:36.464920 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.81625
    I0515 03:00:36.464982 23147 solver.cpp:404]     Test net output #1: loss_c = 0.571463 (* 1 = 0.571463 loss)
    I0515 03:00:36.516676 23147 solver.cpp:228] Iteration 112000, loss = 0.0113652
    I0515 03:00:36.516721 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:00:36.516739 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0113652 (* 1 = 0.0113652 loss)
    I0515 03:00:36.516757 23147 sgd_solver.cpp:106] Iteration 112000, lr = 3.0638e-05
    I0515 03:00:44.609573 23147 solver.cpp:228] Iteration 112100, loss = 0.0168333
    I0515 03:00:44.609628 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:00:44.609658 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0168332 (* 1 = 0.0168332 loss)
    I0515 03:00:44.609679 23147 sgd_solver.cpp:106] Iteration 112100, lr = 3.06192e-05
    I0515 03:00:52.702203 23147 solver.cpp:228] Iteration 112200, loss = 0.0233448
    I0515 03:00:52.702263 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:00:52.702292 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0233448 (* 1 = 0.0233448 loss)
    I0515 03:00:52.702314 23147 sgd_solver.cpp:106] Iteration 112200, lr = 3.06004e-05
    I0515 03:01:00.791003 23147 solver.cpp:228] Iteration 112300, loss = 0.0161087
    I0515 03:01:00.791062 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:01:00.791091 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0161087 (* 1 = 0.0161087 loss)
    I0515 03:01:00.791113 23147 sgd_solver.cpp:106] Iteration 112300, lr = 3.05816e-05
    I0515 03:01:08.878959 23147 solver.cpp:228] Iteration 112400, loss = 0.0301463
    I0515 03:01:08.879187 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:01:08.879232 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0301462 (* 1 = 0.0301462 loss)
    I0515 03:01:08.879257 23147 sgd_solver.cpp:106] Iteration 112400, lr = 3.05629e-05
    I0515 03:01:17.013048 23147 solver.cpp:228] Iteration 112500, loss = 0.0268448
    I0515 03:01:17.013095 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:01:17.013116 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0268447 (* 1 = 0.0268447 loss)
    I0515 03:01:17.013130 23147 sgd_solver.cpp:106] Iteration 112500, lr = 3.05441e-05
    I0515 03:01:25.153698 23147 solver.cpp:228] Iteration 112600, loss = 0.0194106
    I0515 03:01:25.153748 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:01:25.153769 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0194106 (* 1 = 0.0194106 loss)
    I0515 03:01:25.153785 23147 sgd_solver.cpp:106] Iteration 112600, lr = 3.05255e-05
    I0515 03:01:33.295356 23147 solver.cpp:228] Iteration 112700, loss = 0.0176827
    I0515 03:01:33.295397 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:01:33.295418 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0176826 (* 1 = 0.0176826 loss)
    I0515 03:01:33.295433 23147 sgd_solver.cpp:106] Iteration 112700, lr = 3.05068e-05
    I0515 03:01:41.434221 23147 solver.cpp:228] Iteration 112800, loss = 0.0176891
    I0515 03:01:41.434305 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:01:41.434327 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0176891 (* 1 = 0.0176891 loss)
    I0515 03:01:41.434343 23147 sgd_solver.cpp:106] Iteration 112800, lr = 3.04882e-05
    I0515 03:01:49.575286 23147 solver.cpp:228] Iteration 112900, loss = 0.00494092
    I0515 03:01:49.575330 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:01:49.575350 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00494086 (* 1 = 0.00494086 loss)
    I0515 03:01:49.575364 23147 sgd_solver.cpp:106] Iteration 112900, lr = 3.04696e-05
    I0515 03:01:57.634794 23147 solver.cpp:337] Iteration 113000, Testing net (#0)
    I0515 03:02:02.036348 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.8145
    I0515 03:02:02.036401 23147 solver.cpp:404]     Test net output #1: loss_c = 0.557297 (* 1 = 0.557297 loss)
    I0515 03:02:02.092751 23147 solver.cpp:228] Iteration 113000, loss = 0.0302762
    I0515 03:02:02.092811 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:02:02.092842 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0302761 (* 1 = 0.0302761 loss)
    I0515 03:02:02.092869 23147 sgd_solver.cpp:106] Iteration 113000, lr = 3.0451e-05
    I0515 03:02:10.221781 23147 solver.cpp:228] Iteration 113100, loss = 0.0166143
    I0515 03:02:10.221832 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:02:10.221853 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0166142 (* 1 = 0.0166142 loss)
    I0515 03:02:10.221868 23147 sgd_solver.cpp:106] Iteration 113100, lr = 3.04324e-05
    I0515 03:02:18.363718 23147 solver.cpp:228] Iteration 113200, loss = 0.00928574
    I0515 03:02:18.363808 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:02:18.363831 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00928566 (* 1 = 0.00928566 loss)
    I0515 03:02:18.363845 23147 sgd_solver.cpp:106] Iteration 113200, lr = 3.04139e-05
    I0515 03:02:26.504220 23147 solver.cpp:228] Iteration 113300, loss = 0.0139132
    I0515 03:02:26.504276 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:02:26.504304 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0139131 (* 1 = 0.0139131 loss)
    I0515 03:02:26.504326 23147 sgd_solver.cpp:106] Iteration 113300, lr = 3.03954e-05
    I0515 03:02:34.594758 23147 solver.cpp:228] Iteration 113400, loss = 0.0152306
    I0515 03:02:34.594808 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:02:34.594838 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0152305 (* 1 = 0.0152305 loss)
    I0515 03:02:34.594859 23147 sgd_solver.cpp:106] Iteration 113400, lr = 3.03769e-05
    I0515 03:02:42.685746 23147 solver.cpp:228] Iteration 113500, loss = 0.0149845
    I0515 03:02:42.685797 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:02:42.685827 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0149844 (* 1 = 0.0149844 loss)
    I0515 03:02:42.685848 23147 sgd_solver.cpp:106] Iteration 113500, lr = 3.03585e-05
    I0515 03:02:50.772677 23147 solver.cpp:228] Iteration 113600, loss = 0.0303386
    I0515 03:02:50.772879 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:02:50.772909 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0303385 (* 1 = 0.0303385 loss)
    I0515 03:02:50.772949 23147 sgd_solver.cpp:106] Iteration 113600, lr = 3.034e-05
    I0515 03:02:58.859292 23147 solver.cpp:228] Iteration 113700, loss = 0.0168763
    I0515 03:02:58.859344 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:02:58.859374 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0168762 (* 1 = 0.0168762 loss)
    I0515 03:02:58.859395 23147 sgd_solver.cpp:106] Iteration 113700, lr = 3.03216e-05
    I0515 03:03:06.949884 23147 solver.cpp:228] Iteration 113800, loss = 0.0167141
    I0515 03:03:06.949946 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:03:06.949976 23147 solver.cpp:244]     Train net output #1: loss_c = 0.016714 (* 1 = 0.016714 loss)
    I0515 03:03:06.949997 23147 sgd_solver.cpp:106] Iteration 113800, lr = 3.03033e-05
    I0515 03:03:15.037190 23147 solver.cpp:228] Iteration 113900, loss = 0.0172946
    I0515 03:03:15.037246 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:03:15.037276 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0172945 (* 1 = 0.0172945 loss)
    I0515 03:03:15.037297 23147 sgd_solver.cpp:106] Iteration 113900, lr = 3.02849e-05
    I0515 03:03:23.040238 23147 solver.cpp:337] Iteration 114000, Testing net (#0)
    I0515 03:03:27.381541 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.823417
    I0515 03:03:27.381602 23147 solver.cpp:404]     Test net output #1: loss_c = 0.53862 (* 1 = 0.53862 loss)
    I0515 03:03:27.432945 23147 solver.cpp:228] Iteration 114000, loss = 0.0209731
    I0515 03:03:27.432988 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:03:27.433008 23147 solver.cpp:244]     Train net output #1: loss_c = 0.020973 (* 1 = 0.020973 loss)
    I0515 03:03:27.433025 23147 sgd_solver.cpp:106] Iteration 114000, lr = 3.02666e-05
    I0515 03:03:35.573493 23147 solver.cpp:228] Iteration 114100, loss = 0.0149708
    I0515 03:03:35.573537 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:03:35.573557 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0149707 (* 1 = 0.0149707 loss)
    I0515 03:03:35.573571 23147 sgd_solver.cpp:106] Iteration 114100, lr = 3.02483e-05
    I0515 03:03:43.710872 23147 solver.cpp:228] Iteration 114200, loss = 0.0120013
    I0515 03:03:43.710919 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:03:43.710939 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0120012 (* 1 = 0.0120012 loss)
    I0515 03:03:43.710954 23147 sgd_solver.cpp:106] Iteration 114200, lr = 3.023e-05
    I0515 03:03:51.845937 23147 solver.cpp:228] Iteration 114300, loss = 0.0320113
    I0515 03:03:51.845988 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:03:51.846009 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0320112 (* 1 = 0.0320112 loss)
    I0515 03:03:51.846024 23147 sgd_solver.cpp:106] Iteration 114300, lr = 3.02118e-05
    I0515 03:03:59.984071 23147 solver.cpp:228] Iteration 114400, loss = 0.00954143
    I0515 03:03:59.984267 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:03:59.984292 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00954134 (* 1 = 0.00954134 loss)
    I0515 03:03:59.984308 23147 sgd_solver.cpp:106] Iteration 114400, lr = 3.01936e-05
    I0515 03:04:08.109227 23147 solver.cpp:228] Iteration 114500, loss = 0.0137925
    I0515 03:04:08.109278 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:04:08.109299 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0137924 (* 1 = 0.0137924 loss)
    I0515 03:04:08.109313 23147 sgd_solver.cpp:106] Iteration 114500, lr = 3.01754e-05
    I0515 03:04:16.244716 23147 solver.cpp:228] Iteration 114600, loss = 0.0193969
    I0515 03:04:16.244766 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:04:16.244786 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0193969 (* 1 = 0.0193969 loss)
    I0515 03:04:16.244801 23147 sgd_solver.cpp:106] Iteration 114600, lr = 3.01572e-05
    I0515 03:04:24.376514 23147 solver.cpp:228] Iteration 114700, loss = 0.0144788
    I0515 03:04:24.376566 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:04:24.376587 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0144787 (* 1 = 0.0144787 loss)
    I0515 03:04:24.376602 23147 sgd_solver.cpp:106] Iteration 114700, lr = 3.01391e-05
    I0515 03:04:32.509315 23147 solver.cpp:228] Iteration 114800, loss = 0.00933189
    I0515 03:04:32.509461 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:04:32.509508 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0093318 (* 1 = 0.0093318 loss)
    I0515 03:04:32.509533 23147 sgd_solver.cpp:106] Iteration 114800, lr = 3.0121e-05
    I0515 03:04:40.631491 23147 solver.cpp:228] Iteration 114900, loss = 0.0141377
    I0515 03:04:40.631537 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:04:40.631557 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0141376 (* 1 = 0.0141376 loss)
    I0515 03:04:40.631572 23147 sgd_solver.cpp:106] Iteration 114900, lr = 3.01029e-05
    I0515 03:04:48.683203 23147 solver.cpp:337] Iteration 115000, Testing net (#0)
    I0515 03:04:53.102123 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.820417
    I0515 03:04:53.102182 23147 solver.cpp:404]     Test net output #1: loss_c = 0.560527 (* 1 = 0.560527 loss)
    I0515 03:04:53.157353 23147 solver.cpp:228] Iteration 115000, loss = 0.0130231
    I0515 03:04:53.157414 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:04:53.157444 23147 solver.cpp:244]     Train net output #1: loss_c = 0.013023 (* 1 = 0.013023 loss)
    I0515 03:04:53.157470 23147 sgd_solver.cpp:106] Iteration 115000, lr = 3.00848e-05
    I0515 03:05:01.282727 23147 solver.cpp:228] Iteration 115100, loss = 0.059627
    I0515 03:05:01.282778 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 03:05:01.282798 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0596269 (* 1 = 0.0596269 loss)
    I0515 03:05:01.282812 23147 sgd_solver.cpp:106] Iteration 115100, lr = 3.00668e-05
    I0515 03:05:09.417434 23147 solver.cpp:228] Iteration 115200, loss = 0.0364976
    I0515 03:05:09.417515 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:05:09.417539 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0364975 (* 1 = 0.0364975 loss)
    I0515 03:05:09.417556 23147 sgd_solver.cpp:106] Iteration 115200, lr = 3.00488e-05
    I0515 03:05:17.553923 23147 solver.cpp:228] Iteration 115300, loss = 0.0105598
    I0515 03:05:17.553973 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:05:17.553993 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0105597 (* 1 = 0.0105597 loss)
    I0515 03:05:17.554008 23147 sgd_solver.cpp:106] Iteration 115300, lr = 3.00308e-05
    I0515 03:05:25.691179 23147 solver.cpp:228] Iteration 115400, loss = 0.0124058
    I0515 03:05:25.691228 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:05:25.691248 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0124057 (* 1 = 0.0124057 loss)
    I0515 03:05:25.691263 23147 sgd_solver.cpp:106] Iteration 115400, lr = 3.00128e-05
    I0515 03:05:33.830807 23147 solver.cpp:228] Iteration 115500, loss = 0.00922525
    I0515 03:05:33.830855 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:05:33.830876 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00922517 (* 1 = 0.00922517 loss)
    I0515 03:05:33.830890 23147 sgd_solver.cpp:106] Iteration 115500, lr = 2.99949e-05
    I0515 03:05:41.967592 23147 solver.cpp:228] Iteration 115600, loss = 0.0122143
    I0515 03:05:41.967864 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:05:41.967913 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0122143 (* 1 = 0.0122143 loss)
    I0515 03:05:41.967938 23147 sgd_solver.cpp:106] Iteration 115600, lr = 2.9977e-05
    I0515 03:05:50.081152 23147 solver.cpp:228] Iteration 115700, loss = 0.014078
    I0515 03:05:50.081214 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:05:50.081243 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0140779 (* 1 = 0.0140779 loss)
    I0515 03:05:50.081264 23147 sgd_solver.cpp:106] Iteration 115700, lr = 2.99591e-05
    I0515 03:05:58.170119 23147 solver.cpp:228] Iteration 115800, loss = 0.00852074
    I0515 03:05:58.170174 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:05:58.170203 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00852066 (* 1 = 0.00852066 loss)
    I0515 03:05:58.170225 23147 sgd_solver.cpp:106] Iteration 115800, lr = 2.99412e-05
    I0515 03:06:06.256935 23147 solver.cpp:228] Iteration 115900, loss = 0.0208623
    I0515 03:06:06.256995 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:06:06.257025 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0208622 (* 1 = 0.0208622 loss)
    I0515 03:06:06.257046 23147 sgd_solver.cpp:106] Iteration 115900, lr = 2.99234e-05
    I0515 03:06:14.263470 23147 solver.cpp:337] Iteration 116000, Testing net (#0)
    I0515 03:06:18.682580 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.814917
    I0515 03:06:18.682628 23147 solver.cpp:404]     Test net output #1: loss_c = 0.560911 (* 1 = 0.560911 loss)
    I0515 03:06:18.734163 23147 solver.cpp:228] Iteration 116000, loss = 0.020971
    I0515 03:06:18.734230 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:06:18.734253 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0209709 (* 1 = 0.0209709 loss)
    I0515 03:06:18.734272 23147 sgd_solver.cpp:106] Iteration 116000, lr = 2.99056e-05
    I0515 03:06:26.827621 23147 solver.cpp:228] Iteration 116100, loss = 0.0248353
    I0515 03:06:26.827677 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:06:26.827707 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0248352 (* 1 = 0.0248352 loss)
    I0515 03:06:26.827728 23147 sgd_solver.cpp:106] Iteration 116100, lr = 2.98878e-05
    I0515 03:06:34.913962 23147 solver.cpp:228] Iteration 116200, loss = 0.0225045
    I0515 03:06:34.914019 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:06:34.914048 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0225044 (* 1 = 0.0225044 loss)
    I0515 03:06:34.914069 23147 sgd_solver.cpp:106] Iteration 116200, lr = 2.987e-05
    I0515 03:06:43.006141 23147 solver.cpp:228] Iteration 116300, loss = 0.0197488
    I0515 03:06:43.006202 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:06:43.006232 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0197487 (* 1 = 0.0197487 loss)
    I0515 03:06:43.006253 23147 sgd_solver.cpp:106] Iteration 116300, lr = 2.98523e-05
    I0515 03:06:51.092643 23147 solver.cpp:228] Iteration 116400, loss = 0.0430128
    I0515 03:06:51.092870 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:06:51.092902 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0430127 (* 1 = 0.0430127 loss)
    I0515 03:06:51.092924 23147 sgd_solver.cpp:106] Iteration 116400, lr = 2.98346e-05
    I0515 03:06:59.177961 23147 solver.cpp:228] Iteration 116500, loss = 0.0157507
    I0515 03:06:59.178015 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:06:59.178043 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0157506 (* 1 = 0.0157506 loss)
    I0515 03:06:59.178064 23147 sgd_solver.cpp:106] Iteration 116500, lr = 2.98169e-05
    I0515 03:07:07.262953 23147 solver.cpp:228] Iteration 116600, loss = 0.0252348
    I0515 03:07:07.263008 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:07:07.263036 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0252347 (* 1 = 0.0252347 loss)
    I0515 03:07:07.263057 23147 sgd_solver.cpp:106] Iteration 116600, lr = 2.97992e-05
    I0515 03:07:15.352499 23147 solver.cpp:228] Iteration 116700, loss = 0.0254487
    I0515 03:07:15.352556 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:07:15.352584 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0254486 (* 1 = 0.0254486 loss)
    I0515 03:07:15.352607 23147 sgd_solver.cpp:106] Iteration 116700, lr = 2.97816e-05
    I0515 03:07:23.438359 23147 solver.cpp:228] Iteration 116800, loss = 0.0446399
    I0515 03:07:23.438465 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 03:07:23.438494 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0446398 (* 1 = 0.0446398 loss)
    I0515 03:07:23.438516 23147 sgd_solver.cpp:106] Iteration 116800, lr = 2.97639e-05
    I0515 03:07:31.522689 23147 solver.cpp:228] Iteration 116900, loss = 0.0183915
    I0515 03:07:31.522743 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:07:31.522771 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0183914 (* 1 = 0.0183914 loss)
    I0515 03:07:31.522794 23147 sgd_solver.cpp:106] Iteration 116900, lr = 2.97464e-05
    I0515 03:07:39.529647 23147 solver.cpp:337] Iteration 117000, Testing net (#0)
    I0515 03:07:43.925065 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.818667
    I0515 03:07:43.925114 23147 solver.cpp:404]     Test net output #1: loss_c = 0.55605 (* 1 = 0.55605 loss)
    I0515 03:07:43.980432 23147 solver.cpp:228] Iteration 117000, loss = 0.0144764
    I0515 03:07:43.980495 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:07:43.980525 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0144763 (* 1 = 0.0144763 loss)
    I0515 03:07:43.980551 23147 sgd_solver.cpp:106] Iteration 117000, lr = 2.97288e-05
    I0515 03:07:52.112320 23147 solver.cpp:228] Iteration 117100, loss = 0.0130811
    I0515 03:07:52.112370 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:07:52.112390 23147 solver.cpp:244]     Train net output #1: loss_c = 0.013081 (* 1 = 0.013081 loss)
    I0515 03:07:52.112404 23147 sgd_solver.cpp:106] Iteration 117100, lr = 2.97112e-05
    I0515 03:08:00.249158 23147 solver.cpp:228] Iteration 117200, loss = 0.0498291
    I0515 03:08:00.249318 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:08:00.249363 23147 solver.cpp:244]     Train net output #1: loss_c = 0.049829 (* 1 = 0.049829 loss)
    I0515 03:08:00.249388 23147 sgd_solver.cpp:106] Iteration 117200, lr = 2.96937e-05
    I0515 03:08:08.389271 23147 solver.cpp:228] Iteration 117300, loss = 0.0219627
    I0515 03:08:08.389320 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:08:08.389343 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0219626 (* 1 = 0.0219626 loss)
    I0515 03:08:08.389356 23147 sgd_solver.cpp:106] Iteration 117300, lr = 2.96762e-05
    I0515 03:08:16.531000 23147 solver.cpp:228] Iteration 117400, loss = 0.00877439
    I0515 03:08:16.531050 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:08:16.531070 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0087743 (* 1 = 0.0087743 loss)
    I0515 03:08:16.531085 23147 sgd_solver.cpp:106] Iteration 117400, lr = 2.96588e-05
    I0515 03:08:24.672821 23147 solver.cpp:228] Iteration 117500, loss = 0.02232
    I0515 03:08:24.672873 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:08:24.672893 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0223199 (* 1 = 0.0223199 loss)
    I0515 03:08:24.672907 23147 sgd_solver.cpp:106] Iteration 117500, lr = 2.96413e-05
    I0515 03:08:32.812758 23147 solver.cpp:228] Iteration 117600, loss = 0.0124868
    I0515 03:08:32.812950 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:08:32.813001 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0124868 (* 1 = 0.0124868 loss)
    I0515 03:08:32.813016 23147 sgd_solver.cpp:106] Iteration 117600, lr = 2.96239e-05
    I0515 03:08:40.919749 23147 solver.cpp:228] Iteration 117700, loss = 0.0190371
    I0515 03:08:40.919798 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:08:40.919818 23147 solver.cpp:244]     Train net output #1: loss_c = 0.019037 (* 1 = 0.019037 loss)
    I0515 03:08:40.919833 23147 sgd_solver.cpp:106] Iteration 117700, lr = 2.96065e-05
    I0515 03:08:49.060093 23147 solver.cpp:228] Iteration 117800, loss = 0.0100214
    I0515 03:08:49.060144 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:08:49.060164 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0100213 (* 1 = 0.0100213 loss)
    I0515 03:08:49.060179 23147 sgd_solver.cpp:106] Iteration 117800, lr = 2.95891e-05
    I0515 03:08:57.196717 23147 solver.cpp:228] Iteration 117900, loss = 0.0199517
    I0515 03:08:57.196769 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:08:57.196789 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0199516 (* 1 = 0.0199516 loss)
    I0515 03:08:57.196805 23147 sgd_solver.cpp:106] Iteration 117900, lr = 2.95718e-05
    I0515 03:09:05.223294 23147 solver.cpp:337] Iteration 118000, Testing net (#0)
    I0515 03:09:09.585471 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.814083
    I0515 03:09:09.585537 23147 solver.cpp:404]     Test net output #1: loss_c = 0.574372 (* 1 = 0.574372 loss)
    I0515 03:09:09.636982 23147 solver.cpp:228] Iteration 118000, loss = 0.00925835
    I0515 03:09:09.637032 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:09:09.637053 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00925824 (* 1 = 0.00925824 loss)
    I0515 03:09:09.637071 23147 sgd_solver.cpp:106] Iteration 118000, lr = 2.95544e-05
    I0515 03:09:17.777392 23147 solver.cpp:228] Iteration 118100, loss = 0.0209016
    I0515 03:09:17.777443 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:09:17.777463 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0209015 (* 1 = 0.0209015 loss)
    I0515 03:09:17.777478 23147 sgd_solver.cpp:106] Iteration 118100, lr = 2.95371e-05
    I0515 03:09:25.911141 23147 solver.cpp:228] Iteration 118200, loss = 0.0262047
    I0515 03:09:25.911192 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:09:25.911213 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0262045 (* 1 = 0.0262045 loss)
    I0515 03:09:25.911228 23147 sgd_solver.cpp:106] Iteration 118200, lr = 2.95198e-05
    I0515 03:09:34.048797 23147 solver.cpp:228] Iteration 118300, loss = 0.0111976
    I0515 03:09:34.048837 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:09:34.048858 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0111975 (* 1 = 0.0111975 loss)
    I0515 03:09:34.048872 23147 sgd_solver.cpp:106] Iteration 118300, lr = 2.95026e-05
    I0515 03:09:42.185698 23147 solver.cpp:228] Iteration 118400, loss = 0.0176662
    I0515 03:09:42.185791 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:09:42.185817 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0176661 (* 1 = 0.0176661 loss)
    I0515 03:09:42.185832 23147 sgd_solver.cpp:106] Iteration 118400, lr = 2.94853e-05
    I0515 03:09:50.322857 23147 solver.cpp:228] Iteration 118500, loss = 0.0200856
    I0515 03:09:50.322907 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:09:50.322927 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0200855 (* 1 = 0.0200855 loss)
    I0515 03:09:50.322942 23147 sgd_solver.cpp:106] Iteration 118500, lr = 2.94681e-05
    I0515 03:09:58.455699 23147 solver.cpp:228] Iteration 118600, loss = 0.0260537
    I0515 03:09:58.455749 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:09:58.455771 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0260536 (* 1 = 0.0260536 loss)
    I0515 03:09:58.455785 23147 sgd_solver.cpp:106] Iteration 118600, lr = 2.94509e-05
    I0515 03:10:06.551408 23147 solver.cpp:228] Iteration 118700, loss = 0.0154381
    I0515 03:10:06.551451 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:10:06.551470 23147 solver.cpp:244]     Train net output #1: loss_c = 0.015438 (* 1 = 0.015438 loss)
    I0515 03:10:06.551487 23147 sgd_solver.cpp:106] Iteration 118700, lr = 2.94338e-05
    I0515 03:10:14.684846 23147 solver.cpp:228] Iteration 118800, loss = 0.0156401
    I0515 03:10:14.685084 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:10:14.685130 23147 solver.cpp:244]     Train net output #1: loss_c = 0.01564 (* 1 = 0.01564 loss)
    I0515 03:10:14.685155 23147 sgd_solver.cpp:106] Iteration 118800, lr = 2.94166e-05
    I0515 03:10:22.824277 23147 solver.cpp:228] Iteration 118900, loss = 0.00798178
    I0515 03:10:22.824322 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:10:22.824343 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00798168 (* 1 = 0.00798168 loss)
    I0515 03:10:22.824357 23147 sgd_solver.cpp:106] Iteration 118900, lr = 2.93995e-05
    I0515 03:10:30.882139 23147 solver.cpp:337] Iteration 119000, Testing net (#0)
    I0515 03:10:35.287065 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.806833
    I0515 03:10:35.287125 23147 solver.cpp:404]     Test net output #1: loss_c = 0.547584 (* 1 = 0.547584 loss)
    I0515 03:10:35.338814 23147 solver.cpp:228] Iteration 119000, loss = 0.0201325
    I0515 03:10:35.338841 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:10:35.338860 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0201324 (* 1 = 0.0201324 loss)
    I0515 03:10:35.338879 23147 sgd_solver.cpp:106] Iteration 119000, lr = 2.93824e-05
    I0515 03:10:43.432204 23147 solver.cpp:228] Iteration 119100, loss = 0.0220896
    I0515 03:10:43.432255 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:10:43.432284 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0220895 (* 1 = 0.0220895 loss)
    I0515 03:10:43.432306 23147 sgd_solver.cpp:106] Iteration 119100, lr = 2.93654e-05
    I0515 03:10:51.520313 23147 solver.cpp:228] Iteration 119200, loss = 0.0273024
    I0515 03:10:51.520576 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:10:51.520622 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0273023 (* 1 = 0.0273023 loss)
    I0515 03:10:51.520647 23147 sgd_solver.cpp:106] Iteration 119200, lr = 2.93483e-05
    I0515 03:10:59.662044 23147 solver.cpp:228] Iteration 119300, loss = 0.0479208
    I0515 03:10:59.662096 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 03:10:59.662116 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0479207 (* 1 = 0.0479207 loss)
    I0515 03:10:59.662130 23147 sgd_solver.cpp:106] Iteration 119300, lr = 2.93313e-05
    I0515 03:11:07.790962 23147 solver.cpp:228] Iteration 119400, loss = 0.00619745
    I0515 03:11:07.791018 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:11:07.791043 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00619735 (* 1 = 0.00619735 loss)
    I0515 03:11:07.791057 23147 sgd_solver.cpp:106] Iteration 119400, lr = 2.93143e-05
    I0515 03:11:15.927618 23147 solver.cpp:228] Iteration 119500, loss = 0.0634129
    I0515 03:11:15.927670 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 03:11:15.927690 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0634128 (* 1 = 0.0634128 loss)
    I0515 03:11:15.927703 23147 sgd_solver.cpp:106] Iteration 119500, lr = 2.92973e-05
    I0515 03:11:24.062201 23147 solver.cpp:228] Iteration 119600, loss = 0.00657318
    I0515 03:11:24.062449 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:11:24.062496 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00657308 (* 1 = 0.00657308 loss)
    I0515 03:11:24.062521 23147 sgd_solver.cpp:106] Iteration 119600, lr = 2.92803e-05
    I0515 03:11:32.204828 23147 solver.cpp:228] Iteration 119700, loss = 0.018211
    I0515 03:11:32.204872 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:11:32.204892 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0182109 (* 1 = 0.0182109 loss)
    I0515 03:11:32.204907 23147 sgd_solver.cpp:106] Iteration 119700, lr = 2.92634e-05
    I0515 03:11:40.337975 23147 solver.cpp:228] Iteration 119800, loss = 0.0161973
    I0515 03:11:40.338014 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:11:40.338034 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0161973 (* 1 = 0.0161973 loss)
    I0515 03:11:40.338048 23147 sgd_solver.cpp:106] Iteration 119800, lr = 2.92465e-05
    I0515 03:11:48.430496 23147 solver.cpp:228] Iteration 119900, loss = 0.0208772
    I0515 03:11:48.430557 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:11:48.430586 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0208771 (* 1 = 0.0208771 loss)
    I0515 03:11:48.430608 23147 sgd_solver.cpp:106] Iteration 119900, lr = 2.92296e-05
    I0515 03:11:56.436586 23147 solver.cpp:337] Iteration 120000, Testing net (#0)
    I0515 03:12:00.794008 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.809
    I0515 03:12:00.794076 23147 solver.cpp:404]     Test net output #1: loss_c = 0.559937 (* 1 = 0.559937 loss)
    I0515 03:12:00.849154 23147 solver.cpp:228] Iteration 120000, loss = 0.00786712
    I0515 03:12:00.849189 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:12:00.849216 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00786704 (* 1 = 0.00786704 loss)
    I0515 03:12:00.849241 23147 sgd_solver.cpp:106] Iteration 120000, lr = 2.92128e-05
    I0515 03:12:08.945334 23147 solver.cpp:228] Iteration 120100, loss = 0.0204678
    I0515 03:12:08.945391 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:12:08.945420 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0204677 (* 1 = 0.0204677 loss)
    I0515 03:12:08.945441 23147 sgd_solver.cpp:106] Iteration 120100, lr = 2.91959e-05
    I0515 03:12:17.033253 23147 solver.cpp:228] Iteration 120200, loss = 0.0232296
    I0515 03:12:17.033308 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:12:17.033336 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0232295 (* 1 = 0.0232295 loss)
    I0515 03:12:17.033359 23147 sgd_solver.cpp:106] Iteration 120200, lr = 2.91791e-05
    I0515 03:12:25.120009 23147 solver.cpp:228] Iteration 120300, loss = 0.0235955
    I0515 03:12:25.120064 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:12:25.120093 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0235954 (* 1 = 0.0235954 loss)
    I0515 03:12:25.120115 23147 sgd_solver.cpp:106] Iteration 120300, lr = 2.91623e-05
    I0515 03:12:33.214169 23147 solver.cpp:228] Iteration 120400, loss = 0.0109638
    I0515 03:12:33.214282 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:12:33.214311 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0109637 (* 1 = 0.0109637 loss)
    I0515 03:12:33.214334 23147 sgd_solver.cpp:106] Iteration 120400, lr = 2.91455e-05
    I0515 03:12:41.328652 23147 solver.cpp:228] Iteration 120500, loss = 0.00925622
    I0515 03:12:41.328704 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:12:41.328725 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00925615 (* 1 = 0.00925615 loss)
    I0515 03:12:41.328739 23147 sgd_solver.cpp:106] Iteration 120500, lr = 2.91288e-05
    I0515 03:12:49.464357 23147 solver.cpp:228] Iteration 120600, loss = 0.0156715
    I0515 03:12:49.464407 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:12:49.464427 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0156714 (* 1 = 0.0156714 loss)
    I0515 03:12:49.464442 23147 sgd_solver.cpp:106] Iteration 120600, lr = 2.9112e-05
    I0515 03:12:57.579818 23147 solver.cpp:228] Iteration 120700, loss = 0.0187696
    I0515 03:12:57.579879 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:12:57.579908 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0187695 (* 1 = 0.0187695 loss)
    I0515 03:12:57.579929 23147 sgd_solver.cpp:106] Iteration 120700, lr = 2.90953e-05
    I0515 03:13:05.670008 23147 solver.cpp:228] Iteration 120800, loss = 0.0249635
    I0515 03:13:05.670255 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:13:05.670303 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0249634 (* 1 = 0.0249634 loss)
    I0515 03:13:05.670327 23147 sgd_solver.cpp:106] Iteration 120800, lr = 2.90786e-05
    I0515 03:13:13.803136 23147 solver.cpp:228] Iteration 120900, loss = 0.0206559
    I0515 03:13:13.803184 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:13:13.803202 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0206558 (* 1 = 0.0206558 loss)
    I0515 03:13:13.803217 23147 sgd_solver.cpp:106] Iteration 120900, lr = 2.9062e-05
    I0515 03:13:21.856254 23147 solver.cpp:337] Iteration 121000, Testing net (#0)
    I0515 03:13:26.269147 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.821583
    I0515 03:13:26.269198 23147 solver.cpp:404]     Test net output #1: loss_c = 0.540322 (* 1 = 0.540322 loss)
    I0515 03:13:26.324306 23147 solver.cpp:228] Iteration 121000, loss = 0.0301567
    I0515 03:13:26.324368 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:13:26.324396 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0301566 (* 1 = 0.0301566 loss)
    I0515 03:13:26.324424 23147 sgd_solver.cpp:106] Iteration 121000, lr = 2.90453e-05
    I0515 03:13:34.461447 23147 solver.cpp:228] Iteration 121100, loss = 0.0317722
    I0515 03:13:34.461485 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:13:34.461506 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0317722 (* 1 = 0.0317722 loss)
    I0515 03:13:34.461521 23147 sgd_solver.cpp:106] Iteration 121100, lr = 2.90287e-05
    I0515 03:13:42.567735 23147 solver.cpp:228] Iteration 121200, loss = 0.0215357
    I0515 03:13:42.567833 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:13:42.567853 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0215356 (* 1 = 0.0215356 loss)
    I0515 03:13:42.567868 23147 sgd_solver.cpp:106] Iteration 121200, lr = 2.90121e-05
    I0515 03:13:50.704609 23147 solver.cpp:228] Iteration 121300, loss = 0.0100873
    I0515 03:13:50.704659 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:13:50.704680 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0100873 (* 1 = 0.0100873 loss)
    I0515 03:13:50.704694 23147 sgd_solver.cpp:106] Iteration 121300, lr = 2.89956e-05
    I0515 03:13:58.844655 23147 solver.cpp:228] Iteration 121400, loss = 0.0103185
    I0515 03:13:58.844705 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:13:58.844725 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0103184 (* 1 = 0.0103184 loss)
    I0515 03:13:58.844739 23147 sgd_solver.cpp:106] Iteration 121400, lr = 2.8979e-05
    I0515 03:14:06.985538 23147 solver.cpp:228] Iteration 121500, loss = 0.0250433
    I0515 03:14:06.985589 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:14:06.985610 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0250432 (* 1 = 0.0250432 loss)
    I0515 03:14:06.985623 23147 sgd_solver.cpp:106] Iteration 121500, lr = 2.89625e-05
    I0515 03:14:15.125233 23147 solver.cpp:228] Iteration 121600, loss = 0.0147177
    I0515 03:14:15.125327 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:14:15.125351 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0147176 (* 1 = 0.0147176 loss)
    I0515 03:14:15.125366 23147 sgd_solver.cpp:106] Iteration 121600, lr = 2.8946e-05
    I0515 03:14:23.264843 23147 solver.cpp:228] Iteration 121700, loss = 0.0581164
    I0515 03:14:23.264889 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:14:23.264909 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0581163 (* 1 = 0.0581163 loss)
    I0515 03:14:23.264924 23147 sgd_solver.cpp:106] Iteration 121700, lr = 2.89295e-05
    I0515 03:14:31.405014 23147 solver.cpp:228] Iteration 121800, loss = 0.0068606
    I0515 03:14:31.405057 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:14:31.405077 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00686053 (* 1 = 0.00686053 loss)
    I0515 03:14:31.405092 23147 sgd_solver.cpp:106] Iteration 121800, lr = 2.8913e-05
    I0515 03:14:39.546589 23147 solver.cpp:228] Iteration 121900, loss = 0.04511
    I0515 03:14:39.546632 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:14:39.546651 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0451099 (* 1 = 0.0451099 loss)
    I0515 03:14:39.546665 23147 sgd_solver.cpp:106] Iteration 121900, lr = 2.88966e-05
    I0515 03:14:47.605676 23147 solver.cpp:337] Iteration 122000, Testing net (#0)
    I0515 03:14:51.986446 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.818667
    I0515 03:14:51.986515 23147 solver.cpp:404]     Test net output #1: loss_c = 0.566504 (* 1 = 0.566504 loss)
    I0515 03:14:52.042708 23147 solver.cpp:228] Iteration 122000, loss = 0.0147793
    I0515 03:14:52.042754 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:14:52.042783 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0147792 (* 1 = 0.0147792 loss)
    I0515 03:14:52.042814 23147 sgd_solver.cpp:106] Iteration 122000, lr = 2.88802e-05
    I0515 03:15:00.182065 23147 solver.cpp:228] Iteration 122100, loss = 0.0149824
    I0515 03:15:00.182108 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:15:00.182128 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0149823 (* 1 = 0.0149823 loss)
    I0515 03:15:00.182143 23147 sgd_solver.cpp:106] Iteration 122100, lr = 2.88638e-05
    I0515 03:15:08.308657 23147 solver.cpp:228] Iteration 122200, loss = 0.0152432
    I0515 03:15:08.308712 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:15:08.308743 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0152432 (* 1 = 0.0152432 loss)
    I0515 03:15:08.308764 23147 sgd_solver.cpp:106] Iteration 122200, lr = 2.88474e-05
    I0515 03:15:16.398504 23147 solver.cpp:228] Iteration 122300, loss = 0.00864708
    I0515 03:15:16.398566 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:15:16.398596 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00864703 (* 1 = 0.00864703 loss)
    I0515 03:15:16.398617 23147 sgd_solver.cpp:106] Iteration 122300, lr = 2.8831e-05
    I0515 03:15:24.487524 23147 solver.cpp:228] Iteration 122400, loss = 0.0122732
    I0515 03:15:24.487642 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:15:24.487674 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0122732 (* 1 = 0.0122732 loss)
    I0515 03:15:24.487696 23147 sgd_solver.cpp:106] Iteration 122400, lr = 2.88147e-05
    I0515 03:15:32.577455 23147 solver.cpp:228] Iteration 122500, loss = 0.00953194
    I0515 03:15:32.577517 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:15:32.577546 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0095319 (* 1 = 0.0095319 loss)
    I0515 03:15:32.577567 23147 sgd_solver.cpp:106] Iteration 122500, lr = 2.87984e-05
    I0515 03:15:40.663709 23147 solver.cpp:228] Iteration 122600, loss = 0.0307619
    I0515 03:15:40.663770 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:15:40.663800 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0307619 (* 1 = 0.0307619 loss)
    I0515 03:15:40.663821 23147 sgd_solver.cpp:106] Iteration 122600, lr = 2.87821e-05
    I0515 03:15:48.754777 23147 solver.cpp:228] Iteration 122700, loss = 0.0108182
    I0515 03:15:48.754833 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:15:48.754866 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0108182 (* 1 = 0.0108182 loss)
    I0515 03:15:48.754890 23147 sgd_solver.cpp:106] Iteration 122700, lr = 2.87658e-05
    I0515 03:15:56.848724 23147 solver.cpp:228] Iteration 122800, loss = 0.00938312
    I0515 03:15:56.848943 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:15:56.848974 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00938308 (* 1 = 0.00938308 loss)
    I0515 03:15:56.849025 23147 sgd_solver.cpp:106] Iteration 122800, lr = 2.87496e-05
    I0515 03:16:04.936398 23147 solver.cpp:228] Iteration 122900, loss = 0.0190115
    I0515 03:16:04.936456 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:16:04.936486 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0190114 (* 1 = 0.0190114 loss)
    I0515 03:16:04.936507 23147 sgd_solver.cpp:106] Iteration 122900, lr = 2.87333e-05
    I0515 03:16:12.946892 23147 solver.cpp:337] Iteration 123000, Testing net (#0)
    I0515 03:16:17.314187 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.818084
    I0515 03:16:17.314254 23147 solver.cpp:404]     Test net output #1: loss_c = 0.538526 (* 1 = 0.538526 loss)
    I0515 03:16:17.366016 23147 solver.cpp:228] Iteration 123000, loss = 0.0190025
    I0515 03:16:17.366071 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:16:17.366092 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0190025 (* 1 = 0.0190025 loss)
    I0515 03:16:17.366111 23147 sgd_solver.cpp:106] Iteration 123000, lr = 2.87171e-05
    I0515 03:16:25.462258 23147 solver.cpp:228] Iteration 123100, loss = 0.0112732
    I0515 03:16:25.462314 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:16:25.462343 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0112732 (* 1 = 0.0112732 loss)
    I0515 03:16:25.462365 23147 sgd_solver.cpp:106] Iteration 123100, lr = 2.8701e-05
    I0515 03:16:33.551182 23147 solver.cpp:228] Iteration 123200, loss = 0.00944739
    I0515 03:16:33.551342 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:16:33.551386 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00944735 (* 1 = 0.00944735 loss)
    I0515 03:16:33.551411 23147 sgd_solver.cpp:106] Iteration 123200, lr = 2.86848e-05
    I0515 03:16:41.680971 23147 solver.cpp:228] Iteration 123300, loss = 0.0311839
    I0515 03:16:41.681021 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:16:41.681042 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0311839 (* 1 = 0.0311839 loss)
    I0515 03:16:41.681057 23147 sgd_solver.cpp:106] Iteration 123300, lr = 2.86687e-05
    I0515 03:16:49.816592 23147 solver.cpp:228] Iteration 123400, loss = 0.00702885
    I0515 03:16:49.816642 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:16:49.816664 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00702882 (* 1 = 0.00702882 loss)
    I0515 03:16:49.816679 23147 sgd_solver.cpp:106] Iteration 123400, lr = 2.86525e-05
    I0515 03:16:57.957216 23147 solver.cpp:228] Iteration 123500, loss = 0.00631519
    I0515 03:16:57.957267 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:16:57.957286 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00631516 (* 1 = 0.00631516 loss)
    I0515 03:16:57.957301 23147 sgd_solver.cpp:106] Iteration 123500, lr = 2.86364e-05
    I0515 03:17:06.092519 23147 solver.cpp:228] Iteration 123600, loss = 0.00735927
    I0515 03:17:06.092653 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:17:06.092689 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00735924 (* 1 = 0.00735924 loss)
    I0515 03:17:06.092707 23147 sgd_solver.cpp:106] Iteration 123600, lr = 2.86204e-05
    I0515 03:17:14.232542 23147 solver.cpp:228] Iteration 123700, loss = 0.0459849
    I0515 03:17:14.232583 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:17:14.232601 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0459849 (* 1 = 0.0459849 loss)
    I0515 03:17:14.232616 23147 sgd_solver.cpp:106] Iteration 123700, lr = 2.86043e-05
    I0515 03:17:22.372412 23147 solver.cpp:228] Iteration 123800, loss = 0.0120248
    I0515 03:17:22.372454 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:17:22.372474 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0120248 (* 1 = 0.0120248 loss)
    I0515 03:17:22.372489 23147 sgd_solver.cpp:106] Iteration 123800, lr = 2.85883e-05
    I0515 03:17:30.512135 23147 solver.cpp:228] Iteration 123900, loss = 0.063032
    I0515 03:17:30.512176 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:17:30.512195 23147 solver.cpp:244]     Train net output #1: loss_c = 0.063032 (* 1 = 0.063032 loss)
    I0515 03:17:30.512209 23147 sgd_solver.cpp:106] Iteration 123900, lr = 2.85723e-05
    I0515 03:17:38.563204 23147 solver.cpp:337] Iteration 124000, Testing net (#0)
    I0515 03:17:42.940999 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.812083
    I0515 03:17:42.941061 23147 solver.cpp:404]     Test net output #1: loss_c = 0.536816 (* 1 = 0.536816 loss)
    I0515 03:17:42.997047 23147 solver.cpp:228] Iteration 124000, loss = 0.0108194
    I0515 03:17:42.997097 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:17:42.997125 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0108194 (* 1 = 0.0108194 loss)
    I0515 03:17:42.997149 23147 sgd_solver.cpp:106] Iteration 124000, lr = 2.85563e-05
    I0515 03:17:51.089184 23147 solver.cpp:228] Iteration 124100, loss = 0.00721722
    I0515 03:17:51.089244 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:17:51.089274 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00721721 (* 1 = 0.00721721 loss)
    I0515 03:17:51.089296 23147 sgd_solver.cpp:106] Iteration 124100, lr = 2.85403e-05
    I0515 03:17:59.175976 23147 solver.cpp:228] Iteration 124200, loss = 0.0143857
    I0515 03:17:59.176036 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:17:59.176065 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0143857 (* 1 = 0.0143857 loss)
    I0515 03:17:59.176087 23147 sgd_solver.cpp:106] Iteration 124200, lr = 2.85243e-05
    I0515 03:18:07.266547 23147 solver.cpp:228] Iteration 124300, loss = 0.0181631
    I0515 03:18:07.266607 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:18:07.266638 23147 solver.cpp:244]     Train net output #1: loss_c = 0.018163 (* 1 = 0.018163 loss)
    I0515 03:18:07.266659 23147 sgd_solver.cpp:106] Iteration 124300, lr = 2.85084e-05
    I0515 03:18:15.359247 23147 solver.cpp:228] Iteration 124400, loss = 0.0221715
    I0515 03:18:15.359395 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:18:15.359441 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0221715 (* 1 = 0.0221715 loss)
    I0515 03:18:15.359464 23147 sgd_solver.cpp:106] Iteration 124400, lr = 2.84925e-05
    I0515 03:18:23.483654 23147 solver.cpp:228] Iteration 124500, loss = 0.0112866
    I0515 03:18:23.483697 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:18:23.483719 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0112866 (* 1 = 0.0112866 loss)
    I0515 03:18:23.483734 23147 sgd_solver.cpp:106] Iteration 124500, lr = 2.84766e-05
    I0515 03:18:31.623898 23147 solver.cpp:228] Iteration 124600, loss = 0.0204552
    I0515 03:18:31.623946 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:18:31.623968 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0204552 (* 1 = 0.0204552 loss)
    I0515 03:18:31.623983 23147 sgd_solver.cpp:106] Iteration 124600, lr = 2.84607e-05
    I0515 03:18:39.764322 23147 solver.cpp:228] Iteration 124700, loss = 0.0503657
    I0515 03:18:39.764369 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:18:39.764390 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0503657 (* 1 = 0.0503657 loss)
    I0515 03:18:39.764405 23147 sgd_solver.cpp:106] Iteration 124700, lr = 2.84449e-05
    I0515 03:18:47.897430 23147 solver.cpp:228] Iteration 124800, loss = 0.00906221
    I0515 03:18:47.897644 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:18:47.897694 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00906219 (* 1 = 0.00906219 loss)
    I0515 03:18:47.897717 23147 sgd_solver.cpp:106] Iteration 124800, lr = 2.84291e-05
    I0515 03:18:55.984041 23147 solver.cpp:228] Iteration 124900, loss = 0.02757
    I0515 03:18:55.984097 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:18:55.984127 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0275699 (* 1 = 0.0275699 loss)
    I0515 03:18:55.984148 23147 sgd_solver.cpp:106] Iteration 124900, lr = 2.84133e-05
    I0515 03:19:03.990030 23147 solver.cpp:337] Iteration 125000, Testing net (#0)
    I0515 03:19:08.364400 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.81175
    I0515 03:19:08.364457 23147 solver.cpp:404]     Test net output #1: loss_c = 0.550464 (* 1 = 0.550464 loss)
    I0515 03:19:08.424732 23147 solver.cpp:228] Iteration 125000, loss = 0.021005
    I0515 03:19:08.424805 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:19:08.424836 23147 solver.cpp:244]     Train net output #1: loss_c = 0.021005 (* 1 = 0.021005 loss)
    I0515 03:19:08.424861 23147 sgd_solver.cpp:106] Iteration 125000, lr = 2.83975e-05
    I0515 03:19:16.563036 23147 solver.cpp:228] Iteration 125100, loss = 0.0256552
    I0515 03:19:16.563087 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:19:16.563112 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0256552 (* 1 = 0.0256552 loss)
    I0515 03:19:16.563127 23147 sgd_solver.cpp:106] Iteration 125100, lr = 2.83817e-05
    I0515 03:19:24.700448 23147 solver.cpp:228] Iteration 125200, loss = 0.00848287
    I0515 03:19:24.700551 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:19:24.700572 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00848285 (* 1 = 0.00848285 loss)
    I0515 03:19:24.700588 23147 sgd_solver.cpp:106] Iteration 125200, lr = 2.8366e-05
    I0515 03:19:32.833967 23147 solver.cpp:228] Iteration 125300, loss = 0.00662432
    I0515 03:19:32.834010 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:19:32.834029 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0066243 (* 1 = 0.0066243 loss)
    I0515 03:19:32.834044 23147 sgd_solver.cpp:106] Iteration 125300, lr = 2.83502e-05
    I0515 03:19:40.948752 23147 solver.cpp:228] Iteration 125400, loss = 0.00540341
    I0515 03:19:40.948797 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:19:40.948817 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00540339 (* 1 = 0.00540339 loss)
    I0515 03:19:40.948832 23147 sgd_solver.cpp:106] Iteration 125400, lr = 2.83345e-05
    I0515 03:19:49.086302 23147 solver.cpp:228] Iteration 125500, loss = 0.0191825
    I0515 03:19:49.086344 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:19:49.086364 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0191824 (* 1 = 0.0191824 loss)
    I0515 03:19:49.086377 23147 sgd_solver.cpp:106] Iteration 125500, lr = 2.83188e-05
    I0515 03:19:57.205621 23147 solver.cpp:228] Iteration 125600, loss = 0.0110009
    I0515 03:19:57.205713 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:19:57.205741 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0110009 (* 1 = 0.0110009 loss)
    I0515 03:19:57.205763 23147 sgd_solver.cpp:106] Iteration 125600, lr = 2.83032e-05
    I0515 03:20:05.291064 23147 solver.cpp:228] Iteration 125700, loss = 0.0158663
    I0515 03:20:05.291116 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:20:05.291146 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0158663 (* 1 = 0.0158663 loss)
    I0515 03:20:05.291167 23147 sgd_solver.cpp:106] Iteration 125700, lr = 2.82875e-05
    I0515 03:20:13.384028 23147 solver.cpp:228] Iteration 125800, loss = 0.00853816
    I0515 03:20:13.384088 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:20:13.384116 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00853813 (* 1 = 0.00853813 loss)
    I0515 03:20:13.384138 23147 sgd_solver.cpp:106] Iteration 125800, lr = 2.82719e-05
    I0515 03:20:21.473104 23147 solver.cpp:228] Iteration 125900, loss = 0.012691
    I0515 03:20:21.473161 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:20:21.473189 23147 solver.cpp:244]     Train net output #1: loss_c = 0.012691 (* 1 = 0.012691 loss)
    I0515 03:20:21.473212 23147 sgd_solver.cpp:106] Iteration 125900, lr = 2.82563e-05
    I0515 03:20:29.480962 23147 solver.cpp:337] Iteration 126000, Testing net (#0)
    I0515 03:20:33.890624 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.816833
    I0515 03:20:33.890672 23147 solver.cpp:404]     Test net output #1: loss_c = 0.552501 (* 1 = 0.552501 loss)
    I0515 03:20:33.942018 23147 solver.cpp:228] Iteration 126000, loss = 0.0233539
    I0515 03:20:33.942056 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:20:33.942075 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0233539 (* 1 = 0.0233539 loss)
    I0515 03:20:33.942095 23147 sgd_solver.cpp:106] Iteration 126000, lr = 2.82407e-05
    I0515 03:20:42.065811 23147 solver.cpp:228] Iteration 126100, loss = 0.0125326
    I0515 03:20:42.065856 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:20:42.065876 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0125326 (* 1 = 0.0125326 loss)
    I0515 03:20:42.065891 23147 sgd_solver.cpp:106] Iteration 126100, lr = 2.82252e-05
    I0515 03:20:50.206032 23147 solver.cpp:228] Iteration 126200, loss = 0.0140967
    I0515 03:20:50.206075 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:20:50.206097 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0140967 (* 1 = 0.0140967 loss)
    I0515 03:20:50.206112 23147 sgd_solver.cpp:106] Iteration 126200, lr = 2.82096e-05
    I0515 03:20:58.345027 23147 solver.cpp:228] Iteration 126300, loss = 0.0182703
    I0515 03:20:58.345079 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:20:58.345099 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0182703 (* 1 = 0.0182703 loss)
    I0515 03:20:58.345114 23147 sgd_solver.cpp:106] Iteration 126300, lr = 2.81941e-05
    I0515 03:21:06.481966 23147 solver.cpp:228] Iteration 126400, loss = 0.0144677
    I0515 03:21:06.482046 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:21:06.482069 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0144676 (* 1 = 0.0144676 loss)
    I0515 03:21:06.482084 23147 sgd_solver.cpp:106] Iteration 126400, lr = 2.81786e-05
    I0515 03:21:14.610574 23147 solver.cpp:228] Iteration 126500, loss = 0.00864068
    I0515 03:21:14.610618 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:21:14.610640 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00864066 (* 1 = 0.00864066 loss)
    I0515 03:21:14.610656 23147 sgd_solver.cpp:106] Iteration 126500, lr = 2.81631e-05
    I0515 03:21:22.744596 23147 solver.cpp:228] Iteration 126600, loss = 0.0178016
    I0515 03:21:22.744638 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:21:22.744657 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0178016 (* 1 = 0.0178016 loss)
    I0515 03:21:22.744673 23147 sgd_solver.cpp:106] Iteration 126600, lr = 2.81476e-05
    I0515 03:21:30.879063 23147 solver.cpp:228] Iteration 126700, loss = 0.0321663
    I0515 03:21:30.879112 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:21:30.879132 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0321663 (* 1 = 0.0321663 loss)
    I0515 03:21:30.879147 23147 sgd_solver.cpp:106] Iteration 126700, lr = 2.81322e-05
    I0515 03:21:39.006451 23147 solver.cpp:228] Iteration 126800, loss = 0.0167433
    I0515 03:21:39.006579 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:21:39.006625 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0167433 (* 1 = 0.0167433 loss)
    I0515 03:21:39.006650 23147 sgd_solver.cpp:106] Iteration 126800, lr = 2.81168e-05
    I0515 03:21:47.156929 23147 solver.cpp:228] Iteration 126900, loss = 0.0258299
    I0515 03:21:47.156972 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:21:47.156991 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0258299 (* 1 = 0.0258299 loss)
    I0515 03:21:47.157006 23147 sgd_solver.cpp:106] Iteration 126900, lr = 2.81014e-05
    I0515 03:21:55.215570 23147 solver.cpp:337] Iteration 127000, Testing net (#0)
    I0515 03:21:59.556300 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.816417
    I0515 03:21:59.556354 23147 solver.cpp:404]     Test net output #1: loss_c = 0.557237 (* 1 = 0.557237 loss)
    I0515 03:21:59.611680 23147 solver.cpp:228] Iteration 127000, loss = 0.0172652
    I0515 03:21:59.611752 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:21:59.611783 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0172652 (* 1 = 0.0172652 loss)
    I0515 03:21:59.611815 23147 sgd_solver.cpp:106] Iteration 127000, lr = 2.8086e-05
    I0515 03:22:07.743456 23147 solver.cpp:228] Iteration 127100, loss = 0.022174
    I0515 03:22:07.743505 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:22:07.743531 23147 solver.cpp:244]     Train net output #1: loss_c = 0.022174 (* 1 = 0.022174 loss)
    I0515 03:22:07.743546 23147 sgd_solver.cpp:106] Iteration 127100, lr = 2.80706e-05
    I0515 03:22:15.878187 23147 solver.cpp:228] Iteration 127200, loss = 0.102331
    I0515 03:22:15.878417 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:22:15.878464 23147 solver.cpp:244]     Train net output #1: loss_c = 0.102331 (* 1 = 0.102331 loss)
    I0515 03:22:15.878489 23147 sgd_solver.cpp:106] Iteration 127200, lr = 2.80553e-05
    I0515 03:22:24.019117 23147 solver.cpp:228] Iteration 127300, loss = 0.0217106
    I0515 03:22:24.019158 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:22:24.019181 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0217106 (* 1 = 0.0217106 loss)
    I0515 03:22:24.019194 23147 sgd_solver.cpp:106] Iteration 127300, lr = 2.80399e-05
    I0515 03:22:32.157372 23147 solver.cpp:228] Iteration 127400, loss = 0.011978
    I0515 03:22:32.157418 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:22:32.157438 23147 solver.cpp:244]     Train net output #1: loss_c = 0.011978 (* 1 = 0.011978 loss)
    I0515 03:22:32.157452 23147 sgd_solver.cpp:106] Iteration 127400, lr = 2.80246e-05
    I0515 03:22:40.277456 23147 solver.cpp:228] Iteration 127500, loss = 0.00761864
    I0515 03:22:40.277508 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:22:40.277529 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00761861 (* 1 = 0.00761861 loss)
    I0515 03:22:40.277544 23147 sgd_solver.cpp:106] Iteration 127500, lr = 2.80093e-05
    I0515 03:22:48.419505 23147 solver.cpp:228] Iteration 127600, loss = 0.0339275
    I0515 03:22:48.419610 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:22:48.419630 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0339274 (* 1 = 0.0339274 loss)
    I0515 03:22:48.419644 23147 sgd_solver.cpp:106] Iteration 127600, lr = 2.79941e-05
    I0515 03:22:56.559725 23147 solver.cpp:228] Iteration 127700, loss = 0.0231619
    I0515 03:22:56.559770 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:22:56.559789 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0231619 (* 1 = 0.0231619 loss)
    I0515 03:22:56.559805 23147 sgd_solver.cpp:106] Iteration 127700, lr = 2.79788e-05
    I0515 03:23:04.701952 23147 solver.cpp:228] Iteration 127800, loss = 0.0402539
    I0515 03:23:04.702003 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:23:04.702023 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0402539 (* 1 = 0.0402539 loss)
    I0515 03:23:04.702036 23147 sgd_solver.cpp:106] Iteration 127800, lr = 2.79636e-05
    I0515 03:23:12.788030 23147 solver.cpp:228] Iteration 127900, loss = 0.0255839
    I0515 03:23:12.788077 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:23:12.788097 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0255839 (* 1 = 0.0255839 loss)
    I0515 03:23:12.788112 23147 sgd_solver.cpp:106] Iteration 127900, lr = 2.79484e-05
    I0515 03:23:20.848021 23147 solver.cpp:337] Iteration 128000, Testing net (#0)
    I0515 03:23:25.243221 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.816167
    I0515 03:23:25.243279 23147 solver.cpp:404]     Test net output #1: loss_c = 0.561477 (* 1 = 0.561477 loss)
    I0515 03:23:25.298542 23147 solver.cpp:228] Iteration 128000, loss = 0.0151027
    I0515 03:23:25.298609 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:23:25.298640 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0151027 (* 1 = 0.0151027 loss)
    I0515 03:23:25.298667 23147 sgd_solver.cpp:106] Iteration 128000, lr = 2.79332e-05
    I0515 03:23:33.394824 23147 solver.cpp:228] Iteration 128100, loss = 0.0225126
    I0515 03:23:33.394875 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:23:33.394903 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0225126 (* 1 = 0.0225126 loss)
    I0515 03:23:33.394924 23147 sgd_solver.cpp:106] Iteration 128100, lr = 2.7918e-05
    I0515 03:23:41.485308 23147 solver.cpp:228] Iteration 128200, loss = 0.0301799
    I0515 03:23:41.485365 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:23:41.485395 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0301799 (* 1 = 0.0301799 loss)
    I0515 03:23:41.485419 23147 sgd_solver.cpp:106] Iteration 128200, lr = 2.79029e-05
    I0515 03:23:49.576155 23147 solver.cpp:228] Iteration 128300, loss = 0.00427716
    I0515 03:23:49.576208 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:23:49.576236 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00427711 (* 1 = 0.00427711 loss)
    I0515 03:23:49.576257 23147 sgd_solver.cpp:106] Iteration 128300, lr = 2.78877e-05
    I0515 03:23:57.665601 23147 solver.cpp:228] Iteration 128400, loss = 0.0114558
    I0515 03:23:57.665752 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:23:57.665797 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0114558 (* 1 = 0.0114558 loss)
    I0515 03:23:57.665822 23147 sgd_solver.cpp:106] Iteration 128400, lr = 2.78726e-05
    I0515 03:24:05.797832 23147 solver.cpp:228] Iteration 128500, loss = 0.0459135
    I0515 03:24:05.797879 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:24:05.797900 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0459135 (* 1 = 0.0459135 loss)
    I0515 03:24:05.797915 23147 sgd_solver.cpp:106] Iteration 128500, lr = 2.78575e-05
    I0515 03:24:13.934223 23147 solver.cpp:228] Iteration 128600, loss = 0.027144
    I0515 03:24:13.934272 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:24:13.934293 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0271439 (* 1 = 0.0271439 loss)
    I0515 03:24:13.934306 23147 sgd_solver.cpp:106] Iteration 128600, lr = 2.78425e-05
    I0515 03:24:22.070017 23147 solver.cpp:228] Iteration 128700, loss = 0.0113984
    I0515 03:24:22.070061 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:24:22.070081 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0113983 (* 1 = 0.0113983 loss)
    I0515 03:24:22.070096 23147 sgd_solver.cpp:106] Iteration 128700, lr = 2.78274e-05
    I0515 03:24:30.209419 23147 solver.cpp:228] Iteration 128800, loss = 0.0162925
    I0515 03:24:30.209566 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:24:30.209611 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0162924 (* 1 = 0.0162924 loss)
    I0515 03:24:30.209635 23147 sgd_solver.cpp:106] Iteration 128800, lr = 2.78124e-05
    I0515 03:24:38.304312 23147 solver.cpp:228] Iteration 128900, loss = 0.0153693
    I0515 03:24:38.304369 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:24:38.304399 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0153693 (* 1 = 0.0153693 loss)
    I0515 03:24:38.304421 23147 sgd_solver.cpp:106] Iteration 128900, lr = 2.77973e-05
    I0515 03:24:46.309698 23147 solver.cpp:337] Iteration 129000, Testing net (#0)
    I0515 03:24:50.698678 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.82625
    I0515 03:24:50.698739 23147 solver.cpp:404]     Test net output #1: loss_c = 0.560411 (* 1 = 0.560411 loss)
    I0515 03:24:50.753876 23147 solver.cpp:228] Iteration 129000, loss = 0.0144112
    I0515 03:24:50.753937 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:24:50.753968 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0144112 (* 1 = 0.0144112 loss)
    I0515 03:24:50.753993 23147 sgd_solver.cpp:106] Iteration 129000, lr = 2.77823e-05
    I0515 03:24:58.893926 23147 solver.cpp:228] Iteration 129100, loss = 0.0235876
    I0515 03:24:58.893980 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:24:58.893999 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0235876 (* 1 = 0.0235876 loss)
    I0515 03:24:58.894016 23147 sgd_solver.cpp:106] Iteration 129100, lr = 2.77674e-05
    I0515 03:25:07.032361 23147 solver.cpp:228] Iteration 129200, loss = 0.00918102
    I0515 03:25:07.032565 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:25:07.032587 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00918099 (* 1 = 0.00918099 loss)
    I0515 03:25:07.032601 23147 sgd_solver.cpp:106] Iteration 129200, lr = 2.77524e-05
    I0515 03:25:15.135519 23147 solver.cpp:228] Iteration 129300, loss = 0.048805
    I0515 03:25:15.135565 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:25:15.135583 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0488049 (* 1 = 0.0488049 loss)
    I0515 03:25:15.135598 23147 sgd_solver.cpp:106] Iteration 129300, lr = 2.77375e-05
    I0515 03:25:23.268750 23147 solver.cpp:228] Iteration 129400, loss = 0.0171715
    I0515 03:25:23.268796 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:25:23.268815 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0171715 (* 1 = 0.0171715 loss)
    I0515 03:25:23.268831 23147 sgd_solver.cpp:106] Iteration 129400, lr = 2.77225e-05
    I0515 03:25:31.401406 23147 solver.cpp:228] Iteration 129500, loss = 0.00593257
    I0515 03:25:31.401449 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:25:31.401469 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00593254 (* 1 = 0.00593254 loss)
    I0515 03:25:31.401484 23147 sgd_solver.cpp:106] Iteration 129500, lr = 2.77076e-05
    I0515 03:25:39.534437 23147 solver.cpp:228] Iteration 129600, loss = 0.0236713
    I0515 03:25:39.534523 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:25:39.534543 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0236713 (* 1 = 0.0236713 loss)
    I0515 03:25:39.534557 23147 sgd_solver.cpp:106] Iteration 129600, lr = 2.76927e-05
    I0515 03:25:47.670217 23147 solver.cpp:228] Iteration 129700, loss = 0.0195947
    I0515 03:25:47.670264 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:25:47.670284 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0195946 (* 1 = 0.0195946 loss)
    I0515 03:25:47.670298 23147 sgd_solver.cpp:106] Iteration 129700, lr = 2.76779e-05
    I0515 03:25:55.812995 23147 solver.cpp:228] Iteration 129800, loss = 0.0135228
    I0515 03:25:55.813042 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:25:55.813062 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0135228 (* 1 = 0.0135228 loss)
    I0515 03:25:55.813076 23147 sgd_solver.cpp:106] Iteration 129800, lr = 2.7663e-05
    I0515 03:26:03.951455 23147 solver.cpp:228] Iteration 129900, loss = 0.0174609
    I0515 03:26:03.951503 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:26:03.951524 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0174608 (* 1 = 0.0174608 loss)
    I0515 03:26:03.951539 23147 sgd_solver.cpp:106] Iteration 129900, lr = 2.76482e-05
    I0515 03:26:12.012739 23147 solver.cpp:337] Iteration 130000, Testing net (#0)
    I0515 03:26:16.409710 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.824834
    I0515 03:26:16.409757 23147 solver.cpp:404]     Test net output #1: loss_c = 0.522696 (* 1 = 0.522696 loss)
    I0515 03:26:16.461949 23147 solver.cpp:228] Iteration 130000, loss = 0.025727
    I0515 03:26:16.461999 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:26:16.462020 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0257269 (* 1 = 0.0257269 loss)
    I0515 03:26:16.462038 23147 sgd_solver.cpp:106] Iteration 130000, lr = 2.76334e-05
    I0515 03:26:24.601600 23147 solver.cpp:228] Iteration 130100, loss = 0.025133
    I0515 03:26:24.601642 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:26:24.601663 23147 solver.cpp:244]     Train net output #1: loss_c = 0.025133 (* 1 = 0.025133 loss)
    I0515 03:26:24.601678 23147 sgd_solver.cpp:106] Iteration 130100, lr = 2.76186e-05
    I0515 03:26:32.739794 23147 solver.cpp:228] Iteration 130200, loss = 0.0167178
    I0515 03:26:32.739845 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:26:32.739864 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0167178 (* 1 = 0.0167178 loss)
    I0515 03:26:32.739879 23147 sgd_solver.cpp:106] Iteration 130200, lr = 2.76038e-05
    I0515 03:26:40.876272 23147 solver.cpp:228] Iteration 130300, loss = 0.0152388
    I0515 03:26:40.876318 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:26:40.876338 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0152388 (* 1 = 0.0152388 loss)
    I0515 03:26:40.876351 23147 sgd_solver.cpp:106] Iteration 130300, lr = 2.7589e-05
    I0515 03:26:49.012248 23147 solver.cpp:228] Iteration 130400, loss = 0.0221348
    I0515 03:26:49.012501 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:26:49.012547 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0221348 (* 1 = 0.0221348 loss)
    I0515 03:26:49.012570 23147 sgd_solver.cpp:106] Iteration 130400, lr = 2.75743e-05
    I0515 03:26:57.109164 23147 solver.cpp:228] Iteration 130500, loss = 0.0149497
    I0515 03:26:57.109223 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:26:57.109253 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0149497 (* 1 = 0.0149497 loss)
    I0515 03:26:57.109274 23147 sgd_solver.cpp:106] Iteration 130500, lr = 2.75596e-05
    I0515 03:27:05.196152 23147 solver.cpp:228] Iteration 130600, loss = 0.00891844
    I0515 03:27:05.196205 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:27:05.196234 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00891841 (* 1 = 0.00891841 loss)
    I0515 03:27:05.196256 23147 sgd_solver.cpp:106] Iteration 130600, lr = 2.75449e-05
    I0515 03:27:13.283349 23147 solver.cpp:228] Iteration 130700, loss = 0.0156679
    I0515 03:27:13.283404 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:27:13.283433 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0156678 (* 1 = 0.0156678 loss)
    I0515 03:27:13.283455 23147 sgd_solver.cpp:106] Iteration 130700, lr = 2.75302e-05
    I0515 03:27:21.372243 23147 solver.cpp:228] Iteration 130800, loss = 0.00779259
    I0515 03:27:21.372361 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:27:21.372392 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00779255 (* 1 = 0.00779255 loss)
    I0515 03:27:21.372413 23147 sgd_solver.cpp:106] Iteration 130800, lr = 2.75155e-05
    I0515 03:27:29.457816 23147 solver.cpp:228] Iteration 130900, loss = 0.0188309
    I0515 03:27:29.457876 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:27:29.457906 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0188308 (* 1 = 0.0188308 loss)
    I0515 03:27:29.457927 23147 sgd_solver.cpp:106] Iteration 130900, lr = 2.75009e-05
    I0515 03:27:37.466593 23147 solver.cpp:337] Iteration 131000, Testing net (#0)
    I0515 03:27:41.835415 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.826
    I0515 03:27:41.835480 23147 solver.cpp:404]     Test net output #1: loss_c = 0.53211 (* 1 = 0.53211 loss)
    I0515 03:27:41.890733 23147 solver.cpp:228] Iteration 131000, loss = 0.0205053
    I0515 03:27:41.890768 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:27:41.890795 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0205053 (* 1 = 0.0205053 loss)
    I0515 03:27:41.890822 23147 sgd_solver.cpp:106] Iteration 131000, lr = 2.74863e-05
    I0515 03:27:49.982156 23147 solver.cpp:228] Iteration 131100, loss = 0.0264368
    I0515 03:27:49.982213 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:27:49.982242 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0264367 (* 1 = 0.0264367 loss)
    I0515 03:27:49.982264 23147 sgd_solver.cpp:106] Iteration 131100, lr = 2.74716e-05
    I0515 03:27:58.067147 23147 solver.cpp:228] Iteration 131200, loss = 0.0111349
    I0515 03:27:58.067347 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:27:58.067394 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0111348 (* 1 = 0.0111348 loss)
    I0515 03:27:58.067417 23147 sgd_solver.cpp:106] Iteration 131200, lr = 2.74571e-05
    I0515 03:28:06.153887 23147 solver.cpp:228] Iteration 131300, loss = 0.0176586
    I0515 03:28:06.153939 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:28:06.153969 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0176586 (* 1 = 0.0176586 loss)
    I0515 03:28:06.153990 23147 sgd_solver.cpp:106] Iteration 131300, lr = 2.74425e-05
    I0515 03:28:14.240121 23147 solver.cpp:228] Iteration 131400, loss = 0.0265747
    I0515 03:28:14.240175 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:28:14.240203 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0265746 (* 1 = 0.0265746 loss)
    I0515 03:28:14.240226 23147 sgd_solver.cpp:106] Iteration 131400, lr = 2.74279e-05
    I0515 03:28:22.325225 23147 solver.cpp:228] Iteration 131500, loss = 0.0135662
    I0515 03:28:22.325281 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:28:22.325310 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0135662 (* 1 = 0.0135662 loss)
    I0515 03:28:22.325332 23147 sgd_solver.cpp:106] Iteration 131500, lr = 2.74134e-05
    I0515 03:28:30.415244 23147 solver.cpp:228] Iteration 131600, loss = 0.0269893
    I0515 03:28:30.415362 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:28:30.415392 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0269893 (* 1 = 0.0269893 loss)
    I0515 03:28:30.415415 23147 sgd_solver.cpp:106] Iteration 131600, lr = 2.73989e-05
    I0515 03:28:38.502331 23147 solver.cpp:228] Iteration 131700, loss = 0.0294067
    I0515 03:28:38.502388 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:28:38.502419 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0294066 (* 1 = 0.0294066 loss)
    I0515 03:28:38.502440 23147 sgd_solver.cpp:106] Iteration 131700, lr = 2.73844e-05
    I0515 03:28:46.594357 23147 solver.cpp:228] Iteration 131800, loss = 0.0220195
    I0515 03:28:46.594415 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:28:46.594444 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0220195 (* 1 = 0.0220195 loss)
    I0515 03:28:46.594466 23147 sgd_solver.cpp:106] Iteration 131800, lr = 2.73699e-05
    I0515 03:28:54.680873 23147 solver.cpp:228] Iteration 131900, loss = 0.0161491
    I0515 03:28:54.680934 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:28:54.680963 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0161491 (* 1 = 0.0161491 loss)
    I0515 03:28:54.680985 23147 sgd_solver.cpp:106] Iteration 131900, lr = 2.73554e-05
    I0515 03:29:02.692936 23147 solver.cpp:337] Iteration 132000, Testing net (#0)
    I0515 03:29:07.109839 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.8335
    I0515 03:29:07.109896 23147 solver.cpp:404]     Test net output #1: loss_c = 0.535064 (* 1 = 0.535064 loss)
    I0515 03:29:07.162104 23147 solver.cpp:228] Iteration 132000, loss = 0.0187281
    I0515 03:29:07.162127 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:29:07.162145 23147 solver.cpp:244]     Train net output #1: loss_c = 0.018728 (* 1 = 0.018728 loss)
    I0515 03:29:07.162164 23147 sgd_solver.cpp:106] Iteration 132000, lr = 2.7341e-05
    I0515 03:29:15.260679 23147 solver.cpp:228] Iteration 132100, loss = 0.0139757
    I0515 03:29:15.260740 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:29:15.260768 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0139756 (* 1 = 0.0139756 loss)
    I0515 03:29:15.260792 23147 sgd_solver.cpp:106] Iteration 132100, lr = 2.73265e-05
    I0515 03:29:23.351660 23147 solver.cpp:228] Iteration 132200, loss = 0.0102655
    I0515 03:29:23.351721 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:29:23.351749 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0102655 (* 1 = 0.0102655 loss)
    I0515 03:29:23.351771 23147 sgd_solver.cpp:106] Iteration 132200, lr = 2.73121e-05
    I0515 03:29:31.442054 23147 solver.cpp:228] Iteration 132300, loss = 0.0246646
    I0515 03:29:31.442108 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:29:31.442138 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0246645 (* 1 = 0.0246645 loss)
    I0515 03:29:31.442160 23147 sgd_solver.cpp:106] Iteration 132300, lr = 2.72977e-05
    I0515 03:29:39.531613 23147 solver.cpp:228] Iteration 132400, loss = 0.0247164
    I0515 03:29:39.531764 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:29:39.531795 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0247164 (* 1 = 0.0247164 loss)
    I0515 03:29:39.531817 23147 sgd_solver.cpp:106] Iteration 132400, lr = 2.72833e-05
    I0515 03:29:47.618060 23147 solver.cpp:228] Iteration 132500, loss = 0.00734603
    I0515 03:29:47.618114 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:29:47.618144 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00734599 (* 1 = 0.00734599 loss)
    I0515 03:29:47.618165 23147 sgd_solver.cpp:106] Iteration 132500, lr = 2.7269e-05
    I0515 03:29:55.710734 23147 solver.cpp:228] Iteration 132600, loss = 0.0118438
    I0515 03:29:55.710795 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:29:55.710824 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0118438 (* 1 = 0.0118438 loss)
    I0515 03:29:55.710846 23147 sgd_solver.cpp:106] Iteration 132600, lr = 2.72546e-05
    I0515 03:30:03.800299 23147 solver.cpp:228] Iteration 132700, loss = 0.00949827
    I0515 03:30:03.800354 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:30:03.800385 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00949823 (* 1 = 0.00949823 loss)
    I0515 03:30:03.800406 23147 sgd_solver.cpp:106] Iteration 132700, lr = 2.72403e-05
    I0515 03:30:11.893129 23147 solver.cpp:228] Iteration 132800, loss = 0.0154535
    I0515 03:30:11.893244 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:30:11.893272 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0154534 (* 1 = 0.0154534 loss)
    I0515 03:30:11.893295 23147 sgd_solver.cpp:106] Iteration 132800, lr = 2.7226e-05
    I0515 03:30:19.980218 23147 solver.cpp:228] Iteration 132900, loss = 0.0078494
    I0515 03:30:19.980273 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:30:19.980301 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00784935 (* 1 = 0.00784935 loss)
    I0515 03:30:19.980324 23147 sgd_solver.cpp:106] Iteration 132900, lr = 2.72117e-05
    I0515 03:30:28.030959 23147 solver.cpp:337] Iteration 133000, Testing net (#0)
    I0515 03:30:32.438993 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.823583
    I0515 03:30:32.439051 23147 solver.cpp:404]     Test net output #1: loss_c = 0.542179 (* 1 = 0.542179 loss)
    I0515 03:30:32.495573 23147 solver.cpp:228] Iteration 133000, loss = 0.020026
    I0515 03:30:32.495628 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:30:32.495656 23147 solver.cpp:244]     Train net output #1: loss_c = 0.020026 (* 1 = 0.020026 loss)
    I0515 03:30:32.495681 23147 sgd_solver.cpp:106] Iteration 133000, lr = 2.71974e-05
    I0515 03:30:40.618814 23147 solver.cpp:228] Iteration 133100, loss = 0.0134445
    I0515 03:30:40.618861 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:30:40.618880 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0134444 (* 1 = 0.0134444 loss)
    I0515 03:30:40.618894 23147 sgd_solver.cpp:106] Iteration 133100, lr = 2.71832e-05
    I0515 03:30:48.758690 23147 solver.cpp:228] Iteration 133200, loss = 0.0203389
    I0515 03:30:48.758877 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:30:48.758896 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0203388 (* 1 = 0.0203388 loss)
    I0515 03:30:48.758911 23147 sgd_solver.cpp:106] Iteration 133200, lr = 2.71689e-05
    I0515 03:30:56.887682 23147 solver.cpp:228] Iteration 133300, loss = 0.00832422
    I0515 03:30:56.887730 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:30:56.887750 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00832416 (* 1 = 0.00832416 loss)
    I0515 03:30:56.887765 23147 sgd_solver.cpp:106] Iteration 133300, lr = 2.71547e-05
    I0515 03:31:05.021714 23147 solver.cpp:228] Iteration 133400, loss = 0.00936126
    I0515 03:31:05.021760 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:31:05.021781 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00936121 (* 1 = 0.00936121 loss)
    I0515 03:31:05.021796 23147 sgd_solver.cpp:106] Iteration 133400, lr = 2.71405e-05
    I0515 03:31:13.160181 23147 solver.cpp:228] Iteration 133500, loss = 0.0148168
    I0515 03:31:13.160226 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:31:13.160246 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0148167 (* 1 = 0.0148167 loss)
    I0515 03:31:13.160261 23147 sgd_solver.cpp:106] Iteration 133500, lr = 2.71263e-05
    I0515 03:31:21.301980 23147 solver.cpp:228] Iteration 133600, loss = 0.0542262
    I0515 03:31:21.302180 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:31:21.302201 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0542262 (* 1 = 0.0542262 loss)
    I0515 03:31:21.302217 23147 sgd_solver.cpp:106] Iteration 133600, lr = 2.71122e-05
    I0515 03:31:29.440984 23147 solver.cpp:228] Iteration 133700, loss = 0.0249333
    I0515 03:31:29.441028 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:31:29.441048 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0249333 (* 1 = 0.0249333 loss)
    I0515 03:31:29.441063 23147 sgd_solver.cpp:106] Iteration 133700, lr = 2.7098e-05
    I0515 03:31:37.579845 23147 solver.cpp:228] Iteration 133800, loss = 0.012005
    I0515 03:31:37.579890 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:31:37.579910 23147 solver.cpp:244]     Train net output #1: loss_c = 0.012005 (* 1 = 0.012005 loss)
    I0515 03:31:37.579926 23147 sgd_solver.cpp:106] Iteration 133800, lr = 2.70839e-05
    I0515 03:31:45.719522 23147 solver.cpp:228] Iteration 133900, loss = 0.0206211
    I0515 03:31:45.719573 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:31:45.719593 23147 solver.cpp:244]     Train net output #1: loss_c = 0.020621 (* 1 = 0.020621 loss)
    I0515 03:31:45.719607 23147 sgd_solver.cpp:106] Iteration 133900, lr = 2.70698e-05
    I0515 03:31:53.779147 23147 solver.cpp:337] Iteration 134000, Testing net (#0)
    I0515 03:31:58.206454 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.815083
    I0515 03:31:58.206499 23147 solver.cpp:404]     Test net output #1: loss_c = 0.544857 (* 1 = 0.544857 loss)
    I0515 03:31:58.258841 23147 solver.cpp:228] Iteration 134000, loss = 0.0219834
    I0515 03:31:58.258903 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:31:58.258922 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0219834 (* 1 = 0.0219834 loss)
    I0515 03:31:58.258944 23147 sgd_solver.cpp:106] Iteration 134000, lr = 2.70557e-05
    I0515 03:32:06.385130 23147 solver.cpp:228] Iteration 134100, loss = 0.0156284
    I0515 03:32:06.385182 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:32:06.385202 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0156284 (* 1 = 0.0156284 loss)
    I0515 03:32:06.385217 23147 sgd_solver.cpp:106] Iteration 134100, lr = 2.70416e-05
    I0515 03:32:14.519270 23147 solver.cpp:228] Iteration 134200, loss = 0.0089908
    I0515 03:32:14.519318 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:32:14.519340 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00899075 (* 1 = 0.00899075 loss)
    I0515 03:32:14.519354 23147 sgd_solver.cpp:106] Iteration 134200, lr = 2.70275e-05
    I0515 03:32:22.649950 23147 solver.cpp:228] Iteration 134300, loss = 0.00596925
    I0515 03:32:22.649996 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:32:22.650017 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0059692 (* 1 = 0.0059692 loss)
    I0515 03:32:22.650034 23147 sgd_solver.cpp:106] Iteration 134300, lr = 2.70135e-05
    I0515 03:32:30.775754 23147 solver.cpp:228] Iteration 134400, loss = 0.0227484
    I0515 03:32:30.775993 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:32:30.776043 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0227483 (* 1 = 0.0227483 loss)
    I0515 03:32:30.776068 23147 sgd_solver.cpp:106] Iteration 134400, lr = 2.69994e-05
    I0515 03:32:38.912724 23147 solver.cpp:228] Iteration 134500, loss = 0.00432974
    I0515 03:32:38.912767 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:32:38.912786 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00432968 (* 1 = 0.00432968 loss)
    I0515 03:32:38.912801 23147 sgd_solver.cpp:106] Iteration 134500, lr = 2.69854e-05
    I0515 03:32:47.040130 23147 solver.cpp:228] Iteration 134600, loss = 0.0170033
    I0515 03:32:47.040182 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:32:47.040202 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0170032 (* 1 = 0.0170032 loss)
    I0515 03:32:47.040217 23147 sgd_solver.cpp:106] Iteration 134600, lr = 2.69714e-05
    I0515 03:32:55.177208 23147 solver.cpp:228] Iteration 134700, loss = 0.0154224
    I0515 03:32:55.177256 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:32:55.177280 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0154223 (* 1 = 0.0154223 loss)
    I0515 03:32:55.177295 23147 sgd_solver.cpp:106] Iteration 134700, lr = 2.69574e-05
    I0515 03:33:03.314697 23147 solver.cpp:228] Iteration 134800, loss = 0.0399028
    I0515 03:33:03.314798 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:33:03.314818 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0399028 (* 1 = 0.0399028 loss)
    I0515 03:33:03.314833 23147 sgd_solver.cpp:106] Iteration 134800, lr = 2.69435e-05
    I0515 03:33:11.452167 23147 solver.cpp:228] Iteration 134900, loss = 0.0264668
    I0515 03:33:11.452210 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:33:11.452230 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0264668 (* 1 = 0.0264668 loss)
    I0515 03:33:11.452245 23147 sgd_solver.cpp:106] Iteration 134900, lr = 2.69295e-05
    I0515 03:33:19.507175 23147 solver.cpp:337] Iteration 135000, Testing net (#0)
    I0515 03:33:23.909869 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.815
    I0515 03:33:23.909930 23147 solver.cpp:404]     Test net output #1: loss_c = 0.544288 (* 1 = 0.544288 loss)
    I0515 03:33:23.965147 23147 solver.cpp:228] Iteration 135000, loss = 0.0109624
    I0515 03:33:23.965214 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:33:23.965245 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0109624 (* 1 = 0.0109624 loss)
    I0515 03:33:23.965271 23147 sgd_solver.cpp:106] Iteration 135000, lr = 2.69156e-05
    I0515 03:33:32.096329 23147 solver.cpp:228] Iteration 135100, loss = 0.0173888
    I0515 03:33:32.096375 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:33:32.096396 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0173888 (* 1 = 0.0173888 loss)
    I0515 03:33:32.096415 23147 sgd_solver.cpp:106] Iteration 135100, lr = 2.69017e-05
    I0515 03:33:40.234922 23147 solver.cpp:228] Iteration 135200, loss = 0.0350306
    I0515 03:33:40.235074 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:33:40.235119 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0350305 (* 1 = 0.0350305 loss)
    I0515 03:33:40.235144 23147 sgd_solver.cpp:106] Iteration 135200, lr = 2.68878e-05
    I0515 03:33:48.369029 23147 solver.cpp:228] Iteration 135300, loss = 0.0389968
    I0515 03:33:48.369079 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:33:48.369101 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0389967 (* 1 = 0.0389967 loss)
    I0515 03:33:48.369115 23147 sgd_solver.cpp:106] Iteration 135300, lr = 2.68739e-05
    I0515 03:33:56.495309 23147 solver.cpp:228] Iteration 135400, loss = 0.00512507
    I0515 03:33:56.495349 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:33:56.495373 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00512499 (* 1 = 0.00512499 loss)
    I0515 03:33:56.495388 23147 sgd_solver.cpp:106] Iteration 135400, lr = 2.686e-05
    I0515 03:34:04.613803 23147 solver.cpp:228] Iteration 135500, loss = 0.0312716
    I0515 03:34:04.613852 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:34:04.613873 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0312716 (* 1 = 0.0312716 loss)
    I0515 03:34:04.613888 23147 sgd_solver.cpp:106] Iteration 135500, lr = 2.68462e-05
    I0515 03:34:12.753401 23147 solver.cpp:228] Iteration 135600, loss = 0.0114693
    I0515 03:34:12.753639 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:34:12.753677 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0114693 (* 1 = 0.0114693 loss)
    I0515 03:34:12.753695 23147 sgd_solver.cpp:106] Iteration 135600, lr = 2.68324e-05
    I0515 03:34:20.875586 23147 solver.cpp:228] Iteration 135700, loss = 0.011953
    I0515 03:34:20.875630 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:34:20.875649 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0119529 (* 1 = 0.0119529 loss)
    I0515 03:34:20.875664 23147 sgd_solver.cpp:106] Iteration 135700, lr = 2.68186e-05
    I0515 03:34:29.012459 23147 solver.cpp:228] Iteration 135800, loss = 0.0129542
    I0515 03:34:29.012504 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:34:29.012523 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0129541 (* 1 = 0.0129541 loss)
    I0515 03:34:29.012537 23147 sgd_solver.cpp:106] Iteration 135800, lr = 2.68048e-05
    I0515 03:34:37.132666 23147 solver.cpp:228] Iteration 135900, loss = 0.0173351
    I0515 03:34:37.132709 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:34:37.132730 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0173351 (* 1 = 0.0173351 loss)
    I0515 03:34:37.132743 23147 sgd_solver.cpp:106] Iteration 135900, lr = 2.6791e-05
    I0515 03:34:45.190522 23147 solver.cpp:337] Iteration 136000, Testing net (#0)
    I0515 03:34:49.613495 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.817
    I0515 03:34:49.613543 23147 solver.cpp:404]     Test net output #1: loss_c = 0.562694 (* 1 = 0.562694 loss)
    I0515 03:34:49.670506 23147 solver.cpp:228] Iteration 136000, loss = 0.0161008
    I0515 03:34:49.670583 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:34:49.670614 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0161007 (* 1 = 0.0161007 loss)
    I0515 03:34:49.670639 23147 sgd_solver.cpp:106] Iteration 136000, lr = 2.67772e-05
    I0515 03:34:57.805292 23147 solver.cpp:228] Iteration 136100, loss = 0.0133565
    I0515 03:34:57.805343 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:34:57.805363 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0133564 (* 1 = 0.0133564 loss)
    I0515 03:34:57.805378 23147 sgd_solver.cpp:106] Iteration 136100, lr = 2.67635e-05
    I0515 03:35:05.943874 23147 solver.cpp:228] Iteration 136200, loss = 0.0264511
    I0515 03:35:05.943923 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:35:05.943944 23147 solver.cpp:244]     Train net output #1: loss_c = 0.026451 (* 1 = 0.026451 loss)
    I0515 03:35:05.943958 23147 sgd_solver.cpp:106] Iteration 136200, lr = 2.67497e-05
    I0515 03:35:14.044188 23147 solver.cpp:228] Iteration 136300, loss = 0.00987274
    I0515 03:35:14.044250 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:35:14.044281 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00987268 (* 1 = 0.00987268 loss)
    I0515 03:35:14.044302 23147 sgd_solver.cpp:106] Iteration 136300, lr = 2.6736e-05
    I0515 03:35:22.135640 23147 solver.cpp:228] Iteration 136400, loss = 0.00629529
    I0515 03:35:22.135833 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:35:22.135884 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00629524 (* 1 = 0.00629524 loss)
    I0515 03:35:22.135906 23147 sgd_solver.cpp:106] Iteration 136400, lr = 2.67223e-05
    I0515 03:35:30.221935 23147 solver.cpp:228] Iteration 136500, loss = 0.016914
    I0515 03:35:30.221995 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:35:30.222025 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0169139 (* 1 = 0.0169139 loss)
    I0515 03:35:30.222048 23147 sgd_solver.cpp:106] Iteration 136500, lr = 2.67086e-05
    I0515 03:35:38.310716 23147 solver.cpp:228] Iteration 136600, loss = 0.00880482
    I0515 03:35:38.310780 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:35:38.310809 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00880479 (* 1 = 0.00880479 loss)
    I0515 03:35:38.310832 23147 sgd_solver.cpp:106] Iteration 136600, lr = 2.6695e-05
    I0515 03:35:46.398948 23147 solver.cpp:228] Iteration 136700, loss = 0.0115085
    I0515 03:35:46.399009 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:35:46.399039 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0115085 (* 1 = 0.0115085 loss)
    I0515 03:35:46.399060 23147 sgd_solver.cpp:106] Iteration 136700, lr = 2.66813e-05
    I0515 03:35:54.485054 23147 solver.cpp:228] Iteration 136800, loss = 0.028124
    I0515 03:35:54.485319 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:35:54.485364 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0281239 (* 1 = 0.0281239 loss)
    I0515 03:35:54.485389 23147 sgd_solver.cpp:106] Iteration 136800, lr = 2.66677e-05
    I0515 03:36:02.574365 23147 solver.cpp:228] Iteration 136900, loss = 0.0218547
    I0515 03:36:02.574421 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:36:02.574450 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0218547 (* 1 = 0.0218547 loss)
    I0515 03:36:02.574471 23147 sgd_solver.cpp:106] Iteration 136900, lr = 2.66541e-05
    I0515 03:36:10.582837 23147 solver.cpp:337] Iteration 137000, Testing net (#0)
    I0515 03:36:14.941468 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.80375
    I0515 03:36:14.941526 23147 solver.cpp:404]     Test net output #1: loss_c = 0.576099 (* 1 = 0.576099 loss)
    I0515 03:36:14.996660 23147 solver.cpp:228] Iteration 137000, loss = 0.0238296
    I0515 03:36:14.996745 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:36:14.996775 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0238295 (* 1 = 0.0238295 loss)
    I0515 03:36:14.996800 23147 sgd_solver.cpp:106] Iteration 137000, lr = 2.66405e-05
    I0515 03:36:23.132014 23147 solver.cpp:228] Iteration 137100, loss = 0.0151592
    I0515 03:36:23.132055 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:36:23.132074 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0151592 (* 1 = 0.0151592 loss)
    I0515 03:36:23.132091 23147 sgd_solver.cpp:106] Iteration 137100, lr = 2.66269e-05
    I0515 03:36:31.271129 23147 solver.cpp:228] Iteration 137200, loss = 0.0119208
    I0515 03:36:31.271219 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:36:31.271239 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0119207 (* 1 = 0.0119207 loss)
    I0515 03:36:31.271255 23147 sgd_solver.cpp:106] Iteration 137200, lr = 2.66133e-05
    I0515 03:36:39.408383 23147 solver.cpp:228] Iteration 137300, loss = 0.0242487
    I0515 03:36:39.408427 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:36:39.408452 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0242487 (* 1 = 0.0242487 loss)
    I0515 03:36:39.408466 23147 sgd_solver.cpp:106] Iteration 137300, lr = 2.65998e-05
    I0515 03:36:47.549908 23147 solver.cpp:228] Iteration 137400, loss = 0.0381685
    I0515 03:36:47.549948 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:36:47.549973 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0381685 (* 1 = 0.0381685 loss)
    I0515 03:36:47.549988 23147 sgd_solver.cpp:106] Iteration 137400, lr = 2.65862e-05
    I0515 03:36:55.686717 23147 solver.cpp:228] Iteration 137500, loss = 0.0210037
    I0515 03:36:55.686766 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:36:55.686787 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0210037 (* 1 = 0.0210037 loss)
    I0515 03:36:55.686802 23147 sgd_solver.cpp:106] Iteration 137500, lr = 2.65727e-05
    I0515 03:37:03.825593 23147 solver.cpp:228] Iteration 137600, loss = 0.012259
    I0515 03:37:03.825844 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:37:03.825892 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0122589 (* 1 = 0.0122589 loss)
    I0515 03:37:03.825917 23147 sgd_solver.cpp:106] Iteration 137600, lr = 2.65592e-05
    I0515 03:37:11.903674 23147 solver.cpp:228] Iteration 137700, loss = 0.00901692
    I0515 03:37:11.903717 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:37:11.903738 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00901687 (* 1 = 0.00901687 loss)
    I0515 03:37:11.903751 23147 sgd_solver.cpp:106] Iteration 137700, lr = 2.65457e-05
    I0515 03:37:20.043249 23147 solver.cpp:228] Iteration 137800, loss = 0.0132862
    I0515 03:37:20.043292 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:37:20.043311 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0132862 (* 1 = 0.0132862 loss)
    I0515 03:37:20.043326 23147 sgd_solver.cpp:106] Iteration 137800, lr = 2.65323e-05
    I0515 03:37:28.180079 23147 solver.cpp:228] Iteration 137900, loss = 0.006736
    I0515 03:37:28.180135 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:37:28.180155 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00673595 (* 1 = 0.00673595 loss)
    I0515 03:37:28.180168 23147 sgd_solver.cpp:106] Iteration 137900, lr = 2.65188e-05
    I0515 03:37:36.238554 23147 solver.cpp:337] Iteration 138000, Testing net (#0)
    I0515 03:37:40.647896 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.81475
    I0515 03:37:40.647958 23147 solver.cpp:404]     Test net output #1: loss_c = 0.549003 (* 1 = 0.549003 loss)
    I0515 03:37:40.699874 23147 solver.cpp:228] Iteration 138000, loss = 0.0165628
    I0515 03:37:40.699898 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:37:40.699916 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0165628 (* 1 = 0.0165628 loss)
    I0515 03:37:40.699936 23147 sgd_solver.cpp:106] Iteration 138000, lr = 2.65054e-05
    I0515 03:37:48.840739 23147 solver.cpp:228] Iteration 138100, loss = 0.0171645
    I0515 03:37:48.840785 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:37:48.840806 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0171645 (* 1 = 0.0171645 loss)
    I0515 03:37:48.840819 23147 sgd_solver.cpp:106] Iteration 138100, lr = 2.64919e-05
    I0515 03:37:56.976789 23147 solver.cpp:228] Iteration 138200, loss = 0.0110829
    I0515 03:37:56.976837 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:37:56.976856 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0110829 (* 1 = 0.0110829 loss)
    I0515 03:37:56.976871 23147 sgd_solver.cpp:106] Iteration 138200, lr = 2.64785e-05
    I0515 03:38:05.107271 23147 solver.cpp:228] Iteration 138300, loss = 0.0233087
    I0515 03:38:05.107317 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:38:05.107337 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0233087 (* 1 = 0.0233087 loss)
    I0515 03:38:05.107352 23147 sgd_solver.cpp:106] Iteration 138300, lr = 2.64651e-05
    I0515 03:38:13.248539 23147 solver.cpp:228] Iteration 138400, loss = 0.0155204
    I0515 03:38:13.248735 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:38:13.248756 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0155204 (* 1 = 0.0155204 loss)
    I0515 03:38:13.248770 23147 sgd_solver.cpp:106] Iteration 138400, lr = 2.64518e-05
    I0515 03:38:21.383352 23147 solver.cpp:228] Iteration 138500, loss = 0.0124301
    I0515 03:38:21.383395 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:38:21.383416 23147 solver.cpp:244]     Train net output #1: loss_c = 0.01243 (* 1 = 0.01243 loss)
    I0515 03:38:21.383431 23147 sgd_solver.cpp:106] Iteration 138500, lr = 2.64384e-05
    I0515 03:38:29.522752 23147 solver.cpp:228] Iteration 138600, loss = 0.0139245
    I0515 03:38:29.522802 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:38:29.522822 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0139244 (* 1 = 0.0139244 loss)
    I0515 03:38:29.522837 23147 sgd_solver.cpp:106] Iteration 138600, lr = 2.64251e-05
    I0515 03:38:37.654644 23147 solver.cpp:228] Iteration 138700, loss = 0.0106837
    I0515 03:38:37.654687 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:38:37.654707 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0106837 (* 1 = 0.0106837 loss)
    I0515 03:38:37.654722 23147 sgd_solver.cpp:106] Iteration 138700, lr = 2.64117e-05
    I0515 03:38:45.792537 23147 solver.cpp:228] Iteration 138800, loss = 0.0155669
    I0515 03:38:45.792726 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:38:45.792747 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0155669 (* 1 = 0.0155669 loss)
    I0515 03:38:45.792778 23147 sgd_solver.cpp:106] Iteration 138800, lr = 2.63984e-05
    I0515 03:38:53.928419 23147 solver.cpp:228] Iteration 138900, loss = 0.00735661
    I0515 03:38:53.928464 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:38:53.928484 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00735657 (* 1 = 0.00735657 loss)
    I0515 03:38:53.928498 23147 sgd_solver.cpp:106] Iteration 138900, lr = 2.63851e-05
    I0515 03:39:01.985857 23147 solver.cpp:337] Iteration 139000, Testing net (#0)
    I0515 03:39:06.407070 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.81
    I0515 03:39:06.407129 23147 solver.cpp:404]     Test net output #1: loss_c = 0.565489 (* 1 = 0.565489 loss)
    I0515 03:39:06.462291 23147 solver.cpp:228] Iteration 139000, loss = 0.0373844
    I0515 03:39:06.462363 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:39:06.462393 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0373843 (* 1 = 0.0373843 loss)
    I0515 03:39:06.462417 23147 sgd_solver.cpp:106] Iteration 139000, lr = 2.63718e-05
    I0515 03:39:14.599102 23147 solver.cpp:228] Iteration 139100, loss = 0.0189711
    I0515 03:39:14.599143 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:39:14.599162 23147 solver.cpp:244]     Train net output #1: loss_c = 0.018971 (* 1 = 0.018971 loss)
    I0515 03:39:14.599179 23147 sgd_solver.cpp:106] Iteration 139100, lr = 2.63586e-05
    I0515 03:39:22.679412 23147 solver.cpp:228] Iteration 139200, loss = 0.0132774
    I0515 03:39:22.679553 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:39:22.679599 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0132774 (* 1 = 0.0132774 loss)
    I0515 03:39:22.679623 23147 sgd_solver.cpp:106] Iteration 139200, lr = 2.63453e-05
    I0515 03:39:30.820895 23147 solver.cpp:228] Iteration 139300, loss = 0.0361596
    I0515 03:39:30.820941 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:39:30.820961 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0361596 (* 1 = 0.0361596 loss)
    I0515 03:39:30.820976 23147 sgd_solver.cpp:106] Iteration 139300, lr = 2.63321e-05
    I0515 03:39:38.958259 23147 solver.cpp:228] Iteration 139400, loss = 0.0119978
    I0515 03:39:38.958304 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:39:38.958323 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0119978 (* 1 = 0.0119978 loss)
    I0515 03:39:38.958339 23147 sgd_solver.cpp:106] Iteration 139400, lr = 2.63189e-05
    I0515 03:39:47.096305 23147 solver.cpp:228] Iteration 139500, loss = 0.0199468
    I0515 03:39:47.096351 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:39:47.096370 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0199467 (* 1 = 0.0199467 loss)
    I0515 03:39:47.096385 23147 sgd_solver.cpp:106] Iteration 139500, lr = 2.63057e-05
    I0515 03:39:55.232862 23147 solver.cpp:228] Iteration 139600, loss = 0.0190789
    I0515 03:39:55.233055 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:39:55.233077 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0190788 (* 1 = 0.0190788 loss)
    I0515 03:39:55.233110 23147 sgd_solver.cpp:106] Iteration 139600, lr = 2.62925e-05
    I0515 03:40:03.344951 23147 solver.cpp:228] Iteration 139700, loss = 0.0310742
    I0515 03:40:03.344996 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:40:03.345016 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0310742 (* 1 = 0.0310742 loss)
    I0515 03:40:03.345031 23147 sgd_solver.cpp:106] Iteration 139700, lr = 2.62793e-05
    I0515 03:40:11.483180 23147 solver.cpp:228] Iteration 139800, loss = 0.0150164
    I0515 03:40:11.483223 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:40:11.483243 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0150163 (* 1 = 0.0150163 loss)
    I0515 03:40:11.483258 23147 sgd_solver.cpp:106] Iteration 139800, lr = 2.62661e-05
    I0515 03:40:19.596192 23147 solver.cpp:228] Iteration 139900, loss = 0.0148121
    I0515 03:40:19.596243 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:40:19.596263 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0148121 (* 1 = 0.0148121 loss)
    I0515 03:40:19.596279 23147 sgd_solver.cpp:106] Iteration 139900, lr = 2.6253e-05
    I0515 03:40:27.648687 23147 solver.cpp:337] Iteration 140000, Testing net (#0)
    I0515 03:40:32.072337 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.818
    I0515 03:40:32.072387 23147 solver.cpp:404]     Test net output #1: loss_c = 0.560008 (* 1 = 0.560008 loss)
    I0515 03:40:32.123656 23147 solver.cpp:228] Iteration 140000, loss = 0.0192659
    I0515 03:40:32.123688 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:40:32.123708 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0192659 (* 1 = 0.0192659 loss)
    I0515 03:40:32.123725 23147 sgd_solver.cpp:106] Iteration 140000, lr = 2.62399e-05
    I0515 03:40:40.226800 23147 solver.cpp:228] Iteration 140100, loss = 0.0198973
    I0515 03:40:40.226847 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:40:40.226867 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0198972 (* 1 = 0.0198972 loss)
    I0515 03:40:40.226882 23147 sgd_solver.cpp:106] Iteration 140100, lr = 2.62267e-05
    I0515 03:40:48.365741 23147 solver.cpp:228] Iteration 140200, loss = 0.0262687
    I0515 03:40:48.365789 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:40:48.365808 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0262687 (* 1 = 0.0262687 loss)
    I0515 03:40:48.365823 23147 sgd_solver.cpp:106] Iteration 140200, lr = 2.62137e-05
    I0515 03:40:56.499176 23147 solver.cpp:228] Iteration 140300, loss = 0.0151432
    I0515 03:40:56.499224 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:40:56.499248 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0151431 (* 1 = 0.0151431 loss)
    I0515 03:40:56.499263 23147 sgd_solver.cpp:106] Iteration 140300, lr = 2.62006e-05
    I0515 03:41:04.634522 23147 solver.cpp:228] Iteration 140400, loss = 0.013599
    I0515 03:41:04.634676 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:41:04.634722 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0135989 (* 1 = 0.0135989 loss)
    I0515 03:41:04.634747 23147 sgd_solver.cpp:106] Iteration 140400, lr = 2.61875e-05
    I0515 03:41:12.711302 23147 solver.cpp:228] Iteration 140500, loss = 0.0354891
    I0515 03:41:12.711347 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:41:12.711369 23147 solver.cpp:244]     Train net output #1: loss_c = 0.035489 (* 1 = 0.035489 loss)
    I0515 03:41:12.711382 23147 sgd_solver.cpp:106] Iteration 140500, lr = 2.61745e-05
    I0515 03:41:20.852628 23147 solver.cpp:228] Iteration 140600, loss = 0.0222735
    I0515 03:41:20.852672 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:41:20.852695 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0222734 (* 1 = 0.0222734 loss)
    I0515 03:41:20.852710 23147 sgd_solver.cpp:106] Iteration 140600, lr = 2.61614e-05
    I0515 03:41:28.993873 23147 solver.cpp:228] Iteration 140700, loss = 0.00955589
    I0515 03:41:28.993924 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:41:28.993944 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0095558 (* 1 = 0.0095558 loss)
    I0515 03:41:28.993959 23147 sgd_solver.cpp:106] Iteration 140700, lr = 2.61484e-05
    I0515 03:41:37.131921 23147 solver.cpp:228] Iteration 140800, loss = 0.00983119
    I0515 03:41:37.132104 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:41:37.132125 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0098311 (* 1 = 0.0098311 loss)
    I0515 03:41:37.132158 23147 sgd_solver.cpp:106] Iteration 140800, lr = 2.61354e-05
    I0515 03:41:45.270308 23147 solver.cpp:228] Iteration 140900, loss = 0.0195618
    I0515 03:41:45.270354 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:41:45.270375 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0195617 (* 1 = 0.0195617 loss)
    I0515 03:41:45.270388 23147 sgd_solver.cpp:106] Iteration 140900, lr = 2.61224e-05
    I0515 03:41:53.327560 23147 solver.cpp:337] Iteration 141000, Testing net (#0)
    I0515 03:41:57.744174 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.818083
    I0515 03:41:57.744222 23147 solver.cpp:404]     Test net output #1: loss_c = 0.542618 (* 1 = 0.542618 loss)
    I0515 03:41:57.799608 23147 solver.cpp:228] Iteration 141000, loss = 0.00727364
    I0515 03:41:57.799674 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:41:57.799705 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00727355 (* 1 = 0.00727355 loss)
    I0515 03:41:57.799731 23147 sgd_solver.cpp:106] Iteration 141000, lr = 2.61094e-05
    I0515 03:42:05.929131 23147 solver.cpp:228] Iteration 141100, loss = 0.00918265
    I0515 03:42:05.929182 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:42:05.929201 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00918256 (* 1 = 0.00918256 loss)
    I0515 03:42:05.929216 23147 sgd_solver.cpp:106] Iteration 141100, lr = 2.60965e-05
    I0515 03:42:14.025941 23147 solver.cpp:228] Iteration 141200, loss = 0.0185159
    I0515 03:42:14.026059 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:42:14.026089 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0185158 (* 1 = 0.0185158 loss)
    I0515 03:42:14.026111 23147 sgd_solver.cpp:106] Iteration 141200, lr = 2.60835e-05
    I0515 03:42:22.115957 23147 solver.cpp:228] Iteration 141300, loss = 0.0218241
    I0515 03:42:22.116016 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:42:22.116045 23147 solver.cpp:244]     Train net output #1: loss_c = 0.021824 (* 1 = 0.021824 loss)
    I0515 03:42:22.116067 23147 sgd_solver.cpp:106] Iteration 141300, lr = 2.60706e-05
    I0515 03:42:30.207516 23147 solver.cpp:228] Iteration 141400, loss = 0.0214005
    I0515 03:42:30.207572 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:42:30.207602 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0214005 (* 1 = 0.0214005 loss)
    I0515 03:42:30.207624 23147 sgd_solver.cpp:106] Iteration 141400, lr = 2.60577e-05
    I0515 03:42:38.297950 23147 solver.cpp:228] Iteration 141500, loss = 0.044887
    I0515 03:42:38.298007 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 03:42:38.298035 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0448869 (* 1 = 0.0448869 loss)
    I0515 03:42:38.298056 23147 sgd_solver.cpp:106] Iteration 141500, lr = 2.60448e-05
    I0515 03:42:46.384398 23147 solver.cpp:228] Iteration 141600, loss = 0.0255787
    I0515 03:42:46.384665 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:42:46.384709 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0255786 (* 1 = 0.0255786 loss)
    I0515 03:42:46.384735 23147 sgd_solver.cpp:106] Iteration 141600, lr = 2.60319e-05
    I0515 03:42:54.475890 23147 solver.cpp:228] Iteration 141700, loss = 0.0112507
    I0515 03:42:54.475947 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:42:54.475975 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0112506 (* 1 = 0.0112506 loss)
    I0515 03:42:54.475997 23147 sgd_solver.cpp:106] Iteration 141700, lr = 2.6019e-05
    I0515 03:43:02.595059 23147 solver.cpp:228] Iteration 141800, loss = 0.029838
    I0515 03:43:02.595103 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:43:02.595123 23147 solver.cpp:244]     Train net output #1: loss_c = 0.029838 (* 1 = 0.029838 loss)
    I0515 03:43:02.595137 23147 sgd_solver.cpp:106] Iteration 141800, lr = 2.60062e-05
    I0515 03:43:10.736863 23147 solver.cpp:228] Iteration 141900, loss = 0.0182704
    I0515 03:43:10.736917 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:43:10.736937 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0182703 (* 1 = 0.0182703 loss)
    I0515 03:43:10.736953 23147 sgd_solver.cpp:106] Iteration 141900, lr = 2.59933e-05
    I0515 03:43:18.788038 23147 solver.cpp:337] Iteration 142000, Testing net (#0)
    I0515 03:43:23.192530 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.81375
    I0515 03:43:23.192579 23147 solver.cpp:404]     Test net output #1: loss_c = 0.558126 (* 1 = 0.558126 loss)
    I0515 03:43:23.244367 23147 solver.cpp:228] Iteration 142000, loss = 0.0185043
    I0515 03:43:23.244465 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:43:23.244493 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0185043 (* 1 = 0.0185043 loss)
    I0515 03:43:23.244513 23147 sgd_solver.cpp:106] Iteration 142000, lr = 2.59805e-05
    I0515 03:43:31.332954 23147 solver.cpp:228] Iteration 142100, loss = 0.0136382
    I0515 03:43:31.333004 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:43:31.333034 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0136382 (* 1 = 0.0136382 loss)
    I0515 03:43:31.333055 23147 sgd_solver.cpp:106] Iteration 142100, lr = 2.59677e-05
    I0515 03:43:39.420253 23147 solver.cpp:228] Iteration 142200, loss = 0.0209797
    I0515 03:43:39.420303 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:43:39.420333 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0209796 (* 1 = 0.0209796 loss)
    I0515 03:43:39.420354 23147 sgd_solver.cpp:106] Iteration 142200, lr = 2.59549e-05
    I0515 03:43:47.508256 23147 solver.cpp:228] Iteration 142300, loss = 0.0218792
    I0515 03:43:47.508314 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:43:47.508344 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0218791 (* 1 = 0.0218791 loss)
    I0515 03:43:47.508366 23147 sgd_solver.cpp:106] Iteration 142300, lr = 2.59421e-05
    I0515 03:43:55.598130 23147 solver.cpp:228] Iteration 142400, loss = 0.0521613
    I0515 03:43:55.598266 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:43:55.598312 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0521612 (* 1 = 0.0521612 loss)
    I0515 03:43:55.598336 23147 sgd_solver.cpp:106] Iteration 142400, lr = 2.59293e-05
    I0515 03:44:03.725953 23147 solver.cpp:228] Iteration 142500, loss = 0.00588433
    I0515 03:44:03.726001 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:44:03.726021 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00588425 (* 1 = 0.00588425 loss)
    I0515 03:44:03.726034 23147 sgd_solver.cpp:106] Iteration 142500, lr = 2.59166e-05
    I0515 03:44:11.868633 23147 solver.cpp:228] Iteration 142600, loss = 0.0169303
    I0515 03:44:11.868680 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:44:11.868698 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0169302 (* 1 = 0.0169302 loss)
    I0515 03:44:11.868713 23147 sgd_solver.cpp:106] Iteration 142600, lr = 2.59038e-05
    I0515 03:44:20.005626 23147 solver.cpp:228] Iteration 142700, loss = 0.0292089
    I0515 03:44:20.005671 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:44:20.005691 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0292088 (* 1 = 0.0292088 loss)
    I0515 03:44:20.005705 23147 sgd_solver.cpp:106] Iteration 142700, lr = 2.58911e-05
    I0515 03:44:28.145457 23147 solver.cpp:228] Iteration 142800, loss = 0.0192674
    I0515 03:44:28.145640 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:44:28.145661 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0192674 (* 1 = 0.0192674 loss)
    I0515 03:44:28.145676 23147 sgd_solver.cpp:106] Iteration 142800, lr = 2.58784e-05
    I0515 03:44:36.284677 23147 solver.cpp:228] Iteration 142900, loss = 0.0277712
    I0515 03:44:36.284723 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:44:36.284741 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0277712 (* 1 = 0.0277712 loss)
    I0515 03:44:36.284756 23147 sgd_solver.cpp:106] Iteration 142900, lr = 2.58657e-05
    I0515 03:44:44.335620 23147 solver.cpp:337] Iteration 143000, Testing net (#0)
    I0515 03:44:48.752156 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.814917
    I0515 03:44:48.752204 23147 solver.cpp:404]     Test net output #1: loss_c = 0.543463 (* 1 = 0.543463 loss)
    I0515 03:44:48.803670 23147 solver.cpp:228] Iteration 143000, loss = 0.0160128
    I0515 03:44:48.803705 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:44:48.803725 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0160127 (* 1 = 0.0160127 loss)
    I0515 03:44:48.803745 23147 sgd_solver.cpp:106] Iteration 143000, lr = 2.5853e-05
    I0515 03:44:56.920750 23147 solver.cpp:228] Iteration 143100, loss = 0.0157948
    I0515 03:44:56.920799 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:44:56.920817 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0157947 (* 1 = 0.0157947 loss)
    I0515 03:44:56.920832 23147 sgd_solver.cpp:106] Iteration 143100, lr = 2.58404e-05
    I0515 03:45:05.058153 23147 solver.cpp:228] Iteration 143200, loss = 0.0124234
    I0515 03:45:05.058429 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:45:05.058473 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0124233 (* 1 = 0.0124233 loss)
    I0515 03:45:05.058498 23147 sgd_solver.cpp:106] Iteration 143200, lr = 2.58277e-05
    I0515 03:45:13.194118 23147 solver.cpp:228] Iteration 143300, loss = 0.0056365
    I0515 03:45:13.194159 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:45:13.194178 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0056364 (* 1 = 0.0056364 loss)
    I0515 03:45:13.194193 23147 sgd_solver.cpp:106] Iteration 143300, lr = 2.58151e-05
    I0515 03:45:21.334985 23147 solver.cpp:228] Iteration 143400, loss = 0.0340633
    I0515 03:45:21.335026 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:45:21.335047 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0340632 (* 1 = 0.0340632 loss)
    I0515 03:45:21.335062 23147 sgd_solver.cpp:106] Iteration 143400, lr = 2.58025e-05
    I0515 03:45:29.472040 23147 solver.cpp:228] Iteration 143500, loss = 0.00984277
    I0515 03:45:29.472080 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:45:29.472103 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00984267 (* 1 = 0.00984267 loss)
    I0515 03:45:29.472118 23147 sgd_solver.cpp:106] Iteration 143500, lr = 2.57898e-05
    I0515 03:45:37.613297 23147 solver.cpp:228] Iteration 143600, loss = 0.0121658
    I0515 03:45:37.613441 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:45:37.613487 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0121657 (* 1 = 0.0121657 loss)
    I0515 03:45:37.613512 23147 sgd_solver.cpp:106] Iteration 143600, lr = 2.57772e-05
    I0515 03:45:45.740118 23147 solver.cpp:228] Iteration 143700, loss = 0.0099306
    I0515 03:45:45.740167 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:45:45.740187 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00993049 (* 1 = 0.00993049 loss)
    I0515 03:45:45.740202 23147 sgd_solver.cpp:106] Iteration 143700, lr = 2.57647e-05
    I0515 03:45:53.877142 23147 solver.cpp:228] Iteration 143800, loss = 0.0125033
    I0515 03:45:53.877192 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:45:53.877213 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0125032 (* 1 = 0.0125032 loss)
    I0515 03:45:53.877228 23147 sgd_solver.cpp:106] Iteration 143800, lr = 2.57521e-05
    I0515 03:46:02.014360 23147 solver.cpp:228] Iteration 143900, loss = 0.00698322
    I0515 03:46:02.014412 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:46:02.014432 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00698311 (* 1 = 0.00698311 loss)
    I0515 03:46:02.014447 23147 sgd_solver.cpp:106] Iteration 143900, lr = 2.57396e-05
    I0515 03:46:10.072396 23147 solver.cpp:337] Iteration 144000, Testing net (#0)
    I0515 03:46:14.496003 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.819833
    I0515 03:46:14.496055 23147 solver.cpp:404]     Test net output #1: loss_c = 0.526678 (* 1 = 0.526678 loss)
    I0515 03:46:14.547405 23147 solver.cpp:228] Iteration 144000, loss = 0.0119458
    I0515 03:46:14.547443 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:46:14.547463 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0119457 (* 1 = 0.0119457 loss)
    I0515 03:46:14.547482 23147 sgd_solver.cpp:106] Iteration 144000, lr = 2.5727e-05
    I0515 03:46:22.687163 23147 solver.cpp:228] Iteration 144100, loss = 0.0189425
    I0515 03:46:22.687208 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:46:22.687228 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0189424 (* 1 = 0.0189424 loss)
    I0515 03:46:22.687242 23147 sgd_solver.cpp:106] Iteration 144100, lr = 2.57145e-05
    I0515 03:46:30.776962 23147 solver.cpp:228] Iteration 144200, loss = 0.0128732
    I0515 03:46:30.777016 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:46:30.777042 23147 solver.cpp:244]     Train net output #1: loss_c = 0.012873 (* 1 = 0.012873 loss)
    I0515 03:46:30.777056 23147 sgd_solver.cpp:106] Iteration 144200, lr = 2.5702e-05
    I0515 03:46:38.831730 23147 solver.cpp:228] Iteration 144300, loss = 0.00772697
    I0515 03:46:38.831786 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:46:38.831812 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00772685 (* 1 = 0.00772685 loss)
    I0515 03:46:38.831827 23147 sgd_solver.cpp:106] Iteration 144300, lr = 2.56895e-05
    I0515 03:46:46.926144 23147 solver.cpp:228] Iteration 144400, loss = 0.00926362
    I0515 03:46:46.926288 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:46:46.926333 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0092635 (* 1 = 0.0092635 loss)
    I0515 03:46:46.926358 23147 sgd_solver.cpp:106] Iteration 144400, lr = 2.5677e-05
    I0515 03:46:55.064335 23147 solver.cpp:228] Iteration 144500, loss = 0.0206905
    I0515 03:46:55.064380 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:46:55.064399 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0206904 (* 1 = 0.0206904 loss)
    I0515 03:46:55.064414 23147 sgd_solver.cpp:106] Iteration 144500, lr = 2.56645e-05
    I0515 03:47:03.193173 23147 solver.cpp:228] Iteration 144600, loss = 0.0151285
    I0515 03:47:03.193218 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:47:03.193238 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0151284 (* 1 = 0.0151284 loss)
    I0515 03:47:03.193251 23147 sgd_solver.cpp:106] Iteration 144600, lr = 2.56521e-05
    I0515 03:47:11.325301 23147 solver.cpp:228] Iteration 144700, loss = 0.0101481
    I0515 03:47:11.325347 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:47:11.325367 23147 solver.cpp:244]     Train net output #1: loss_c = 0.010148 (* 1 = 0.010148 loss)
    I0515 03:47:11.325382 23147 sgd_solver.cpp:106] Iteration 144700, lr = 2.56397e-05
    I0515 03:47:19.461439 23147 solver.cpp:228] Iteration 144800, loss = 0.0116175
    I0515 03:47:19.461684 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:47:19.461730 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0116174 (* 1 = 0.0116174 loss)
    I0515 03:47:19.461755 23147 sgd_solver.cpp:106] Iteration 144800, lr = 2.56272e-05
    I0515 03:47:27.558524 23147 solver.cpp:228] Iteration 144900, loss = 0.0270573
    I0515 03:47:27.558584 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:47:27.558614 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0270572 (* 1 = 0.0270572 loss)
    I0515 03:47:27.558636 23147 sgd_solver.cpp:106] Iteration 144900, lr = 2.56148e-05
    I0515 03:47:35.563609 23147 solver.cpp:337] Iteration 145000, Testing net (#0)
    I0515 03:47:39.924931 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.8215
    I0515 03:47:39.924990 23147 solver.cpp:404]     Test net output #1: loss_c = 0.569102 (* 1 = 0.569102 loss)
    I0515 03:47:39.976802 23147 solver.cpp:228] Iteration 145000, loss = 0.0100743
    I0515 03:47:39.976853 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:47:39.976876 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0100742 (* 1 = 0.0100742 loss)
    I0515 03:47:39.976896 23147 sgd_solver.cpp:106] Iteration 145000, lr = 2.56024e-05
    I0515 03:47:48.117293 23147 solver.cpp:228] Iteration 145100, loss = 0.0121873
    I0515 03:47:48.117343 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:47:48.117363 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0121872 (* 1 = 0.0121872 loss)
    I0515 03:47:48.117378 23147 sgd_solver.cpp:106] Iteration 145100, lr = 2.55901e-05
    I0515 03:47:56.257632 23147 solver.cpp:228] Iteration 145200, loss = 0.0443266
    I0515 03:47:56.257748 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:47:56.257769 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0443265 (* 1 = 0.0443265 loss)
    I0515 03:47:56.257784 23147 sgd_solver.cpp:106] Iteration 145200, lr = 2.55777e-05
    I0515 03:48:04.349228 23147 solver.cpp:228] Iteration 145300, loss = 0.00996679
    I0515 03:48:04.349285 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:48:04.349314 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00996666 (* 1 = 0.00996666 loss)
    I0515 03:48:04.349336 23147 sgd_solver.cpp:106] Iteration 145300, lr = 2.55653e-05
    I0515 03:48:12.445364 23147 solver.cpp:228] Iteration 145400, loss = 0.0421255
    I0515 03:48:12.445425 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:48:12.445454 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0421253 (* 1 = 0.0421253 loss)
    I0515 03:48:12.445475 23147 sgd_solver.cpp:106] Iteration 145400, lr = 2.5553e-05
    I0515 03:48:20.535774 23147 solver.cpp:228] Iteration 145500, loss = 0.0110161
    I0515 03:48:20.535835 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:48:20.535863 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0110159 (* 1 = 0.0110159 loss)
    I0515 03:48:20.535886 23147 sgd_solver.cpp:106] Iteration 145500, lr = 2.55407e-05
    I0515 03:48:28.623816 23147 solver.cpp:228] Iteration 145600, loss = 0.0250949
    I0515 03:48:28.623913 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:48:28.623944 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0250948 (* 1 = 0.0250948 loss)
    I0515 03:48:28.623965 23147 sgd_solver.cpp:106] Iteration 145600, lr = 2.55284e-05
    I0515 03:48:36.714694 23147 solver.cpp:228] Iteration 145700, loss = 0.0135533
    I0515 03:48:36.714753 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:48:36.714783 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0135531 (* 1 = 0.0135531 loss)
    I0515 03:48:36.714804 23147 sgd_solver.cpp:106] Iteration 145700, lr = 2.55161e-05
    I0515 03:48:44.807113 23147 solver.cpp:228] Iteration 145800, loss = 0.0109819
    I0515 03:48:44.807168 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:48:44.807199 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0109818 (* 1 = 0.0109818 loss)
    I0515 03:48:44.807220 23147 sgd_solver.cpp:106] Iteration 145800, lr = 2.55038e-05
    I0515 03:48:52.892644 23147 solver.cpp:228] Iteration 145900, loss = 0.0157928
    I0515 03:48:52.892705 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:48:52.892735 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0157926 (* 1 = 0.0157926 loss)
    I0515 03:48:52.892755 23147 sgd_solver.cpp:106] Iteration 145900, lr = 2.54915e-05
    I0515 03:49:00.899741 23147 solver.cpp:337] Iteration 146000, Testing net (#0)
    I0515 03:49:05.254815 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.81525
    I0515 03:49:05.254884 23147 solver.cpp:404]     Test net output #1: loss_c = 0.550638 (* 1 = 0.550638 loss)
    I0515 03:49:05.310425 23147 solver.cpp:228] Iteration 146000, loss = 0.00959614
    I0515 03:49:05.310482 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:49:05.310514 23147 solver.cpp:244]     Train net output #1: loss_c = 0.009596 (* 1 = 0.009596 loss)
    I0515 03:49:05.310539 23147 sgd_solver.cpp:106] Iteration 146000, lr = 2.54792e-05
    I0515 03:49:13.408655 23147 solver.cpp:228] Iteration 146100, loss = 0.0132174
    I0515 03:49:13.408704 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:49:13.408732 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0132173 (* 1 = 0.0132173 loss)
    I0515 03:49:13.408753 23147 sgd_solver.cpp:106] Iteration 146100, lr = 2.5467e-05
    I0515 03:49:21.496778 23147 solver.cpp:228] Iteration 146200, loss = 0.0162468
    I0515 03:49:21.496829 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:49:21.496857 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0162467 (* 1 = 0.0162467 loss)
    I0515 03:49:21.496878 23147 sgd_solver.cpp:106] Iteration 146200, lr = 2.54548e-05
    I0515 03:49:29.585086 23147 solver.cpp:228] Iteration 146300, loss = 0.00858785
    I0515 03:49:29.585142 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:49:29.585171 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00858772 (* 1 = 0.00858772 loss)
    I0515 03:49:29.585192 23147 sgd_solver.cpp:106] Iteration 146300, lr = 2.54426e-05
    I0515 03:49:37.673883 23147 solver.cpp:228] Iteration 146400, loss = 0.0163236
    I0515 03:49:37.674018 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:49:37.674065 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0163234 (* 1 = 0.0163234 loss)
    I0515 03:49:37.674089 23147 sgd_solver.cpp:106] Iteration 146400, lr = 2.54304e-05
    I0515 03:49:45.727972 23147 solver.cpp:228] Iteration 146500, loss = 0.0485337
    I0515 03:49:45.728024 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:49:45.728044 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0485336 (* 1 = 0.0485336 loss)
    I0515 03:49:45.728060 23147 sgd_solver.cpp:106] Iteration 146500, lr = 2.54182e-05
    I0515 03:49:53.848793 23147 solver.cpp:228] Iteration 146600, loss = 0.0086057
    I0515 03:49:53.848845 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:49:53.848867 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00860556 (* 1 = 0.00860556 loss)
    I0515 03:49:53.848882 23147 sgd_solver.cpp:106] Iteration 146600, lr = 2.5406e-05
    I0515 03:50:01.975060 23147 solver.cpp:228] Iteration 146700, loss = 0.0140756
    I0515 03:50:01.975112 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:50:01.975132 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0140755 (* 1 = 0.0140755 loss)
    I0515 03:50:01.975147 23147 sgd_solver.cpp:106] Iteration 146700, lr = 2.53938e-05
    I0515 03:50:10.112730 23147 solver.cpp:228] Iteration 146800, loss = 0.0100152
    I0515 03:50:10.112828 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:50:10.112851 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0100151 (* 1 = 0.0100151 loss)
    I0515 03:50:10.112866 23147 sgd_solver.cpp:106] Iteration 146800, lr = 2.53817e-05
    I0515 03:50:18.254019 23147 solver.cpp:228] Iteration 146900, loss = 0.0205345
    I0515 03:50:18.254070 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:50:18.254089 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0205344 (* 1 = 0.0205344 loss)
    I0515 03:50:18.254104 23147 sgd_solver.cpp:106] Iteration 146900, lr = 2.53696e-05
    I0515 03:50:26.304404 23147 solver.cpp:337] Iteration 147000, Testing net (#0)
    I0515 03:50:30.712347 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.804834
    I0515 03:50:30.712396 23147 solver.cpp:404]     Test net output #1: loss_c = 0.569646 (* 1 = 0.569646 loss)
    I0515 03:50:30.765669 23147 solver.cpp:228] Iteration 147000, loss = 0.00748563
    I0515 03:50:30.765740 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:50:30.765766 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0074855 (* 1 = 0.0074855 loss)
    I0515 03:50:30.765784 23147 sgd_solver.cpp:106] Iteration 147000, lr = 2.53574e-05
    I0515 03:50:38.904469 23147 solver.cpp:228] Iteration 147100, loss = 0.0309914
    I0515 03:50:38.904508 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:50:38.904530 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0309913 (* 1 = 0.0309913 loss)
    I0515 03:50:38.904544 23147 sgd_solver.cpp:106] Iteration 147100, lr = 2.53453e-05
    I0515 03:50:47.047452 23147 solver.cpp:228] Iteration 147200, loss = 0.0131974
    I0515 03:50:47.047703 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:50:47.047747 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0131973 (* 1 = 0.0131973 loss)
    I0515 03:50:47.047772 23147 sgd_solver.cpp:106] Iteration 147200, lr = 2.53332e-05
    I0515 03:50:55.184231 23147 solver.cpp:228] Iteration 147300, loss = 0.0170068
    I0515 03:50:55.184278 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:50:55.184298 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0170067 (* 1 = 0.0170067 loss)
    I0515 03:50:55.184312 23147 sgd_solver.cpp:106] Iteration 147300, lr = 2.53212e-05
    I0515 03:51:03.324766 23147 solver.cpp:228] Iteration 147400, loss = 0.00891404
    I0515 03:51:03.324815 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:51:03.324833 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00891392 (* 1 = 0.00891392 loss)
    I0515 03:51:03.324848 23147 sgd_solver.cpp:106] Iteration 147400, lr = 2.53091e-05
    I0515 03:51:11.460145 23147 solver.cpp:228] Iteration 147500, loss = 0.0104323
    I0515 03:51:11.460188 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:51:11.460208 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0104322 (* 1 = 0.0104322 loss)
    I0515 03:51:11.460222 23147 sgd_solver.cpp:106] Iteration 147500, lr = 2.5297e-05
    I0515 03:51:19.596321 23147 solver.cpp:228] Iteration 147600, loss = 0.0180917
    I0515 03:51:19.596427 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:51:19.596448 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0180916 (* 1 = 0.0180916 loss)
    I0515 03:51:19.596462 23147 sgd_solver.cpp:106] Iteration 147600, lr = 2.5285e-05
    I0515 03:51:27.733916 23147 solver.cpp:228] Iteration 147700, loss = 0.0218851
    I0515 03:51:27.733963 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:51:27.733981 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0218849 (* 1 = 0.0218849 loss)
    I0515 03:51:27.733996 23147 sgd_solver.cpp:106] Iteration 147700, lr = 2.5273e-05
    I0515 03:51:35.867013 23147 solver.cpp:228] Iteration 147800, loss = 0.00896534
    I0515 03:51:35.867055 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:51:35.867075 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00896524 (* 1 = 0.00896524 loss)
    I0515 03:51:35.867089 23147 sgd_solver.cpp:106] Iteration 147800, lr = 2.5261e-05
    I0515 03:51:44.002177 23147 solver.cpp:228] Iteration 147900, loss = 0.00667513
    I0515 03:51:44.002223 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:51:44.002243 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00667503 (* 1 = 0.00667503 loss)
    I0515 03:51:44.002259 23147 sgd_solver.cpp:106] Iteration 147900, lr = 2.5249e-05
    I0515 03:51:52.048676 23147 solver.cpp:337] Iteration 148000, Testing net (#0)
    I0515 03:51:56.473301 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.80475
    I0515 03:51:56.473347 23147 solver.cpp:404]     Test net output #1: loss_c = 0.565614 (* 1 = 0.565614 loss)
    I0515 03:51:56.524737 23147 solver.cpp:228] Iteration 148000, loss = 0.0249202
    I0515 03:51:56.524771 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:51:56.524791 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0249201 (* 1 = 0.0249201 loss)
    I0515 03:51:56.524809 23147 sgd_solver.cpp:106] Iteration 148000, lr = 2.5237e-05
    I0515 03:52:04.662057 23147 solver.cpp:228] Iteration 148100, loss = 0.0197574
    I0515 03:52:04.662107 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:52:04.662127 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0197573 (* 1 = 0.0197573 loss)
    I0515 03:52:04.662142 23147 sgd_solver.cpp:106] Iteration 148100, lr = 2.5225e-05
    I0515 03:52:12.794636 23147 solver.cpp:228] Iteration 148200, loss = 0.00770937
    I0515 03:52:12.794688 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:52:12.794708 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00770926 (* 1 = 0.00770926 loss)
    I0515 03:52:12.794723 23147 sgd_solver.cpp:106] Iteration 148200, lr = 2.5213e-05
    I0515 03:52:20.932113 23147 solver.cpp:228] Iteration 148300, loss = 0.0414804
    I0515 03:52:20.932164 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:52:20.932188 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0414803 (* 1 = 0.0414803 loss)
    I0515 03:52:20.932202 23147 sgd_solver.cpp:106] Iteration 148300, lr = 2.52011e-05
    I0515 03:52:29.060636 23147 solver.cpp:228] Iteration 148400, loss = 0.0205424
    I0515 03:52:29.060721 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:52:29.060745 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0205423 (* 1 = 0.0205423 loss)
    I0515 03:52:29.060760 23147 sgd_solver.cpp:106] Iteration 148400, lr = 2.51892e-05
    I0515 03:52:37.191953 23147 solver.cpp:228] Iteration 148500, loss = 0.0313715
    I0515 03:52:37.192004 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:52:37.192025 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0313714 (* 1 = 0.0313714 loss)
    I0515 03:52:37.192040 23147 sgd_solver.cpp:106] Iteration 148500, lr = 2.51772e-05
    I0515 03:52:45.300541 23147 solver.cpp:228] Iteration 148600, loss = 0.0235323
    I0515 03:52:45.300590 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:52:45.300611 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0235322 (* 1 = 0.0235322 loss)
    I0515 03:52:45.300626 23147 sgd_solver.cpp:106] Iteration 148600, lr = 2.51653e-05
    I0515 03:52:53.389056 23147 solver.cpp:228] Iteration 148700, loss = 0.0145752
    I0515 03:52:53.389112 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:52:53.389142 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0145751 (* 1 = 0.0145751 loss)
    I0515 03:52:53.389163 23147 sgd_solver.cpp:106] Iteration 148700, lr = 2.51534e-05
    I0515 03:53:01.481309 23147 solver.cpp:228] Iteration 148800, loss = 0.0148828
    I0515 03:53:01.481416 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:53:01.481446 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0148827 (* 1 = 0.0148827 loss)
    I0515 03:53:01.481468 23147 sgd_solver.cpp:106] Iteration 148800, lr = 2.51416e-05
    I0515 03:53:09.570399 23147 solver.cpp:228] Iteration 148900, loss = 0.0115713
    I0515 03:53:09.570458 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:53:09.570487 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0115712 (* 1 = 0.0115712 loss)
    I0515 03:53:09.570509 23147 sgd_solver.cpp:106] Iteration 148900, lr = 2.51297e-05
    I0515 03:53:17.576725 23147 solver.cpp:337] Iteration 149000, Testing net (#0)
    I0515 03:53:21.941941 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.816583
    I0515 03:53:21.942006 23147 solver.cpp:404]     Test net output #1: loss_c = 0.533688 (* 1 = 0.533688 loss)
    I0515 03:53:21.998245 23147 solver.cpp:228] Iteration 149000, loss = 0.0124525
    I0515 03:53:21.998302 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:53:21.998332 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0124523 (* 1 = 0.0124523 loss)
    I0515 03:53:21.998355 23147 sgd_solver.cpp:106] Iteration 149000, lr = 2.51178e-05
    I0515 03:53:30.131439 23147 solver.cpp:228] Iteration 149100, loss = 0.0212128
    I0515 03:53:30.131489 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:53:30.131512 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0212127 (* 1 = 0.0212127 loss)
    I0515 03:53:30.131527 23147 sgd_solver.cpp:106] Iteration 149100, lr = 2.5106e-05
    I0515 03:53:38.245723 23147 solver.cpp:228] Iteration 149200, loss = 0.0449529
    I0515 03:53:38.245885 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.97
    I0515 03:53:38.245906 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0449528 (* 1 = 0.0449528 loss)
    I0515 03:53:38.245936 23147 sgd_solver.cpp:106] Iteration 149200, lr = 2.50942e-05
    I0515 03:53:46.387048 23147 solver.cpp:228] Iteration 149300, loss = 0.0368354
    I0515 03:53:46.387092 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:53:46.387114 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0368353 (* 1 = 0.0368353 loss)
    I0515 03:53:46.387127 23147 sgd_solver.cpp:106] Iteration 149300, lr = 2.50823e-05
    I0515 03:53:54.526983 23147 solver.cpp:228] Iteration 149400, loss = 0.00835259
    I0515 03:53:54.527034 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:53:54.527055 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00835246 (* 1 = 0.00835246 loss)
    I0515 03:53:54.527068 23147 sgd_solver.cpp:106] Iteration 149400, lr = 2.50705e-05
    I0515 03:54:02.665812 23147 solver.cpp:228] Iteration 149500, loss = 0.0353151
    I0515 03:54:02.665865 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:54:02.665885 23147 solver.cpp:244]     Train net output #1: loss_c = 0.035315 (* 1 = 0.035315 loss)
    I0515 03:54:02.665900 23147 sgd_solver.cpp:106] Iteration 149500, lr = 2.50588e-05
    I0515 03:54:10.808208 23147 solver.cpp:228] Iteration 149600, loss = 0.0331806
    I0515 03:54:10.808322 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.99
    I0515 03:54:10.808343 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0331804 (* 1 = 0.0331804 loss)
    I0515 03:54:10.808358 23147 sgd_solver.cpp:106] Iteration 149600, lr = 2.5047e-05
    I0515 03:54:18.946816 23147 solver.cpp:228] Iteration 149700, loss = 0.0281425
    I0515 03:54:18.946871 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:54:18.946897 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0281424 (* 1 = 0.0281424 loss)
    I0515 03:54:18.946913 23147 sgd_solver.cpp:106] Iteration 149700, lr = 2.50352e-05
    I0515 03:54:27.088737 23147 solver.cpp:228] Iteration 149800, loss = 0.044687
    I0515 03:54:27.088788 23147 solver.cpp:244]     Train net output #0: accuracy_class = 0.98
    I0515 03:54:27.088807 23147 solver.cpp:244]     Train net output #1: loss_c = 0.0446869 (* 1 = 0.0446869 loss)
    I0515 03:54:27.088822 23147 sgd_solver.cpp:106] Iteration 149800, lr = 2.50235e-05
    I0515 03:54:35.230064 23147 solver.cpp:228] Iteration 149900, loss = 0.00900063
    I0515 03:54:35.230114 23147 solver.cpp:244]     Train net output #0: accuracy_class = 1
    I0515 03:54:35.230134 23147 solver.cpp:244]     Train net output #1: loss_c = 0.00900051 (* 1 = 0.00900051 loss)
    I0515 03:54:35.230150 23147 sgd_solver.cpp:106] Iteration 149900, lr = 2.50117e-05
    I0515 03:54:43.293306 23147 solver.cpp:454] Snapshotting to binary proto file dvia_train_iter_150000.caffemodel
    I0515 03:54:43.325276 23147 sgd_solver.cpp:273] Snapshotting solver state to binary proto file dvia_train_iter_150000.solverstate
    I0515 03:54:43.363188 23147 solver.cpp:317] Iteration 150000, loss = 0.00851257
    I0515 03:54:43.363240 23147 solver.cpp:337] Iteration 150000, Testing net (#0)
    I0515 03:54:47.722923 23147 solver.cpp:404]     Test net output #0: accuracy_class = 0.807083
    I0515 03:54:47.722976 23147 solver.cpp:404]     Test net output #1: loss_c = 0.558356 (* 1 = 0.558356 loss)
    I0515 03:54:47.722996 23147 solver.cpp:322] Optimization Done.
    I0515 03:54:47.723006 23147 caffe.cpp:222] Optimization Done.
    CPU times: user 2min 51s, sys: 20.4 s, total: 3min 11s
    Wall time: 3h 34min 4s


Caffe brewed. 
## Test the model completely on test data
Let's test directly in command-line:


```python
%%time
!caffe test -model dvia_test.prototxt -weights dvia_train_iter_150000.caffemodel -iterations 100
```

    I0515 11:38:54.638804 28601 caffe.cpp:246] Use CPU.
    I0515 11:38:54.845983 28601 net.cpp:49] Initializing net from parameters: 
    state {
      phase: TEST
    }
    layer {
      name: "data"
      type: "Data"
      top: "data"
      top: "label"
      transform_param {
        scale: 0.00390625
      }
      data_param {
        source: "/home/maheriya/Projects/IMAGES/dvia/png.48x48/data/dvia_48x48/dvia_val_lmdb"
        batch_size: 120
        backend: LMDB
      }
    }
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 64
        kernel_size: 5
        stride: 2
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "cccp1a"
      type: "Convolution"
      bottom: "conv1"
      top: "cccp1a"
      convolution_param {
        num_output: 42
        kernel_size: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu1a"
      type: "ReLU"
      bottom: "cccp1a"
      top: "cccp1a"
    }
    layer {
      name: "cccp1b"
      type: "Convolution"
      bottom: "cccp1a"
      top: "cccp1b"
      convolution_param {
        num_output: 32
        kernel_size: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool1"
      type: "Pooling"
      bottom: "cccp1b"
      top: "pool1"
      pooling_param {
        pool: MAX
        kernel_size: 3
        stride: 2
      }
    }
    layer {
      name: "drop1"
      type: "Dropout"
      bottom: "pool1"
      top: "pool1"
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
      convolution_param {
        num_output: 64
        kernel_size: 3
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
        kernel_size: 3
        stride: 2
      }
    }
    layer {
      name: "drop2"
      type: "Dropout"
      bottom: "pool2"
      top: "pool2"
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
      convolution_param {
        num_output: 96
        kernel_size: 3
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
        pool: AVE
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
      name: "fc1"
      type: "InnerProduct"
      bottom: "pool3"
      top: "fc1"
      inner_product_param {
        num_output: 500
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu4"
      type: "ReLU"
      bottom: "fc1"
      top: "fc1"
    }
    layer {
      name: "fc_class"
      type: "InnerProduct"
      bottom: "fc1"
      top: "fc_class"
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
      name: "loss_c"
      type: "SoftmaxWithLoss"
      bottom: "fc_class"
      bottom: "label"
      top: "loss_c"
    }
    I0515 11:38:54.846966 28601 layer_factory.hpp:77] Creating layer data
    I0515 11:38:54.848333 28601 net.cpp:91] Creating Layer data
    I0515 11:38:54.848366 28601 net.cpp:399] data -> data
    I0515 11:38:54.848423 28601 net.cpp:399] data -> label
    I0515 11:38:54.849392 28604 db_lmdb.cpp:38] Opened lmdb /home/maheriya/Projects/IMAGES/dvia/png.48x48/data/dvia_48x48/dvia_val_lmdb
    I0515 11:38:54.850031 28601 data_layer.cpp:41] output data size: 120,3,48,48
    I0515 11:38:54.858739 28601 net.cpp:141] Setting up data
    I0515 11:38:54.858878 28601 net.cpp:148] Top shape: 120 3 48 48 (829440)
    I0515 11:38:54.858903 28601 net.cpp:148] Top shape: 120 (120)
    I0515 11:38:54.858921 28601 net.cpp:156] Memory required for data: 3318240
    I0515 11:38:54.858953 28601 layer_factory.hpp:77] Creating layer label_data_1_split
    I0515 11:38:54.858990 28601 net.cpp:91] Creating Layer label_data_1_split
    I0515 11:38:54.859010 28601 net.cpp:425] label_data_1_split <- label
    I0515 11:38:54.859052 28601 net.cpp:399] label_data_1_split -> label_data_1_split_0
    I0515 11:38:54.859082 28601 net.cpp:399] label_data_1_split -> label_data_1_split_1
    I0515 11:38:54.859112 28601 net.cpp:141] Setting up label_data_1_split
    I0515 11:38:54.859132 28601 net.cpp:148] Top shape: 120 (120)
    I0515 11:38:54.859215 28601 net.cpp:148] Top shape: 120 (120)
    I0515 11:38:54.859232 28601 net.cpp:156] Memory required for data: 3319200
    I0515 11:38:54.859249 28601 layer_factory.hpp:77] Creating layer conv1
    I0515 11:38:54.859289 28601 net.cpp:91] Creating Layer conv1
    I0515 11:38:54.859307 28601 net.cpp:425] conv1 <- data
    I0515 11:38:54.859328 28601 net.cpp:399] conv1 -> conv1
    I0515 11:38:54.859472 28601 net.cpp:141] Setting up conv1
    I0515 11:38:54.859494 28601 net.cpp:148] Top shape: 120 64 22 22 (3717120)
    I0515 11:38:54.859527 28601 net.cpp:156] Memory required for data: 18187680
    I0515 11:38:54.859561 28601 layer_factory.hpp:77] Creating layer cccp1a
    I0515 11:38:54.859591 28601 net.cpp:91] Creating Layer cccp1a
    I0515 11:38:54.859608 28601 net.cpp:425] cccp1a <- conv1
    I0515 11:38:54.859629 28601 net.cpp:399] cccp1a -> cccp1a
    I0515 11:38:54.859721 28601 net.cpp:141] Setting up cccp1a
    I0515 11:38:54.859746 28601 net.cpp:148] Top shape: 120 42 22 22 (2439360)
    I0515 11:38:54.859762 28601 net.cpp:156] Memory required for data: 27945120
    I0515 11:38:54.859787 28601 layer_factory.hpp:77] Creating layer relu1a
    I0515 11:38:54.859812 28601 net.cpp:91] Creating Layer relu1a
    I0515 11:38:54.859828 28601 net.cpp:425] relu1a <- cccp1a
    I0515 11:38:54.859848 28601 net.cpp:386] relu1a -> cccp1a (in-place)
    I0515 11:38:54.859869 28601 net.cpp:141] Setting up relu1a
    I0515 11:38:54.859889 28601 net.cpp:148] Top shape: 120 42 22 22 (2439360)
    I0515 11:38:54.859905 28601 net.cpp:156] Memory required for data: 37702560
    I0515 11:38:54.859920 28601 layer_factory.hpp:77] Creating layer cccp1b
    I0515 11:38:54.859946 28601 net.cpp:91] Creating Layer cccp1b
    I0515 11:38:54.859962 28601 net.cpp:425] cccp1b <- cccp1a
    I0515 11:38:54.859987 28601 net.cpp:399] cccp1b -> cccp1b
    I0515 11:38:54.860117 28601 net.cpp:141] Setting up cccp1b
    I0515 11:38:54.860138 28601 net.cpp:148] Top shape: 120 32 22 22 (1858560)
    I0515 11:38:54.860154 28601 net.cpp:156] Memory required for data: 45136800
    I0515 11:38:54.860178 28601 layer_factory.hpp:77] Creating layer pool1
    I0515 11:38:54.860206 28601 net.cpp:91] Creating Layer pool1
    I0515 11:38:54.860224 28601 net.cpp:425] pool1 <- cccp1b
    I0515 11:38:54.860244 28601 net.cpp:399] pool1 -> pool1
    I0515 11:38:54.860288 28601 net.cpp:141] Setting up pool1
    I0515 11:38:54.860309 28601 net.cpp:148] Top shape: 120 32 11 11 (464640)
    I0515 11:38:54.860326 28601 net.cpp:156] Memory required for data: 46995360
    I0515 11:38:54.860342 28601 layer_factory.hpp:77] Creating layer drop1
    I0515 11:38:54.860364 28601 net.cpp:91] Creating Layer drop1
    I0515 11:38:54.860380 28601 net.cpp:425] drop1 <- pool1
    I0515 11:38:54.860399 28601 net.cpp:386] drop1 -> pool1 (in-place)
    I0515 11:38:54.860424 28601 net.cpp:141] Setting up drop1
    I0515 11:38:54.860445 28601 net.cpp:148] Top shape: 120 32 11 11 (464640)
    I0515 11:38:54.860460 28601 net.cpp:156] Memory required for data: 48853920
    I0515 11:38:54.860476 28601 layer_factory.hpp:77] Creating layer relu1b
    I0515 11:38:54.860501 28601 net.cpp:91] Creating Layer relu1b
    I0515 11:38:54.860517 28601 net.cpp:425] relu1b <- pool1
    I0515 11:38:54.860537 28601 net.cpp:386] relu1b -> pool1 (in-place)
    I0515 11:38:54.860556 28601 net.cpp:141] Setting up relu1b
    I0515 11:38:54.860574 28601 net.cpp:148] Top shape: 120 32 11 11 (464640)
    I0515 11:38:54.860590 28601 net.cpp:156] Memory required for data: 50712480
    I0515 11:38:54.860606 28601 layer_factory.hpp:77] Creating layer conv2
    I0515 11:38:54.860635 28601 net.cpp:91] Creating Layer conv2
    I0515 11:38:54.860651 28601 net.cpp:425] conv2 <- pool1
    I0515 11:38:54.860671 28601 net.cpp:399] conv2 -> conv2
    I0515 11:38:54.860980 28601 net.cpp:141] Setting up conv2
    I0515 11:38:54.861003 28601 net.cpp:148] Top shape: 120 64 9 9 (622080)
    I0515 11:38:54.861019 28601 net.cpp:156] Memory required for data: 53200800
    I0515 11:38:54.861040 28601 layer_factory.hpp:77] Creating layer pool2
    I0515 11:38:54.861064 28601 net.cpp:91] Creating Layer pool2
    I0515 11:38:54.861080 28601 net.cpp:425] pool2 <- conv2
    I0515 11:38:54.861100 28601 net.cpp:399] pool2 -> pool2
    I0515 11:38:54.861125 28601 net.cpp:141] Setting up pool2
    I0515 11:38:54.861166 28601 net.cpp:148] Top shape: 120 64 4 4 (122880)
    I0515 11:38:54.861182 28601 net.cpp:156] Memory required for data: 53692320
    I0515 11:38:54.861199 28601 layer_factory.hpp:77] Creating layer drop2
    I0515 11:38:54.861222 28601 net.cpp:91] Creating Layer drop2
    I0515 11:38:54.861239 28601 net.cpp:425] drop2 <- pool2
    I0515 11:38:54.861258 28601 net.cpp:386] drop2 -> pool2 (in-place)
    I0515 11:38:54.861279 28601 net.cpp:141] Setting up drop2
    I0515 11:38:54.861299 28601 net.cpp:148] Top shape: 120 64 4 4 (122880)
    I0515 11:38:54.861315 28601 net.cpp:156] Memory required for data: 54183840
    I0515 11:38:54.861331 28601 layer_factory.hpp:77] Creating layer relu2
    I0515 11:38:54.861354 28601 net.cpp:91] Creating Layer relu2
    I0515 11:38:54.861371 28601 net.cpp:425] relu2 <- pool2
    I0515 11:38:54.861390 28601 net.cpp:386] relu2 -> pool2 (in-place)
    I0515 11:38:54.861412 28601 net.cpp:141] Setting up relu2
    I0515 11:38:54.861430 28601 net.cpp:148] Top shape: 120 64 4 4 (122880)
    I0515 11:38:54.861446 28601 net.cpp:156] Memory required for data: 54675360
    I0515 11:38:54.861462 28601 layer_factory.hpp:77] Creating layer conv3
    I0515 11:38:54.861487 28601 net.cpp:91] Creating Layer conv3
    I0515 11:38:54.861505 28601 net.cpp:425] conv3 <- pool2
    I0515 11:38:54.861526 28601 net.cpp:399] conv3 -> conv3
    I0515 11:38:54.862351 28601 net.cpp:141] Setting up conv3
    I0515 11:38:54.862373 28601 net.cpp:148] Top shape: 120 96 2 2 (46080)
    I0515 11:38:54.862390 28601 net.cpp:156] Memory required for data: 54859680
    I0515 11:38:54.862416 28601 layer_factory.hpp:77] Creating layer pool3
    I0515 11:38:54.862439 28601 net.cpp:91] Creating Layer pool3
    I0515 11:38:54.862457 28601 net.cpp:425] pool3 <- conv3
    I0515 11:38:54.862478 28601 net.cpp:399] pool3 -> pool3
    I0515 11:38:54.862501 28601 net.cpp:141] Setting up pool3
    I0515 11:38:54.862520 28601 net.cpp:148] Top shape: 120 96 1 1 (11520)
    I0515 11:38:54.862537 28601 net.cpp:156] Memory required for data: 54905760
    I0515 11:38:54.862553 28601 layer_factory.hpp:77] Creating layer relu3
    I0515 11:38:54.862572 28601 net.cpp:91] Creating Layer relu3
    I0515 11:38:54.862588 28601 net.cpp:425] relu3 <- pool3
    I0515 11:38:54.862610 28601 net.cpp:386] relu3 -> pool3 (in-place)
    I0515 11:38:54.862632 28601 net.cpp:141] Setting up relu3
    I0515 11:38:54.862651 28601 net.cpp:148] Top shape: 120 96 1 1 (11520)
    I0515 11:38:54.862668 28601 net.cpp:156] Memory required for data: 54951840
    I0515 11:38:54.862684 28601 layer_factory.hpp:77] Creating layer fc1
    I0515 11:38:54.862710 28601 net.cpp:91] Creating Layer fc1
    I0515 11:38:54.862726 28601 net.cpp:425] fc1 <- pool3
    I0515 11:38:54.862746 28601 net.cpp:399] fc1 -> fc1
    I0515 11:38:54.863464 28601 net.cpp:141] Setting up fc1
    I0515 11:38:54.863487 28601 net.cpp:148] Top shape: 120 500 (60000)
    I0515 11:38:54.863509 28601 net.cpp:156] Memory required for data: 55191840
    I0515 11:38:54.863533 28601 layer_factory.hpp:77] Creating layer relu4
    I0515 11:38:54.863553 28601 net.cpp:91] Creating Layer relu4
    I0515 11:38:54.863569 28601 net.cpp:425] relu4 <- fc1
    I0515 11:38:54.863587 28601 net.cpp:386] relu4 -> fc1 (in-place)
    I0515 11:38:54.863607 28601 net.cpp:141] Setting up relu4
    I0515 11:38:54.863626 28601 net.cpp:148] Top shape: 120 500 (60000)
    I0515 11:38:54.863642 28601 net.cpp:156] Memory required for data: 55431840
    I0515 11:38:54.863659 28601 layer_factory.hpp:77] Creating layer fc_class
    I0515 11:38:54.863682 28601 net.cpp:91] Creating Layer fc_class
    I0515 11:38:54.863698 28601 net.cpp:425] fc_class <- fc1
    I0515 11:38:54.863718 28601 net.cpp:399] fc_class -> fc_class
    I0515 11:38:54.863786 28601 net.cpp:141] Setting up fc_class
    I0515 11:38:54.863806 28601 net.cpp:148] Top shape: 120 4 (480)
    I0515 11:38:54.863821 28601 net.cpp:156] Memory required for data: 55433760
    I0515 11:38:54.863842 28601 layer_factory.hpp:77] Creating layer fc_class_fc_class_0_split
    I0515 11:38:54.863862 28601 net.cpp:91] Creating Layer fc_class_fc_class_0_split
    I0515 11:38:54.863878 28601 net.cpp:425] fc_class_fc_class_0_split <- fc_class
    I0515 11:38:54.863901 28601 net.cpp:399] fc_class_fc_class_0_split -> fc_class_fc_class_0_split_0
    I0515 11:38:54.863945 28601 net.cpp:399] fc_class_fc_class_0_split -> fc_class_fc_class_0_split_1
    I0515 11:38:54.863970 28601 net.cpp:141] Setting up fc_class_fc_class_0_split
    I0515 11:38:54.863989 28601 net.cpp:148] Top shape: 120 4 (480)
    I0515 11:38:54.864007 28601 net.cpp:148] Top shape: 120 4 (480)
    I0515 11:38:54.864023 28601 net.cpp:156] Memory required for data: 55437600
    I0515 11:38:54.864039 28601 layer_factory.hpp:77] Creating layer accuracy_class
    I0515 11:38:54.864065 28601 net.cpp:91] Creating Layer accuracy_class
    I0515 11:38:54.864083 28601 net.cpp:425] accuracy_class <- fc_class_fc_class_0_split_0
    I0515 11:38:54.864100 28601 net.cpp:425] accuracy_class <- label_data_1_split_0
    I0515 11:38:54.864120 28601 net.cpp:399] accuracy_class -> accuracy_class
    I0515 11:38:54.864145 28601 net.cpp:141] Setting up accuracy_class
    I0515 11:38:54.864166 28601 net.cpp:148] Top shape: (1)
    I0515 11:38:54.864181 28601 net.cpp:156] Memory required for data: 55437604
    I0515 11:38:54.864197 28601 layer_factory.hpp:77] Creating layer loss_c
    I0515 11:38:54.864222 28601 net.cpp:91] Creating Layer loss_c
    I0515 11:38:54.864238 28601 net.cpp:425] loss_c <- fc_class_fc_class_0_split_1
    I0515 11:38:54.864256 28601 net.cpp:425] loss_c <- label_data_1_split_1
    I0515 11:38:54.864277 28601 net.cpp:399] loss_c -> loss_c
    I0515 11:38:54.864303 28601 layer_factory.hpp:77] Creating layer loss_c
    I0515 11:38:54.864347 28601 net.cpp:141] Setting up loss_c
    I0515 11:38:54.864372 28601 net.cpp:148] Top shape: (1)
    I0515 11:38:54.864388 28601 net.cpp:151]     with loss weight 1
    I0515 11:38:54.864442 28601 net.cpp:156] Memory required for data: 55437608
    I0515 11:38:54.864459 28601 net.cpp:217] loss_c needs backward computation.
    I0515 11:38:54.864475 28601 net.cpp:219] accuracy_class does not need backward computation.
    I0515 11:38:54.864492 28601 net.cpp:217] fc_class_fc_class_0_split needs backward computation.
    I0515 11:38:54.864508 28601 net.cpp:217] fc_class needs backward computation.
    I0515 11:38:54.864524 28601 net.cpp:217] relu4 needs backward computation.
    I0515 11:38:54.864539 28601 net.cpp:217] fc1 needs backward computation.
    I0515 11:38:54.864558 28601 net.cpp:217] relu3 needs backward computation.
    I0515 11:38:54.864574 28601 net.cpp:217] pool3 needs backward computation.
    I0515 11:38:54.864591 28601 net.cpp:217] conv3 needs backward computation.
    I0515 11:38:54.864608 28601 net.cpp:217] relu2 needs backward computation.
    I0515 11:38:54.864624 28601 net.cpp:217] drop2 needs backward computation.
    I0515 11:38:54.864639 28601 net.cpp:217] pool2 needs backward computation.
    I0515 11:38:54.864655 28601 net.cpp:217] conv2 needs backward computation.
    I0515 11:38:54.864671 28601 net.cpp:217] relu1b needs backward computation.
    I0515 11:38:54.864691 28601 net.cpp:217] drop1 needs backward computation.
    I0515 11:38:54.864707 28601 net.cpp:217] pool1 needs backward computation.
    I0515 11:38:54.864723 28601 net.cpp:217] cccp1b needs backward computation.
    I0515 11:38:54.864740 28601 net.cpp:217] relu1a needs backward computation.
    I0515 11:38:54.864756 28601 net.cpp:217] cccp1a needs backward computation.
    I0515 11:38:54.864773 28601 net.cpp:217] conv1 needs backward computation.
    I0515 11:38:54.864794 28601 net.cpp:219] label_data_1_split does not need backward computation.
    I0515 11:38:54.864811 28601 net.cpp:219] data does not need backward computation.
    I0515 11:38:54.864826 28601 net.cpp:261] This network produces output accuracy_class
    I0515 11:38:54.864845 28601 net.cpp:261] This network produces output loss_c
    I0515 11:38:54.864886 28601 net.cpp:274] Network initialization done.
    I0515 11:38:54.866160 28601 caffe.cpp:252] Running for 100 iterations.
    I0515 11:38:54.866209 28601 blocking_queue.cpp:50] Data layer prefetch queue empty
    I0515 11:38:56.229737 28601 caffe.cpp:275] Batch 0, accuracy_class = 0.816667
    I0515 11:38:56.229786 28601 caffe.cpp:275] Batch 0, loss_c = 0.471206
    I0515 11:38:57.392223 28601 caffe.cpp:275] Batch 1, accuracy_class = 0.808333
    I0515 11:38:57.392271 28601 caffe.cpp:275] Batch 1, loss_c = 0.488403
    I0515 11:38:58.547792 28601 caffe.cpp:275] Batch 2, accuracy_class = 0.816667
    I0515 11:38:58.547837 28601 caffe.cpp:275] Batch 2, loss_c = 0.466926
    I0515 11:38:59.701941 28601 caffe.cpp:275] Batch 3, accuracy_class = 0.85
    I0515 11:38:59.701992 28601 caffe.cpp:275] Batch 3, loss_c = 0.395092
    I0515 11:39:00.859534 28601 caffe.cpp:275] Batch 4, accuracy_class = 0.783333
    I0515 11:39:00.859591 28601 caffe.cpp:275] Batch 4, loss_c = 0.525201
    I0515 11:39:02.020304 28601 caffe.cpp:275] Batch 5, accuracy_class = 0.783333
    I0515 11:39:02.020370 28601 caffe.cpp:275] Batch 5, loss_c = 0.806289
    I0515 11:39:03.184242 28601 caffe.cpp:275] Batch 6, accuracy_class = 0.85
    I0515 11:39:03.184298 28601 caffe.cpp:275] Batch 6, loss_c = 0.496948
    I0515 11:39:04.342854 28601 caffe.cpp:275] Batch 7, accuracy_class = 0.791667
    I0515 11:39:04.342905 28601 caffe.cpp:275] Batch 7, loss_c = 0.630246
    I0515 11:39:05.513975 28601 caffe.cpp:275] Batch 8, accuracy_class = 0.825
    I0515 11:39:05.514041 28601 caffe.cpp:275] Batch 8, loss_c = 0.474167
    I0515 11:39:06.685894 28601 caffe.cpp:275] Batch 9, accuracy_class = 0.833333
    I0515 11:39:06.685945 28601 caffe.cpp:275] Batch 9, loss_c = 0.467154
    I0515 11:39:07.841236 28601 caffe.cpp:275] Batch 10, accuracy_class = 0.808333
    I0515 11:39:07.841280 28601 caffe.cpp:275] Batch 10, loss_c = 0.455633
    I0515 11:39:08.998078 28601 caffe.cpp:275] Batch 11, accuracy_class = 0.8
    I0515 11:39:08.998121 28601 caffe.cpp:275] Batch 11, loss_c = 0.548092
    I0515 11:39:10.154923 28601 caffe.cpp:275] Batch 12, accuracy_class = 0.75
    I0515 11:39:10.154973 28601 caffe.cpp:275] Batch 12, loss_c = 0.795354
    I0515 11:39:11.306586 28601 caffe.cpp:275] Batch 13, accuracy_class = 0.766667
    I0515 11:39:11.306632 28601 caffe.cpp:275] Batch 13, loss_c = 0.552017
    I0515 11:39:12.462949 28601 caffe.cpp:275] Batch 14, accuracy_class = 0.825
    I0515 11:39:12.463003 28601 caffe.cpp:275] Batch 14, loss_c = 0.461105
    I0515 11:39:13.616019 28601 caffe.cpp:275] Batch 15, accuracy_class = 0.833333
    I0515 11:39:13.616070 28601 caffe.cpp:275] Batch 15, loss_c = 0.52661
    I0515 11:39:14.772279 28601 caffe.cpp:275] Batch 16, accuracy_class = 0.783333
    I0515 11:39:14.772333 28601 caffe.cpp:275] Batch 16, loss_c = 0.521001
    I0515 11:39:15.926847 28601 caffe.cpp:275] Batch 17, accuracy_class = 0.766667
    I0515 11:39:15.926900 28601 caffe.cpp:275] Batch 17, loss_c = 0.692437
    I0515 11:39:17.078124 28601 caffe.cpp:275] Batch 18, accuracy_class = 0.841667
    I0515 11:39:17.078176 28601 caffe.cpp:275] Batch 18, loss_c = 0.476905
    I0515 11:39:18.232856 28601 caffe.cpp:275] Batch 19, accuracy_class = 0.833333
    I0515 11:39:18.232900 28601 caffe.cpp:275] Batch 19, loss_c = 0.596642
    I0515 11:39:19.442282 28601 caffe.cpp:275] Batch 20, accuracy_class = 0.775
    I0515 11:39:19.442337 28601 caffe.cpp:275] Batch 20, loss_c = 0.722597
    I0515 11:39:20.600692 28601 caffe.cpp:275] Batch 21, accuracy_class = 0.858333
    I0515 11:39:20.600746 28601 caffe.cpp:275] Batch 21, loss_c = 0.391944
    I0515 11:39:21.756937 28601 caffe.cpp:275] Batch 22, accuracy_class = 0.816667
    I0515 11:39:21.756989 28601 caffe.cpp:275] Batch 22, loss_c = 0.594939
    I0515 11:39:22.911003 28601 caffe.cpp:275] Batch 23, accuracy_class = 0.791667
    I0515 11:39:22.911051 28601 caffe.cpp:275] Batch 23, loss_c = 0.595701
    I0515 11:39:24.063243 28601 caffe.cpp:275] Batch 24, accuracy_class = 0.775
    I0515 11:39:24.063290 28601 caffe.cpp:275] Batch 24, loss_c = 0.530024
    I0515 11:39:25.216856 28601 caffe.cpp:275] Batch 25, accuracy_class = 0.841667
    I0515 11:39:25.217067 28601 caffe.cpp:275] Batch 25, loss_c = 0.458338
    I0515 11:39:26.403440 28601 caffe.cpp:275] Batch 26, accuracy_class = 0.791667
    I0515 11:39:26.403494 28601 caffe.cpp:275] Batch 26, loss_c = 0.529727
    I0515 11:39:27.554997 28601 caffe.cpp:275] Batch 27, accuracy_class = 0.833333
    I0515 11:39:27.555047 28601 caffe.cpp:275] Batch 27, loss_c = 0.418492
    I0515 11:39:28.746369 28601 caffe.cpp:275] Batch 28, accuracy_class = 0.808333
    I0515 11:39:28.746417 28601 caffe.cpp:275] Batch 28, loss_c = 0.567968
    I0515 11:39:29.904592 28601 caffe.cpp:275] Batch 29, accuracy_class = 0.808333
    I0515 11:39:29.904644 28601 caffe.cpp:275] Batch 29, loss_c = 0.6903
    I0515 11:39:31.056937 28601 caffe.cpp:275] Batch 30, accuracy_class = 0.858333
    I0515 11:39:31.056982 28601 caffe.cpp:275] Batch 30, loss_c = 0.526205
    I0515 11:39:32.210930 28601 caffe.cpp:275] Batch 31, accuracy_class = 0.775
    I0515 11:39:32.210975 28601 caffe.cpp:275] Batch 31, loss_c = 0.547397
    I0515 11:39:33.365535 28601 caffe.cpp:275] Batch 32, accuracy_class = 0.766667
    I0515 11:39:33.365576 28601 caffe.cpp:275] Batch 32, loss_c = 0.524216
    I0515 11:39:34.534677 28601 caffe.cpp:275] Batch 33, accuracy_class = 0.766667
    I0515 11:39:34.534726 28601 caffe.cpp:275] Batch 33, loss_c = 0.635452
    I0515 11:39:35.688740 28601 caffe.cpp:275] Batch 34, accuracy_class = 0.808333
    I0515 11:39:35.688791 28601 caffe.cpp:275] Batch 34, loss_c = 0.645186
    I0515 11:39:36.842947 28601 caffe.cpp:275] Batch 35, accuracy_class = 0.783333
    I0515 11:39:36.843000 28601 caffe.cpp:275] Batch 35, loss_c = 0.675602
    I0515 11:39:37.996212 28601 caffe.cpp:275] Batch 36, accuracy_class = 0.775
    I0515 11:39:37.996270 28601 caffe.cpp:275] Batch 36, loss_c = 0.809454
    I0515 11:39:39.149714 28601 caffe.cpp:275] Batch 37, accuracy_class = 0.85
    I0515 11:39:39.149770 28601 caffe.cpp:275] Batch 37, loss_c = 0.40477
    I0515 11:39:40.304852 28601 caffe.cpp:275] Batch 38, accuracy_class = 0.775
    I0515 11:39:40.304903 28601 caffe.cpp:275] Batch 38, loss_c = 0.642784
    I0515 11:39:41.456658 28601 caffe.cpp:275] Batch 39, accuracy_class = 0.841667
    I0515 11:39:41.456708 28601 caffe.cpp:275] Batch 39, loss_c = 0.535593
    I0515 11:39:42.610018 28601 caffe.cpp:275] Batch 40, accuracy_class = 0.8
    I0515 11:39:42.610079 28601 caffe.cpp:275] Batch 40, loss_c = 0.6357
    I0515 11:39:43.761297 28601 caffe.cpp:275] Batch 41, accuracy_class = 0.825
    I0515 11:39:43.761346 28601 caffe.cpp:275] Batch 41, loss_c = 0.510917
    I0515 11:39:44.915210 28601 caffe.cpp:275] Batch 42, accuracy_class = 0.816667
    I0515 11:39:44.915264 28601 caffe.cpp:275] Batch 42, loss_c = 0.618154
    I0515 11:39:46.069922 28601 caffe.cpp:275] Batch 43, accuracy_class = 0.8
    I0515 11:39:46.069972 28601 caffe.cpp:275] Batch 43, loss_c = 0.481045
    I0515 11:39:47.223844 28601 caffe.cpp:275] Batch 44, accuracy_class = 0.808333
    I0515 11:39:47.223891 28601 caffe.cpp:275] Batch 44, loss_c = 0.50078
    I0515 11:39:48.378168 28601 caffe.cpp:275] Batch 45, accuracy_class = 0.833333
    I0515 11:39:48.378216 28601 caffe.cpp:275] Batch 45, loss_c = 0.443616
    I0515 11:39:49.534816 28601 caffe.cpp:275] Batch 46, accuracy_class = 0.833333
    I0515 11:39:49.534863 28601 caffe.cpp:275] Batch 46, loss_c = 0.422958
    I0515 11:39:50.688335 28601 caffe.cpp:275] Batch 47, accuracy_class = 0.783333
    I0515 11:39:50.688382 28601 caffe.cpp:275] Batch 47, loss_c = 0.579915
    I0515 11:39:51.842715 28601 caffe.cpp:275] Batch 48, accuracy_class = 0.775
    I0515 11:39:51.842770 28601 caffe.cpp:275] Batch 48, loss_c = 0.761036
    I0515 11:39:52.993996 28601 caffe.cpp:275] Batch 49, accuracy_class = 0.866667
    I0515 11:39:52.994053 28601 caffe.cpp:275] Batch 49, loss_c = 0.475415
    I0515 11:39:54.149816 28601 caffe.cpp:275] Batch 50, accuracy_class = 0.8
    I0515 11:39:54.149875 28601 caffe.cpp:275] Batch 50, loss_c = 0.615327
    I0515 11:39:55.303182 28601 caffe.cpp:275] Batch 51, accuracy_class = 0.816667
    I0515 11:39:55.303311 28601 caffe.cpp:275] Batch 51, loss_c = 0.480191
    I0515 11:39:56.459586 28601 caffe.cpp:275] Batch 52, accuracy_class = 0.825
    I0515 11:39:56.459632 28601 caffe.cpp:275] Batch 52, loss_c = 0.474947
    I0515 11:39:57.612857 28601 caffe.cpp:275] Batch 53, accuracy_class = 0.825
    I0515 11:39:57.612907 28601 caffe.cpp:275] Batch 53, loss_c = 0.439382
    I0515 11:39:58.771347 28601 caffe.cpp:275] Batch 54, accuracy_class = 0.775
    I0515 11:39:58.771404 28601 caffe.cpp:275] Batch 54, loss_c = 0.573052
    I0515 11:39:59.929386 28601 caffe.cpp:275] Batch 55, accuracy_class = 0.75
    I0515 11:39:59.929435 28601 caffe.cpp:275] Batch 55, loss_c = 0.806667
    I0515 11:40:01.090803 28601 caffe.cpp:275] Batch 56, accuracy_class = 0.775
    I0515 11:40:01.090859 28601 caffe.cpp:275] Batch 56, loss_c = 0.536959
    I0515 11:40:02.247642 28601 caffe.cpp:275] Batch 57, accuracy_class = 0.825
    I0515 11:40:02.247694 28601 caffe.cpp:275] Batch 57, loss_c = 0.459662
    I0515 11:40:03.399092 28601 caffe.cpp:275] Batch 58, accuracy_class = 0.841667
    I0515 11:40:03.399140 28601 caffe.cpp:275] Batch 58, loss_c = 0.51818
    I0515 11:40:04.557137 28601 caffe.cpp:275] Batch 59, accuracy_class = 0.775
    I0515 11:40:04.557190 28601 caffe.cpp:275] Batch 59, loss_c = 0.610925
    I0515 11:40:05.712749 28601 caffe.cpp:275] Batch 60, accuracy_class = 0.783333
    I0515 11:40:05.712795 28601 caffe.cpp:275] Batch 60, loss_c = 0.59891
    I0515 11:40:06.864835 28601 caffe.cpp:275] Batch 61, accuracy_class = 0.841667
    I0515 11:40:06.864881 28601 caffe.cpp:275] Batch 61, loss_c = 0.46952
    I0515 11:40:08.020221 28601 caffe.cpp:275] Batch 62, accuracy_class = 0.825
    I0515 11:40:08.020268 28601 caffe.cpp:275] Batch 62, loss_c = 0.60759
    I0515 11:40:09.178206 28601 caffe.cpp:275] Batch 63, accuracy_class = 0.783333
    I0515 11:40:09.178251 28601 caffe.cpp:275] Batch 63, loss_c = 0.716222
    I0515 11:40:10.335377 28601 caffe.cpp:275] Batch 64, accuracy_class = 0.841667
    I0515 11:40:10.335432 28601 caffe.cpp:275] Batch 64, loss_c = 0.406811
    I0515 11:40:11.491799 28601 caffe.cpp:275] Batch 65, accuracy_class = 0.816667
    I0515 11:40:11.491839 28601 caffe.cpp:275] Batch 65, loss_c = 0.641359
    I0515 11:40:12.648480 28601 caffe.cpp:275] Batch 66, accuracy_class = 0.791667
    I0515 11:40:12.648525 28601 caffe.cpp:275] Batch 66, loss_c = 0.604626
    I0515 11:40:13.808816 28601 caffe.cpp:275] Batch 67, accuracy_class = 0.783333
    I0515 11:40:13.808869 28601 caffe.cpp:275] Batch 67, loss_c = 0.496144
    I0515 11:40:14.965126 28601 caffe.cpp:275] Batch 68, accuracy_class = 0.841667
    I0515 11:40:14.965180 28601 caffe.cpp:275] Batch 68, loss_c = 0.431503
    I0515 11:40:16.256688 28601 caffe.cpp:275] Batch 69, accuracy_class = 0.791667
    I0515 11:40:16.256742 28601 caffe.cpp:275] Batch 69, loss_c = 0.546386
    I0515 11:40:17.432252 28601 caffe.cpp:275] Batch 70, accuracy_class = 0.825
    I0515 11:40:17.432301 28601 caffe.cpp:275] Batch 70, loss_c = 0.416555
    I0515 11:40:18.590848 28601 caffe.cpp:275] Batch 71, accuracy_class = 0.8
    I0515 11:40:18.590901 28601 caffe.cpp:275] Batch 71, loss_c = 0.569596
    I0515 11:40:19.744510 28601 caffe.cpp:275] Batch 72, accuracy_class = 0.825
    I0515 11:40:19.744557 28601 caffe.cpp:275] Batch 72, loss_c = 0.671426
    I0515 11:40:20.904144 28601 caffe.cpp:275] Batch 73, accuracy_class = 0.866667
    I0515 11:40:20.904192 28601 caffe.cpp:275] Batch 73, loss_c = 0.516334
    I0515 11:40:22.118685 28601 caffe.cpp:275] Batch 74, accuracy_class = 0.758333
    I0515 11:40:22.118731 28601 caffe.cpp:275] Batch 74, loss_c = 0.568805
    I0515 11:40:23.703411 28601 caffe.cpp:275] Batch 75, accuracy_class = 0.775
    I0515 11:40:23.703485 28601 caffe.cpp:275] Batch 75, loss_c = 0.505188
    I0515 11:40:25.430537 28601 caffe.cpp:275] Batch 76, accuracy_class = 0.775
    I0515 11:40:25.430732 28601 caffe.cpp:275] Batch 76, loss_c = 0.631142
    I0515 11:40:26.649700 28601 caffe.cpp:275] Batch 77, accuracy_class = 0.808333
    I0515 11:40:26.649756 28601 caffe.cpp:275] Batch 77, loss_c = 0.655526
    I0515 11:40:27.810035 28601 caffe.cpp:275] Batch 78, accuracy_class = 0.766667
    I0515 11:40:27.810099 28601 caffe.cpp:275] Batch 78, loss_c = 0.702333
    I0515 11:40:28.965179 28601 caffe.cpp:275] Batch 79, accuracy_class = 0.775
    I0515 11:40:28.965229 28601 caffe.cpp:275] Batch 79, loss_c = 0.8245
    I0515 11:40:30.123010 28601 caffe.cpp:275] Batch 80, accuracy_class = 0.858333
    I0515 11:40:30.123066 28601 caffe.cpp:275] Batch 80, loss_c = 0.364406
    I0515 11:40:31.286633 28601 caffe.cpp:275] Batch 81, accuracy_class = 0.766667
    I0515 11:40:31.286681 28601 caffe.cpp:275] Batch 81, loss_c = 0.673585
    I0515 11:40:32.449627 28601 caffe.cpp:275] Batch 82, accuracy_class = 0.85
    I0515 11:40:32.449679 28601 caffe.cpp:275] Batch 82, loss_c = 0.518058
    I0515 11:40:33.611143 28601 caffe.cpp:275] Batch 83, accuracy_class = 0.8
    I0515 11:40:33.611194 28601 caffe.cpp:275] Batch 83, loss_c = 0.627194
    I0515 11:40:34.766258 28601 caffe.cpp:275] Batch 84, accuracy_class = 0.833333
    I0515 11:40:34.766309 28601 caffe.cpp:275] Batch 84, loss_c = 0.492251
    I0515 11:40:35.921571 28601 caffe.cpp:275] Batch 85, accuracy_class = 0.8
    I0515 11:40:35.921622 28601 caffe.cpp:275] Batch 85, loss_c = 0.650728
    I0515 11:40:37.080235 28601 caffe.cpp:275] Batch 86, accuracy_class = 0.8
    I0515 11:40:37.080281 28601 caffe.cpp:275] Batch 86, loss_c = 0.513491
    I0515 11:40:38.238572 28601 caffe.cpp:275] Batch 87, accuracy_class = 0.808333
    I0515 11:40:38.238622 28601 caffe.cpp:275] Batch 87, loss_c = 0.535132
    I0515 11:40:39.392933 28601 caffe.cpp:275] Batch 88, accuracy_class = 0.825
    I0515 11:40:39.392982 28601 caffe.cpp:275] Batch 88, loss_c = 0.391176
    I0515 11:40:40.549769 28601 caffe.cpp:275] Batch 89, accuracy_class = 0.85
    I0515 11:40:40.549823 28601 caffe.cpp:275] Batch 89, loss_c = 0.38691
    I0515 11:40:41.701493 28601 caffe.cpp:275] Batch 90, accuracy_class = 0.775
    I0515 11:40:41.701555 28601 caffe.cpp:275] Batch 90, loss_c = 0.597088
    I0515 11:40:42.856848 28601 caffe.cpp:275] Batch 91, accuracy_class = 0.783333
    I0515 11:40:42.856900 28601 caffe.cpp:275] Batch 91, loss_c = 0.746002
    I0515 11:40:44.062525 28601 caffe.cpp:275] Batch 92, accuracy_class = 0.85
    I0515 11:40:44.062571 28601 caffe.cpp:275] Batch 92, loss_c = 0.555904
    I0515 11:40:45.214056 28601 caffe.cpp:275] Batch 93, accuracy_class = 0.808333
    I0515 11:40:45.214107 28601 caffe.cpp:275] Batch 93, loss_c = 0.553135
    I0515 11:40:46.429575 28601 caffe.cpp:275] Batch 94, accuracy_class = 0.816667
    I0515 11:40:46.429635 28601 caffe.cpp:275] Batch 94, loss_c = 0.485951
    I0515 11:40:47.581607 28601 caffe.cpp:275] Batch 95, accuracy_class = 0.825
    I0515 11:40:47.581656 28601 caffe.cpp:275] Batch 95, loss_c = 0.468804
    I0515 11:40:48.729755 28601 caffe.cpp:275] Batch 96, accuracy_class = 0.833333
    I0515 11:40:48.729805 28601 caffe.cpp:275] Batch 96, loss_c = 0.426607
    I0515 11:40:49.879626 28601 caffe.cpp:275] Batch 97, accuracy_class = 0.775
    I0515 11:40:49.879676 28601 caffe.cpp:275] Batch 97, loss_c = 0.563038
    I0515 11:40:51.028136 28601 caffe.cpp:275] Batch 98, accuracy_class = 0.75
    I0515 11:40:51.028187 28601 caffe.cpp:275] Batch 98, loss_c = 0.819366
    I0515 11:40:52.177384 28601 caffe.cpp:275] Batch 99, accuracy_class = 0.775
    I0515 11:40:52.177430 28601 caffe.cpp:275] Batch 99, loss_c = 0.537878
    I0515 11:40:52.177446 28601 caffe.cpp:280] Loss: 0.557261
    I0515 11:40:52.177481 28601 caffe.cpp:292] accuracy_class = 0.80675
    I0515 11:40:52.177510 28601 caffe.cpp:292] loss_c = 0.557261 (* 1 = 0.557261 loss)
    CPU times: user 1.69 s, sys: 204 ms, total: 1.89 s
    Wall time: 1min 58s


## The model achieved near 80% accuracy
The above is purely test/validation database that is not used for training.


```python
!jupyter nbconvert --to markdown dvia-train.ipynb
```

    [NbConvertApp] Converting notebook custom-cifar-100.ipynb to markdown
    [NbConvertApp] Writing 731885 bytes to custom-cifar-100.md



```python

```
