#!/bin/csh -f

set DBDIR   = data/cifar_32x32
set DATA    = $HOME/Projects/IMAGES/dvia/cifar_png.32x32
set TOOLS   = /usr/local/caffe/bin
set EXTRA_OPTS = ( --encoded --encode_type=png )
#set EXTRA_OPTS = ( --encoded=false  )

set TRAIN_LIST      = "$DATA/train_new.txt"
set VAL_LIST        = "$DATA/test_new.txt"
set TRAIN_DATA_ROOT = "$DATA/"
set VAL_DATA_ROOT   = "$DATA/"
set TRAIN_LMDB      = "$DBDIR/trn_lmdb"
set VAL_LMDB        = "$DBDIR/val_lmdb"

#
echo "DB will be created in $DATA/$DBDIR"
cd $DATA
if ( ! -d $DBDIR ) then
  mkdir -p $DBDIR
endif


# Set RESIZE=true to resize the images to 32x24. Leave as false if images have
# already been resized using another tool.
set RESIZE = 0
if ($RESIZE) then
  # DVIA dimensions
  ## Aspect ratio 480x640 (portrait): Size 24x32
  #set RESIZE_HEIGHT = 32
  #set RESIZE_WIDTH  = 24
  ## LeNet dimensions
  set RESIZE_HEIGHT = 64
  set RESIZE_WIDTH  = 64
else
  set RESIZE_HEIGHT = 0
  set RESIZE_WIDTH  = 0
endif

if ( ! -d "$TRAIN_DATA_ROOT" ) then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in $0 to the path" \
       "where the training data is stored."
  exit 1
endif

if ( ! -d "$VAL_DATA_ROOT" ) then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
endif

echo "Creating training lmdb..."

if ( -d $TRAIN_LMDB ) rm -rf $TRAIN_LMDB
if ( -d $VAL_LMDB )   rm -rf $VAL_LMDB

setenv GLOG_logtostderr 1
$TOOLS/convert_imageset \
    --backend=lmdb \
    $EXTRA_OPTS \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $TRAIN_LIST \
    $TRAIN_LMDB

echo "Creating validation lmdb..."

setenv GLOG_logtostderr 1
$TOOLS/convert_imageset \
    --backend=lmdb \
    $EXTRA_OPTS \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $VAL_LIST \
    $VAL_LMDB

echo "Done."
