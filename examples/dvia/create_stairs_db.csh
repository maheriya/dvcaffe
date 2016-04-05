#!/bin/csh -f

set DBDIR   = data/dvia
set DATA    = $HOME/Projects/IMAGES/dvia/png
set TOOLS   = build/tools
set EXTRA_OPTS = ( --encoded --encode_type=png )
#set EXTRA_OPTS = ( --encoded=false  )

set TRAIN_DATA_ROOT = $DATA/
set VAL_DATA_ROOT   = $DATA/

# Set RESIZE=true to resize the images to 32x24. Leave as false if images have
# already been resized using another tool.
set RESIZE = 0
if ($RESIZE) then
  # DVIA dimensions
  ## Aspect ratio 480x640 (portrait): Size 24x32
  #set RESIZE_HEIGHT = 32
  #set RESIZE_WIDTH  = 24
  ## LeNet dimensions
  set RESIZE_HEIGHT = 40
  set RESIZE_WIDTH  = 40
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

if ( -d $DBDIR/dvia_train_lmdb ) rm -rf $DBDIR/dvia_train_lmdb
if ( -d $DBDIR/dvia_val_lmdb )   rm -rf $DBDIR/dvia_val_lmdb

setenv GLOG_logtostderr 1
$TOOLS/convert_imageset \
    --backend=lmdb \
    --gray \
    $EXTRA_OPTS \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $DBDIR/dvia_train_lmdb

echo "Creating validation lmdb..."

setenv GLOG_logtostderr 1
$TOOLS/convert_imageset \
    --backend=lmdb \
    --gray \
    $EXTRA_OPTS \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $DBDIR/dvia_val_lmdb

echo "Done."
