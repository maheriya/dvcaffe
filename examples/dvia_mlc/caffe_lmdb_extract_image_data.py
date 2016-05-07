#!/usr/bin/env python
#
# Extracts an image from lmdb and displays it using eog
# Only grayscale images are supported.
#
import sys
import os
import numpy as np
import lmdb
import caffe
from caffe.proto import caffe_pb2
from pprint import pprint

# Make this True or False to choose mnist or dvia db
mnist = False
####### no need to change anything below for mnist vs dvia db

imgid = "23"
if len(sys.argv) == 2:
    imgid = sys.argv[1]




def write_png(buf, width, height):
    """ buf: must be bytes or a bytearray in py3, a regular string in py2. formatted RGBARGBA... """
    import zlib, struct
    # IHDR values
    bitDepth         = 8
    colorType        = 0  # 0: grayscale, 6 RGBA, 
    compresionMethod = 0  # 0: zip
    filterMethod     = 0
    interlaceMethod  = 0  # no interlace

    # reverse the vertical line order and add null bytes at the start
    if colorType == 6:
        width_byte_4 = width * 4
        raw_data = b''.join(b'\x00' + buf[span:span + width_byte_4]
                            for span in range((height - 1) * width * 4, -1, - width_byte_4))
    else: # Assume grayscale
        raw_data = b''.join(b'\x00' + buf[span : (span + width)]
                            for span in range(0, height * width, width))


        
 
    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (struct.pack("!I", len(data)) +
                chunk_head +
                struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))
 
    return b''.join([
        b'\x89PNG\r\n\x1a\n',
        png_pack(b'IHDR', struct.pack("!2I5B", width, height, bitDepth, colorType, compresionMethod, filterMethod, interlaceMethod)),
        png_pack(b'IDAT', zlib.compress(raw_data, 9)),
        png_pack(b'IEND', b'')])


if mnist:
    env = lmdb.open('examples/mnist/mnist_train_lmdb', readonly=True)
else:
    env = lmdb.open('data/dvia_mlc/dvia_trn_np_lmdb', readonly=True)
    #env = lmdb.open('examples/dvia_mlc/dvia_train_lmdb', readonly=True)

with env.begin() as txn:
    print "Retriving image {} from lmdb".format(imgid)
    raw_datum = txn.get(b'{i:0>8}'.format(i=imgid))

datum = caffe_pb2.Datum()
datum.ParseFromString(raw_datum)
flat_x = np.fromstring(datum.data, dtype=np.uint8)
if mnist:
    buf = write_png(datum.data, datum.width, datum.height)
else:
    buf = flat_x

imgname = "/tmp/newimage.png"
if os.path.exists(imgname): os.unlink(imgname)
with open(imgname, 'wb') as fd:
    fd.write(buf)

print "len(datum.data): ", len(datum.data)
print "Class Label: ", datum.label
print "NPX Label: ", datum.npx
print "NPY Label: ", datum.npy
print "Wrote image ", imgname
os.system("eog {}".format(imgname))

#
