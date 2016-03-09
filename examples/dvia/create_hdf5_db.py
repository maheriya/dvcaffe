#!/usr/bin/env python
"""
This script creates HDF5 database
Only greyscale images are supported.
TODO: If the HDF5 database file becomes too big, split data into
multiple batches, or use chunking
"""
import h5py, os
import caffe
import numpy as np
import argparse

def main(argv):
    # Invoke the parser
    parser = argparse.ArgumentParser()
    # Required arguments: input and output.
    parser.add_argument(
            "input_file",
            help="Input txt filename. This .txt, must be list of filenames with labels.\
            'filename, label1, label2, ....'"
            )
    parser.add_argument(
            "output_file",
            help="Output h5 filename."
            )
    parser.add_argument(
            "--size",
            type=int,
            help="Image size."
            )
    parser.add_argument(
            "--n_l",
            type=int,
            help="Number of labels."
            )
    args = parser.parse_args()

    print("Reading images...")
    if args.input_file.lower().endswith('txt'):
        with open( args.input_file, 'r' ) as T :
            lines = T.readlines()

            # Initiallize the data and label array
            X  = np.zeros( (len(lines), 1, args.size, args.size), dtype='f4' )
            y = np.zeros( (len(lines), 1, args.n_l), dtype='f4' )
            for i,l in enumerate(lines):
                l = l.strip()
                sp = l.split(' ')
                img = caffe.io.load_image( sp[0] , color=False)
                img = np.transpose( img , (2,0,1))
                X[i] = img
                y[i] = (sp[1:])
                #print y[i]

        # Transpose the X and y arrays for HDF5
        X = np.transpose( X , (2,3,0,1))
        y = np.transpose(y, (1, 2, 0))

        # Write to database hdf5
        with h5py.File(args.output_file,'w') as H:
            H.create_dataset( 'data', data=X ) # note the name data given to the dataset!
            H.create_dataset( 'label', data=y ) # note the name label given to the dataset!
    else:
        raise Exception("Unknown input file type: not in txt.")

if __name__ == "__main__":
    import sys
    main(sys.argv)
