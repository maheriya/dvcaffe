#!/usr/bin/env sh

caffe train --solver=examples/dvia_mlc/dvia_lenet_solver.prototxt 2>&1 | tee  dvia_hdf_np_log.txt
