import copy
import os
from subprocess import call

print("")

print("Downloading...")
if not os.path.exists("cifar-100-python.tar.gz"):
    call(
        "wget http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        shell=True
    )
    print("Downloading done.\n")
else:
    print("Dataset already downloaded. Did not download twice.\n")


print("Extracting...")
cifar_python_directory = os.path.abspath("cifar-100-python")
if not os.path.exists(cifar_python_directory):
    call(
        "tar -zxvf cifar-100-python.tar.gz",
        shell=True
    )
    print("Extracting successfully done to {}.".format(cifar_python_directory))
else:
    print("Dataset already extracted. Did not extract twice.\n")



