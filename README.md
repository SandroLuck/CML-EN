## **CML-EN:**
This is the code for the CML-EN.
This project is based on the CML from:
https://github.com/changun/CollMetric

The dataset included is goodbooks10k:
https://github.com/zygmuntz/goodbooks-10k


----------

## Install:

Python 2.7.13
tensorflow, we used the gpu version and the AdamOptimizer.
It can also be run using the AdagradOptimizer.
Additional Installation instructions can be found:
https://github.com/changun/CollMetric


## Run

    python CML-EN.py

The error warning in the beginning is normal.
The reason for this error is, that if randomly intialized: 
Some values will be zero in the beginning and this throws an error.
However, this has nothing to do with the training.
It is related to the Evaluation.
