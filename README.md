This is an example of the CML-EN:
The dataset included is goodbooks10k
to run it:

python CML.py

The error warning in the beginning is normal.
The reason for this error is, that if randomly intialized: 
Some values will be zero in the beginning and this throws an error.
However, this has nothing to do with the training.
It is related to the Evaluation 

To install:
python 2.7.13
tensorflow, we used the gpu version.
And the AdamOptimizer,
this can be changed but needs retuning the parameters
