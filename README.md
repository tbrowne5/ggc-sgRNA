# ggc-sgRNA: Golden Gate Cloning sgRNA Prediction

This github repository contains the code to run a suite of 4 machine learning prediction models for Golden Gate Cloning sgRNA read count abundance in bacteria. While the use of independant data is not yet facilitated, you may run the test of the model. It will train the model and predict on the 20% test set.

Four models are run by this program:
<ol>
<li> Multiple-linear regression</li>
<li> CNN-based deep learning</li>
<li> BGRU RNN-based deep learning</li>
<li> Hybrid CNN-BGRU RNN-based deep learning</li>
</ol>

## Requirements

These are in a file called requirements.txt and should be in the working directory.
```
python>=3.8.8
numpy==1.19.2
numpy-base==1.19.2
biopython==1.78
h5py==2.10.0
hdf5==1.10.6
keras-preprocessing==1.1.2
pandas==1.2.2
scikit-learn==0.24.1
scipy==1.6.1
tensorflow==2.4.0
```

These can be instantiated within a conda environment:

```
conda create --name ggsgRNA python=[ACTIVE VERSION]
conda activate ggsgRNA
conda install --file requirements.txt
```

This installation has been tested in Ubuntu 20.04.4 and Mac OSX 10.14.5, but has not been tested on Windows.

## Run model test
```
python gg-sgRNA.py
```
