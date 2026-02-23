cnn.py: simple convolutional neural network to read Raman spectra.

Requires:

  X_reference.npy: reference array with spectre signal coming directly from laboratory isolates         (downloaded from my database).
  
  y_reference.npy: labels (0-29) associated to the samples in X_reference.npy                           (downloaded from my database).
  
.  

-training.py: train the cnn to identify labels based on the Raman spectra.

Requires: 

  X_train_split.npy:  numpy array with 90% of the previously shuffled X_reference.npy                   (created after running cnn.py).
  
  y_train_split.npy: numpy array with 10% of the previously shuffled y_reference.npy                    (created after running cnn.py).

. 

-test.py: tests what the model has learnt using some of the reference data (same dataset used by training.py) and some clinical data.

Requires:

  X_test_split.npy: numpy array with 10% of the previously shuffled X_reference.npy                     (created after running cnn.py).
  
  y_test_split.npy: numpy array with 10% of the previously shuffled y_reference.npy                     (created after running cnn.py).
  
  X_2019clinical.npy: numpy array with clinical spectra coming from 30 patients                         (downloaded from my database).
  
  y_2019clinical.npy: numpy array with labels (0,2,3,5,6) associated to clinical samples                (downloaded from my database).
  
  simple_raman_cnn.pth: trained model                                                                   (created after running training.py).




