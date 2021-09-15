# ITU - BLG 561E Deep Learning - Fall/2020
`Ozan Güldali & Cihat Kırankaya`

## Final Project

## A NOVEL CNN ARCHITECTURE TO DIAGNOSE PNEUMONIA AND ITS CAUSES FROM CHEST X-RAY

* ###### dataset may be a must for some run-configurations
    - Link to dataset: https://github.com/ozanguldali/final-covid-chestxray/blob/master/dataset
* ###### dataset_constructor.py was used to create the dataset from kaggle source
* ###### image_operations.py was used to augment the COVID-19 labeled data
* ###### visualize_layers was used to visualize the layers of proposed cnn model
* ###### To run only ML, only CNN or both as transfer learning, app.py file can be run with corresponding function parameters.

- transfer_learning: True if wanted to transfer deep features from CNN model
- load_numpy: True if wanted to use previously computed features. Default is False.
- method: "ML" or "CNN". Required if transfer_learning is False
- ml_model_name: "svm"
- cv: any positive integer. Default is 10.
- dataset_folder: "dataset"
- pretrain_file: pth file name without "pth" extension
- batch_size: any positive integer
- img_size: 224
- num_workers: 4
- cnn_model_name: "proposednet", "covidnet", "novelnet", "darkcovidnet", "alexnet", "resnet50", "squeezenet1_1". Required if method is not ML.
- optimizer_name: "Adam", "AdamW" or "SGD"
- validation_freq: any positive rational number
- lr: any positive rational number
- momentum: any positive rational number
- weight_decay: any positive rational number
- update_lr: True if wanted to periodically decrease the learning rate
- fine_tune: True if wanted to freeze first convolution block on CNN models
- num_epochs: any positive integer       
- normalize: True if wanted to normalize the data
- seed: 4

_Example of Transfer Leaning:_
1. (Dataset folder is required: https://github.com/ozanguldali/final-covid-chestxray/blob/master/dataset)
- Unless exists, 87.21_proposednet_Adam_out.pth file must be downloaded and inserted into "cnn/saved_models" directory.
- Link to file:
  - https://github.com/ozanguldali/final-covid-chestxray/blob/master/cnn/saved_models/87.21_proposednet_Adam_out.pth
    
`app.main(transfer_learning=True, ml_model_name="svm", cnn_model_name="proposednet", is_pre_trained=True,
         dataset_folder="dataset", pretrain_file="87.21_proposednet_Adam_out", seed=4)`
  
2. (Dataset folder is not needed)
- Unless exists, X_cnn.npy and y.npy files must be downloaded and inserted into project root directory.
- Link to files: 
  - https://github.com/ozanguldali/final-covid-chestxray/blob/master/X_cnn.npy
  - https://github.com/ozanguldali/final-covid-chestxray/blob/master/y.npy
    
`app.main(transfer_learning=True, load_numpy=True, seed=4)`


  
