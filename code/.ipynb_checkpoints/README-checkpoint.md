#Code for Data Loading and Emotion Classification

## To set up the data into CSV files:
1) Make sure the dataset is downloaded somewhere, we will call it dir_path
2) set up dir_path in setup_data.py
3) `python setup_data.py`
4) Data is saved at dir_path

## To train the model
1) Create the following directories: saved_models and results_emotion
2) Train the model: `python train_model.py --dir_path dir_path --train_model --num_epochs 3`
3) Evaluate the model: `python train_model.py --dir_path dir_path --epoch_to_load epoch_to_load`


