# Code for Data Loading and Emotion Classification

## To set up the data into CSV files:
1) Make sure the dataset is downloaded somewhere, we will call it dir_path
2) set up dir_path in setup_data.py
3) `python setup_data.py`
4) Data is saved at dir_path

## To train the model
1) Create the following directories: saved_models and results_emotion
2) Train the model: `python train_model.py --dir_path dir_path --train_model --num_epochs 3`
3) Evaluate the model: `python train_model.py --dir_path dir_path --epoch_to_load epoch_to_load`


# Code for Different Speaker Prediction (DSP)

## Train Model (end-to-end pipeline with data splitting and evaluating)
1) select the type of model, eg, **joint training with different emotion prediction** `dsp_emo_joint.py`
2) each model uses Bert to obtain the current context embedding, Bert pre-trained model `models/bert.py`
3) train: `python dsp_emo_joint.py`
4) the script handles data splitting and tensorizing
5) Tensor logger plots loss on the train set and performance on the dev set, in `log_dir`
6) the output prints the performance on the test set
7) Performance metrics: P, R, F1 scores on each `+/-` class for DSP

## Evaluate Model
1) best model checkpoint is saved in the `model_path` directory
2) run same script used for training, eg, `dsp_emo_joint.py`
3) choose  the `pred()` function from the `main` and comment out `run()` function
4) the `pred()`function saves all the `false negative` examples => model predicts no speaker change but actually speaker changes, which might mean cultural norm violation
