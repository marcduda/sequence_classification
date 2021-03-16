This Package presents a code to train a model, evaluate it and make prediction. Furthermore an app with Flask allows to make predictions and this package can also be installed with a Dockerfile.

# Installation
Since the package is using the rdkit library, we need to first have this library installed. A way to install this can be through Miniconda.
After having cloned the repository, cd to the folder and install all the necessary libraries by typing:
```
python setup.py install
```

# Description of the 3 main commands

All 3 commands can be executed the same way(`python servier command args`), for example if we want to train a model:
```
python servier train path/to/data path/to/model/save/dir --model_type Model_1
```

## Train
Expected arguments:
- A csv file containing the training data (in csv format and containing at least the columns P1 and smiles).
- A path to where the trained model will be saved.
- An option (`Model_1` or `Model_2`) specifying which type of preprocessing to do and thus which model to train (the architecture will be the same but the input size varies slightly so the model has to adapt).
  - `Model_1`: the data is preprocessed using the rdkit package encoding the string of characters into a vector.
  - `Model_2`: we use the text encoding capabilities of Tensorflow to encode each character of the input data also resulting in a vector.


  The data is preprocessed and then split between a train and a validation subset then we use a upsampling algorithm (here SMOTE) to upsample the minority class because the training data has much more examples of the class 1 than the class 0. A neural network having an LSTM layer is then trained on this data. We capture the loss and accuracy at each epoch on the training and validation data.

  Improvements on train command:
  - Test different methods for the imbalance class problem like other imputation methods, or use weights so the model pays more attention to the minority class.
  - Add arguments to make the training more flexible.
  - Add a subset in the split (train/validation and test) to be sure that we don't overtune our model to improve the performances on the validation set.
  - Add a hyperparameter search.
  - Add an early stop to prevent the model from overfitting.
  - Plot curves for the loss and accuracy during the training to better understand what's going on.

## Evaluate
Expected arguments:
- A csv file containing the data to evaluate the model (in csv format and containing at least the columns P1 and smiles).
- A path to a trained model.
- An option (`Model_1` or `Model_2`) specifying which type of preprocessing to do. It has to be the same that was used in the trained model.

The data is processed and then evaluated and a bunch of metrics are printed.

Improvements:
- Compute other metrics (true positive, true negative, false positive and false negative or f1) to have a clearer picture of the performances of the model.

## Predict
Expected arguments:
- A csv file containing the data to evaluate the model (in csv format and containing at least the column smiles)
- A path to a trained model
- An option (`Model_1` or `Model_2`) specifying which type of preprocessing to do. It has to be the same that was used in the trained model.

The data is processed either with the `Model_1` or the `Model_2` method and then we predict the output on the processed data.

# Model serving
To locally serve the model, change the path of the trained model in `serve.py` and launch the app:
```
python app.py
```
Then the interface should be available at `http://0.0.0.0:5000/`

# Dockerfile
We first need to build an image of our package (the dockerfile needs to be in the current directory):
```
 docker  build . -t servier
```
Since we didn't put the data inside we need to mount a directory in the docker image from a local directory and to execute the flask application we also need to do a mapping on the ports:
```
docker run -v /path/to/data:/app -p 5000:5000
```

# `Model_3` improvement :
For a problem where there are more than one predicting outputs (P1 to P9 for example), we have to modify the model to take that into account, some selected possibilities:
- Do N models (same as before) each predicting an output `Pi`, the training is very long though and we have to basically optimize N models.
- See it as a multi-label problem (i.e an observation can have a positive value of 1 for several outputs at the same time) and train a single model. A change to the last layer may be enough to accommodate to the new problem.
- See the problem as a sequence to sequence problem meaning that from a string of characters we want to predict the sequence of `P1`to `Pn`, the advantage is also that we train a single model but the architecture needs more adjustments.

General Improvement:
- Add error treatment (when data is not as expected for example).
- Replace prints with a proper logger.
