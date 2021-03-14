from __future__ import absolute_import, division, print_function
import numpy as np
import click
import pandas as pd
from feature_extractor import fingerprint_features
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)

@click.group()
def main():
    pass

def preprocess_data(data, model_type):
    data.drop_duplicates(subset=['mol_id'])
    # shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    if model_type =='Model_1':
        data['smiles'] = data['smiles'].apply(lambda x: fingerprint_features(x).flatten())
        data['smiles'] = data['smiles'].apply(lambda x:  np.asarray(x).astype('float32'))
        
        Y = data['P1'].to_numpy().astype(np.int32) 
        X = np.stack(data['smiles'].to_numpy())
        input_dim = np.amax(X)+1

    else : 
        # case Model_2
        seqs = data.smiles.values
        tokenizer = Tokenizer(char_level=True, filters=None, lower=False)
        tokenizer.fit_on_texts(seqs)
        # represent input data as word rank number sequences
        max_length = 74
        X = tokenizer.texts_to_sequences(seqs)
        X = sequence.pad_sequences(X, maxlen=max_length)

        Y = data['P1'].to_numpy().astype(np.int32) 
        input_dim = len(tokenizer.word_index)+1

    
    return X, Y, int(input_dim)

def create_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=input_dim,
            output_dim=64,
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') # softmax to get all the metrics below
        ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(1e-2),
        metrics=keras.metrics.BinaryAccuracy(name='accuracy') #metrics
        )
    return model

@click.command(options_metavar='<options>')
@click.option("--model_type", default=True, type=click.STRING, help="Whether training Model_1 or Model_2")
@click.argument("X", type=click.File('rb'))
@click.argument("output", type=click.STRING)
def train(x, output, model_type):
    """Train a neural network with the given X and output parameters.
    
    Arguments:\n
        [X] must be a file path to a CSV which holds the training data\n

    """

    # preprocess the data
    data = pd.read_csv(x)
    X, Y, input_dim = preprocess_data(data, model_type)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    # in the case of a direct vectorization, 
    # we can apply a SMOTE algorithm to oversample the minority class (here 0)
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # shuffle the data
    batch_size = 64
    shuffle_buffer_size = 100
    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    model = create_model(input_dim)

    history = model.fit(
        train_dataset, 
        epochs=30,
        validation_data=test_dataset
        )

    # could save the whole model or just the weights, depending on the usage
    model.save(output)

@click.command(options_metavar='<options>')
@click.argument("X", type=click.File('rb'))
@click.argument("model_path", type=click.STRING)
@click.option("--model_type", default=True, type=click.STRING)
def evaluate(x, model_path, model_type):
    """Evaluate the given network on the train and validation sets
    
    Arguments:\n
        [X] must be a file path to a CSV which holds your training data\n
    """
    data = pd.read_csv(x)
    X, Y, input_dim = preprocess_data(data, model_type)

    model = tf.keras.models.load_model(model_path)

    results = model.evaluate(X, Y)

    print(f"\n\
            Loss (BinaryCrossentropy): {results[0]}\n \
            Accuracy: {results[1]}\n"
    )

    # Print other metrics through predictions
    predictions = model.predict(X).astype(np.int32)
    Y = Y.astype(np.int32) 
    print(f"F1 Score: {f1_score(Y, predictions)}\n\
            Precision Score: {precision_score(Y, predictions)}\n\
            Recall Score: {recall_score(Y, predictions)}\n\
            AUC: {roc_auc_score(Y, predictions)}\n"
    )
    print("confusion matrix")
    print(confusion_matrix(Y, predictions))

@click.command(options_metavar='<options>')
@click.option("--model_type", default=True, type=click.STRING)
@click.argument("X", type=click.File('rb'))
@click.argument("model_path", type=click.STRING)
def predict(x, model_path, model_type):
    """
    predict an output with a given row. Prints the index of the prediction of the output row.
    
    Arguments:\n
        [x] the file that holds the 1 * n row example that should be predicted  \n
    """

    data = pd.read_csv(x)
    X, _, input_dim = preprocess_data(data, model_type)
    dataset = tf.data.Dataset.from_tensor_slices(X)
    model = tf.keras.models.load_model(model_path, compile=False)

    print(model.predict(X))

main.add_command(train)
main.add_command(evaluate)
main.add_command(predict)