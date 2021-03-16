import tensorflow as tf
import pandas as pd
from servier.commands import preprocess_data


def get_model_api():
    model = tf.keras.models.load_model("/app/model", compile=False)

    def model_api(input_data):
        model_type = input_data["model_type"]
        data = input_data["data"]
        data = pd.read_json(data)
        X, _, input_dim = preprocess_data(data, model_type)
        dataset = tf.data.Dataset.from_tensor_slices(X)
        return model.predict(dataset)

    return model_api
