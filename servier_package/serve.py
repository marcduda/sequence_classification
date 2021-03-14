import tensorflow as tf
import pandas as pd


def get_model_api():
    model = tf.keras.models.load_model("/Users/marc/Documents/ML_TEST_SERVIER/servier_package/model", compile=False)

    def model_api(input_data):
        model_type = input_data["model_type"]
        data = input_data["data"]
        data = pd.read_json(data)
        X, _, input_dim = preprocess_data(data, model_type)
        dataset = tf.data.Dataset.from_tensor_slices(X)
        model = tf.keras.models.load_model(model_path, compile=False)
        return model.predict(X)

    return model_api