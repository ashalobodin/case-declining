from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import adam_v2

from nltk.translate import bleu
import numpy as np


default_optimizer = adam_v2.Adam(learning_rate=1e-3)


class CaseDecliningModelBase:

    SAVED_MODELS_DIR = 'saved_models'

    def __init__(self, model_name, input_shape, load=True, new_model_type=Sequential):
        self.input_shape = input_shape          # (max_length, encoding_length)
        self.model_name = f'saved_models/{model_name}.h5'
        self.model = load_model(self.model_name) if load else new_model_type()
        self.batch_size = 1

    @staticmethod
    def reshape_data(data):
        data = np.array(data)
        return data.reshape((1, *data.shape))

    def fit(self, train_x, train_y, epochs=600, shuffle=True, verbose=2):
        self.model.fit(train_x, train_y, epochs=epochs, verbose=verbose, shuffle=shuffle)
        print(self.model.summary())

    def save(self):
        self.model.save(self.model_name)

    def predict(self, encoded_inp_value, verbose=0):
        if len(encoded_inp_value.shape) == 2:
            encoded_inp_value = self.reshape_data(encoded_inp_value)
        return self.model.predict(encoded_inp_value, verbose=verbose)

    # TODO: Introduce other score functions
    @staticmethod
    def bleu_score(references, hypothesis):
        return np.average(bleu([references], hypothesis, (1./3., )))

    def save_results(self):
        # TODO:
        pass
