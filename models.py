import functools
import numpy as np
import tensorflow.compat.v2 as tf
from transformers import *


class Model(object):
    def __init__(self, debug=True):
        self.debug = debug
        # override with procedure to load model
        # leave the debug option open
        if self.debug:
            if self.run_self_test():
                print(" * [i] Model has loaded successfully")
            else:
                print(" * [e] An error has occured in self test!")

    def run_self_test(self):
        # leave in a simple test to see if the model runs
        # also to warm up model
        try:
            TEST_DATA = None
            _ = self.predict(TEST_DATA)
            return True
        except Exception as e:
            print(" * [e] Error:", e)
            return False

    def predict(self, input_data):
        # wrap the model.predict function here
        input_data = self.preprocess(input_data)
        return NotImplementedError

    def preprocess(self, input_data):
        # preprocessing function
        return NotImplementedError


class IntentBert(Model):
    def __init__(self, weights="bert_intent.h5", debug=True):
        MODEL = "distilbert-base-uncased"
        print(" * [i] Loading:", MODEL)
        self.debug = debug
        # load model
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL)
        self.model = TFDistilBertForSequenceClassification.from_pretrained(MODEL,
                                                                           num_labels=5)
        self.model.load_weights(weights)
        self.model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy")
        self.classes = ['make_update',
                        'none',
                        'setup_printer',
                        'shutdown_computer',
                        'software_recommendation']
        if self.debug:
            self.model.summary()
            if self.run_self_test():
                print(" * [i] Model has loaded successfully:", MODEL)
            else:
                print(" * [e] An error has occured in self test!")
                
    def run_self_test(self):
        # leave in a simple test to see if the model runs
        # also to warm up model
        try:
            TEST_DATA = "How do I turn off my computer?"
            _ = self.predict(TEST_DATA)
            return True
        except Exception as e:
            print(" * [e] Error:", e)
            return False
    
    @tf.function
    def _predict(self, input_data):
        return self.model(input_data)

    def predict(self, input_data):
        # wrap the model.predict function here
        input_data = self.preprocess(input_data)
        pred = self._predict(input_data)[0]
        pred = np.argmax(pred)
        return self.classes[pred]

    def preprocess(self, input_data):
        input_data = str(input_data).strip().lower()
        input_data = self.tokenizer.encode(input_data, add_special_tokens=False)
        input_data = tf.keras.preprocessing.sequence.pad_sequences(
            [input_data],
            maxlen=16,
            dtype='int32',
            padding='pre',
            truncating='pre',
            value=0.0
        )[0]
        print(str(input_data))
        return np.asarray([input_data], dtype="int")
    
    