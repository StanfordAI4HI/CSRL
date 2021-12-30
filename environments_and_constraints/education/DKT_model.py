import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
MASK_VALUE = -1

def get_target(y_true, y_pred):
    # Get skills and labels from y_true
    mask = 1. - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask

    skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill
    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)

    return y_true, y_pred

def custom_loss_func(y_true, y_pred):
    y_true, y_pred = get_target(y_true, y_pred)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

class MetricWrapper(tf.keras.metrics.Metric):
    def __init__(self, base_metric_class, **kwargs):
        super(MetricWrapper, self).__init__(name = base_metric_class.name, **kwargs)
        self.metric_class = base_metric_class
        
    
    def update_state(self, y_true, y_pred, sample_weight = None):
        true, pred = get_target(y_true, y_pred)
        self.metric_class.update_state(y_true = true, y_pred = pred, sample_weight = sample_weight)
    
    def result(self):
        return self.metric_class.result()
    
    def reset_states(self):
        self.metric_class.reset_states()

def get_target(y_true, y_pred):
    # Get skills and labels from y_true
    mask = 1. - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask

    skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill
    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)

    return y_true, y_pred

def custom_loss(y_true, y_pred):
    y_true, y_pred = get_target(y_true, y_pred)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

class MetricWrapper(tf.keras.metrics.Metric):
    def __init__(self, base_metric_class, **kwargs):
        super(MetricWrapper, self).__init__(name = base_metric_class.name, **kwargs)
        self.metric_class = base_metric_class
        
    
    def update_state(self, y_true, y_pred, sample_weight = None):
        true, pred = get_target(y_true, y_pred)
        self.metric_class.update_state(y_true = true, y_pred = pred, sample_weight = sample_weight)
    
    def result(self):
        return self.metric_class.result()
    
    def reset_states(self):
        self.metric_class.reset_states()


class DKT_model(object):
    def __init__(self, n_inputs, n_hidden, dropout_rate, n_outputs):
        inputs = tf.keras.Input(shape=(None, n_inputs), name='inputs')
        x = tf.keras.layers.Masking(mask_value=MASK_VALUE)(inputs)
        x = tf.keras.layers.LSTM(n_hidden, return_sequences=True, dropout=dropout_rate)(x)
        dense = tf.keras.layers.Dense(n_outputs, activation='sigmoid')
        outputs = tf.keras.layers.TimeDistributed(dense, name='outputs')(x)
        student_model = tf.keras.Model(inputs, outputs)
        self.student_model = student_model
    
    def compile_model(self, loss_func, metrics, optimizer, verbose = False):
        self.student_model.compile(
                    loss=loss_func,
                    optimizer=optimizer,
                    metrics=metrics,
                    experimental_run_tf_function=False)
    
        if verbose:
            print(self.student_model.summary())
    
    def train(self, train_dataset, val_dataset, epochs, verbose, callbacks, shuffle):
        return self.student_model.fit(x=train_dataset,
                                      epochs=epochs,
                                      verbose=verbose,
                                      callbacks=callbacks,
                                      validation_data=val_dataset,
                                      shuffle=shuffle)
    
    def load_weights(self, model_weights_file):
#         self.student_model.restore(model_weights_file).expect_partial()
        self.student_model.load_weights(model_weights_file)
    
    def evaluate(self, evaluation_dataset,  verbose, callbacks = []):
        return self.student_model.evaluate(evaluation_dataset,
                                           verbose=verbose,
                                           callbacks=callbacks)
    
    def predict(self, x):
        return self.student_model.predict(x)

#The DKT model we used in our experiments
def get_custom_DKT_model(saved_data_folder = 'saved_data'):
    all_problems_file_name = f'{saved_data_folder}/all_problems.txt'
    all_problems = []

    with open(all_problems_file_name, 'r') as filehandle:
        for line in filehandle:
            problem = line[:-1] # remove linebreak which is the last character of the string
            all_problems.append(int(problem))

    n_problems = len(all_problems)
    n_features = 2*n_problems

    MASK = -1.
    val_fraction = 0.2
    verbose = 1 # Verbose = {0,1,2}
    optimizer = "adam" # Optimizer to use
    lstm_units = 200 # Number of LSTM units
    dropout_rate = 0.1 # Dropout rate
    metrics = [MetricWrapper(tf.keras.metrics.AUC()), 
                MetricWrapper(tf.keras.metrics.BinaryAccuracy())]

    student_model = DKT_model(n_features, lstm_units, dropout_rate, n_problems)
    student_model.compile_model(custom_loss_func, metrics, optimizer, verbose)
    
    return student_model

