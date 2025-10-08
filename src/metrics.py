import keras

class Preplexity(keras.metrics.Metric):
    """Custom Keras metric that measures model uncertainty by exponentiating average cross-entropy loss."""
    def __init__(self, name="Preplexity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.cross_entropy = keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        """expects y_pred to be logits"""
        ce = keras.losses.sparse_categorical_crossentropy(y_true, y_pred , from_logits=True)
        # mean over batch and seq_len dimmensions
        self.cross_entropy.update_state(ce, sample_weight=sample_weight)

    def result(self):
        return keras.ops.exp(self.cross_entropy.result())

    def reset_state(self):
        self.cross_entropy.reset_state()