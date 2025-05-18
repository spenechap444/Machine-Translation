import tensorflow as tf

class AnnealTeacherForcing(tf.keras.callbacks.Callback):
    def __init__(self, seq2seq_model, final_ratio, epochs=100):
        super().__init__()
        self.model = seq2seq_model
        self.final_ratio = final_ratio
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        new_ratio = 1.0 - (epoch+1)/self.epochs * (1.0 - self.final_ratio)
        self.model.teacher_forcing_ratio = max(self.final_ratio, new_ratio)