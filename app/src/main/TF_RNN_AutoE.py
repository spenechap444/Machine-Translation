from tensorflow.keras.models import Model
from tensorflow.keras.layers import SimpleRNN, Dense
import tensorflow as tf

class EncoderRNN(Model):
    def __init__(self, enc_units):
        super(EncoderRNN, self).__init__()
        self.rnn = SimpleRNN(enc_units,
                             return_state=True,
                             name="encoder_rnn")

    def call(self, x, training=False):
        # x shape: (batch_size, timesteps, num_encoder_tokens)
        _, state_h = self.rnn(x, training=training)
        # returning the final hidden state to pass to the decoder
        return state_h

class DecoderRNN(Model):
    def __init__(self, dec_units, num_decoder_tokens):
        super(DecoderRNN, self).__init__()
        self.rnn = SimpleRNN(dec_units,
                             return_sequences=True,
                             return_state=True,
                             name = "decoder_rnn")
        self.dense = Dense(num_decoder_tokens,
                           activation='softmax',
                           name='decoder_dense')

    def call(self, x, state, training=False):
        # x shape: (batch_size, timesteps, num_decoder_tokens)
        seq_output, state_h = self.rnn(x,
                                       initial_state=state,
                                       training=training)
        output = self.dense(seq_output)
        # returning the sequence predictions and new state
        return output, state_h

class Seq2SeqRNN(Model):
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN, teacher_forcing_ratio=1.0):
        super(Seq2SeqRNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def call(self, inputs, training=False):
        enc_input, dec_input = inputs
        # Encoding the input sequence to get input for decoder
        enc_state = self.encoder(enc_input,
                                 training=training)
        # Decoder returning predicted target sequences
        dec_output, _ = self.decoder(dec_input,
                                     enc_state,
                                     training=training)

        return dec_output

    # override train step
    def train_step(self, data):
        # 1. Unpacking inputs + targets
        (inputs, dec_target) = data
        enc_input, dec_input = inputs

        # 2. Grab dynamic batch & sequence lengths

        batch_size = dec_input.shape[0]
        seq_len = dec_input.shape[1]
        vocab_size = dec_input.shape[2]
        DTYPE = dec_input.dtype
        # batch_size = tf.shape(enc_input)[0]
        # seq_len = tf.shape(dec_target)[1]
        # vocab_size = tf.shape(dec_target)[2]

        print(f'enc_input: {enc_input}')
        print(f'dec_input: {dec_input}')
        print(f'dec_target: {dec_target}')
        print(f'batch size: {batch_size}')
        print(f'vocab_size: {vocab_size}')

        with tf.GradientTape() as tape:
            # 3. Run encoder once
            enc_state = self.encoder(enc_input, training=True)

            # 4. Prepare for decoder loop
            # removing <start> token
            decoder_input_t = dec_input[:, 0:1]
            state = enc_state

            # TensorArray to collect per-step predictions
            preds_ta = tf.TensorArray(
                dtype=tf.float32,
                size=seq_len - 1,
                element_shape=tf.TensorShape([batch_size, 1, vocab_size]) # (batch_size, vocab size)
            )

            # 5. Define loop condition and body
            def cond(t, decoder_input_t, state, preds_ta):
                return t < seq_len

            def body(t, decoder_input_t, state, preds_ta):
                # 5a. one step of decoding
                pred_t, state = self.decoder(decoder_input_t, state, training=True)
                # squeezing out time dimension
                # pred_t = tf.squeeze(pred_t, axis=1)

                # 5b. Writing prediction at index (t-1)
                preds_ta = preds_ta.write(t - 1, pred_t)
                #c. deciding next input via teacher forcing
                use_teacher = tf.random.uniform([]) < self.teacher_forcing_ratio

                one_hot_pred = tf.one_hot(
                    tf.argmax(pred_t, axis=-1, output_type = tf.int32),
                    depth=vocab_size,
                    dtype=DTYPE
                ) # (batch_size, vocab_size)

                true_next = tf.expand_dims(tf.squeeze(dec_input[:, t:t+1], axis=1), axis=1)
                # gather along time-axis, then add abck the length-1 axis:
                # false_next = tf.squeeze(one_hot_pred, axis=1)
                false_next = one_hot_pred

                print(f'Decoder input shape: {true_next.shape}')
                print(f'Prediction shape: {false_next.shape}')
                next_input = tf.cond(
                    use_teacher,
                    lambda: true_next,
                    lambda: false_next
                )
                return t + 1, next_input, state, preds_ta

            # 6. Launch the while loop
            t0 = tf.constant(1) # start after the <start> token
            _, _, final_state, preds_ta = tf.while_loop(
                cond=cond,
                body=body,
                loop_vars=(t0, decoder_input_t, state, preds_ta)
            )

            # 7. Stack & permute to (batch, time, vocab)
            logits = preds_ta.stack()  # (time, batch, vocab)
            print(f'orig logits shape: {logits.shape}')
            logits = tf.transpose(logits, [1, 0, 2, 3])  # (batch, time, 1, vocab)
            logits = tf.squeeze(logits, axis=2)
            print(f'new logits shape: {logits.shape}')
            # 8. Compute loss skipping the first token
            loss = self.compiled_loss(
                dec_target[:, 1:], # true tokens (skip <start>)
                logits,
                regularization_losses=self.losses
            )

        #9. Backprop via GradientTape
        train_vars = self.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))

        #10. Update and return metrics
        self.compiled_metrics.update_state(dec_target[:, 1:], logits)
        return {m.name: m.result() for m in self.metrics}