'''Computes the depth of the model based on the architecture and dimensionality'''
def count_dense_params(in_dim, out_dim) -> int:
    # Dense layer = Weights + biases
    return in_dim * out_dim + out_dim

def count_rnn_params(in_dim, hidden_dim) -> int:
    # RNN layer = (in_dim + hidden_dim) * hidden_dim + biases
    return (in_dim + hidden_dim) * hidden_dim + hidden_dim

def count_lstm_params(in_dim, hidden_dim) -> int:
    # LSTM layer : 4 gates with weights and biases
    return 4 * ((in_dim + hidden_dim) * hidden_dim + hidden_dim)

