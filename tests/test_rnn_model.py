import torch
from rnn_model import LSTMModel

def test_lstm_forward():
    model = LSTMModel()
    x = torch.randn(2, 10, 1)
    y = model(x)
    assert y.shape == (2, 10, 1) 