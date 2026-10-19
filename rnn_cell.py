"""Core implementation - Simple RNN cell (LAB 05).

Một RNN cell đơn giản, tự implement bằng numpy để hiểu computation graph:

    h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)

Không dùng làm training framework, chỉ dùng để kiểm tra công thức forward
và đối chiếu với nn.RNNCell của PyTorch (xem _test_against_pytorch ở cuối file).
"""

import numpy as np


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        scale = 1.0 / np.sqrt(hidden_size)
        self.W_xh = rng.uniform(-scale, scale, size=(hidden_size, input_size))
        self.W_hh = rng.uniform(-scale, scale, size=(hidden_size, hidden_size))
        self.b_h = np.zeros(hidden_size)

    def forward(self, x_t, h_prev):
        """x_t: (input_size,), h_prev: (hidden_size,) -> h_t: (hidden_size,)."""
        a_t = self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h
        h_t = np.tanh(a_t)
        return h_t

    def forward_sequence(self, x_seq, h0=None):
        """x_seq: (T, input_size) -> h_seq: (T, hidden_size)"""
        h = np.zeros(self.hidden_size) if h0 is None else h0
        h_seq = []
        for x_t in x_seq:
            h = self.forward(x_t, h)
            h_seq.append(h)
        return np.stack(h_seq, axis=0)


def _test_against_pytorch():
    """Đối chiếu SimpleRNNCell với torch.nn.RNNCell trên cùng một bộ weight."""
    import torch
    import torch.nn as nn

    input_size, hidden_size = 3, 4
    cell = SimpleRNNCell(input_size, hidden_size, seed=42)

    torch_cell = nn.RNNCell(input_size, hidden_size, nonlinearity="tanh")
    with torch.no_grad():
        torch_cell.weight_ih.copy_(torch.tensor(cell.W_xh, dtype=torch.float32))
        torch_cell.weight_hh.copy_(torch.tensor(cell.W_hh, dtype=torch.float32))
        torch_cell.bias_ih.zero_()
        torch_cell.bias_hh.copy_(torch.tensor(cell.b_h, dtype=torch.float32))

    rng = np.random.default_rng(0)
    x_t = rng.standard_normal(input_size)
    h_prev = rng.standard_normal(hidden_size)

    h_numpy = cell.forward(x_t, h_prev)
    h_torch = torch_cell(
        torch.tensor(x_t, dtype=torch.float32).unsqueeze(0),
        torch.tensor(h_prev, dtype=torch.float32).unsqueeze(0),
    ).detach().numpy().squeeze(0)

    max_diff = np.max(np.abs(h_numpy - h_torch))
    print("h_t (numpy) :", np.round(h_numpy, 6))
    print("h_t (torch) :", np.round(h_torch, 6))
    print("sai số tối đa:", max_diff)
    assert max_diff < 1e-5, "SimpleRNNCell không khớp với nn.RNNCell"
    print("OK: SimpleRNNCell khớp với torch.nn.RNNCell (sai số < 1e-5)")


if __name__ == "__main__":
    # Ví dụ nhỏ: input_size=2, hidden_size=2, T=3
    cell = SimpleRNNCell(input_size=2, hidden_size=2, seed=0)
    x_seq = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    h_seq = cell.forward_sequence(x_seq)
    print("Input sequence:\n", x_seq)
    print("Hidden states:\n", np.round(h_seq, 6))

    print("\n--- Đối chiếu với torch.nn.RNNCell ---")
    _test_against_pytorch()
