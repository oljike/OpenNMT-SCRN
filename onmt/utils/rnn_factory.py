"""
 RNN tools
"""
import torch.nn as nn
import onmt.models


import math
import torch
import torch.nn as nn
import torch.jit as jit
from torch.nn import Parameter



def rnn_factory(rnn_type, **kwargs):
    """ rnn factory, Use pytorch version when available. """
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        # SRU doesn't support PackedSequence.
        no_pack_padded_seq = True
        rnn = onmt.models.sru.SRU(**kwargs)

    elif rnn_type.lower() == 'custom_lstm':
        no_pack_padded_seq = True
        rnn = LSTMLayer(**kwargs)

    elif rnn_type.lower() == 'scrn':
        no_pack_padded_seq = True
        rnn = SCRNLayer(**kwargs)

    else:
        no_pack_padded_seq = True
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn, no_pack_padded_seq

class SCRNCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, context_size=40, alpha=0.95):
        super(SCRNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size - context_size
        self.context_size = context_size
        self.alpha = alpha

        self.B = Parameter(torch.randn(self.input_size, self.context_size))
        self.A = Parameter(torch.randn(self.input_size, self.hidden_size))
        self.R = Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.P = Parameter(torch.randn(self.context_size, self.hidden_size))
        self.bias_term = Parameter(torch.randn(self.hidden_size))


    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        if state is None:
            state = (torch.zeros(1, self.hidden_size).to(input.device),
                     torch.zeros(1, self.context_size).to(input.device))

        hidden_state, context_state = state

        new_context = (1.0 - self.alpha) * torch.mm(input, self.B) + self.alpha * context_state

        new_hidden = torch.sigmoid(
            torch.mm(new_context, self.P) + torch.mm(input, self.A) + torch.mm(hidden_state, self.R) + self.bias_term)

        new_output = torch.cat((new_hidden, new_context), 1)

        return new_output, (new_hidden, new_context)

class SCRNLayer(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, bidirectional=False, bias=True):
        super(SCRNLayer, self).__init__()
        self.cell = SCRNCell(input_size, hidden_size)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):

            out, state = self.cell(inputs[i], state)
            outputs += [out.unsqueeze(1)]

        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.transpose(1, 0).contiguous()

        return outputs, (state[0].unsqueeze(0), state[1].unsqueeze(0))

class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class LSTMLayer(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, bidirectional=False, bias=True):
        super(LSTMLayer, self).__init__()
        self.cell = LSTMCell(input_size, hidden_size)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):

            out, state = self.cell(inputs[i], state)
            outputs += [out.unsqueeze(1)]

        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.transpose(1, 0).contiguous()

        return outputs, (state[0].unsqueeze(0), state[1].unsqueeze(0))
#