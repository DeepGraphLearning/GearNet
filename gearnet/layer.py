import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add

from torchdrug import layers


class IEConvLayer(nn.Module):
    eps = 1e-6

    def __init__(self, input_dim, hidden_dim, output_dim, edge_input_dim, kernel_hidden_dim=32,
                dropout=0.05, dropout_before_conv=0.2, activation="relu", aggregate_func="sum"):
        super(IEConvLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.kernel_hidden_dim = kernel_hidden_dim
        self.aggregate_func = aggregate_func

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.kernel = layers.MLP(edge_input_dim, [kernel_hidden_dim, (hidden_dim + 1) * hidden_dim])
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.input_batch_norm = nn.BatchNorm1d(input_dim)
        self.message_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.update_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.output_batch_norm = nn.BatchNorm1d(output_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout_before_conv = nn.Dropout(dropout_before_conv)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def message(self, graph, input, edge_input):
        node_in = graph.edge_list[:, 0]
        message = self.linear1(input[node_in])
        message = self.message_batch_norm(message)
        message = self.dropout_before_conv(self.activation(message))
        kernel = self.kernel(edge_input).view(-1, self.hidden_dim + 1, self.hidden_dim)
        message = torch.einsum('ijk, ik->ij', kernel[:, 1:, :], message) + kernel[:, 0, :]

        return message
    
    def aggregate(self, graph, message):
        node_in, node_out = graph.edge_list.t()[:2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        
        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node) 
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
        return update

    def combine(self, input, update):
        output = self.linear2(update)
        return output

    def forward(self, graph, input, edge_input):
        input = self.input_batch_norm(input)
        layer_input = self.dropout(self.activation(input))
        
        message = self.message(graph, layer_input, edge_input)
        update = self.aggregate(graph, message)
        update = self.dropout(self.activation(self.update_batch_norm(update)))
        
        output = self.combine(input, update)
        output = self.output_batch_norm(output)
        return output