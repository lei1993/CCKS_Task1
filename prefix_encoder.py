"""
prompt encoder
"""
import torch


class PrefixEncoder(torch.nn.Module):
    r"""
    The Torch.nn model to encode the prefix

    Input shape: (batch_size, prefix_length)

    Output shape: (batch_size, prefix_length, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # use a two layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.model_config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.model_config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.model_config.num_hidden_layers * 2 * config.model_config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.model_config.num_hidden_layers * 2 * config.model_config.hidden_size)


    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
