from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class CorrformerConfig(BasicTSModelConfig):

    """
    Config class for Corrformer model.
    """

    # forecasting task
    seq_len: int = field(default=None, metadata={"help": "Input sequence length."}) # inputs given to encoder for prediction
    pred_len: int = field(default=None, metadata={"help": "Output sequence length."})
    label_len: int = field(default=None, metadata={"help": "Label length for decoder."}) # decoder uses the last label_len inputs for prediction

    # model define
    n_heads: int = field(default=8, metadata={"help": "Number of heads in multi-head attention."})
    enc_in: int = field(default=2, metadata={"help": "encoder input size, must be num_spatial / num_node"})
    dec_in: int = field(default=1, metadata={"help": "decoder input size"})
    c_out: int = field(default=7, metadata={'help': 'output size'})
    dropout: float = field(default=0.05, metadata={"help": "Dropout rate."})
    d_model: int = field(default=512, metadata={"help": "Hidden size; d_model."}) # dimensions of model
    factor: int = field(default=1, metadata={"help": "Factor in auto-correlation."}) # features, but projected into latent space, so its 1
    factor_temporal: int = field(default=1, metadata={"help": "Factor in auto-correlation. E.g. year, month, day, hour"})
    factor_spatial: int = field(default=1, metadata={"help": "Factor in auto-correlation. E.g. longitude, latitude and altitude"})
    node_num: int = field(default=100, metadata={"help": "Number of total nodes, must be num_spatial / enc_in"})
    node_list: list[int] = field(default='[23,37,40]', metadata={"help":'number of nodes on each level of the tree (Divides the stations into groups)'})
    #num_nodes = [int(x) for x in num_nodes.split(',')]
    num_spatial_features: int = field(default=3, metadata={"help": "Number of spatial features (e.g. 3 for longitude, latitude and altitude)."})
    timestamp_features: list | tuple = field(default=('hourofday', 'dayofweek', 'dayofmonth', 'dayofyear'), metadata={"help": "Timestamp_features, which will be fed at each timestamp, e.g. hourofday, dayofyear"})


    consistency_lambda: float = field(default=1.0, metadata={'help': 'consistency loss weight'})
    enc_tcn_layers: int = field(default=1, metadata={'help': 'num of enc tcn layers'})
    dec_tcn_layers: int = field(default=3, metadata={'help': 'num of dec tcn layers'})
    e_layers: int = field(default=2, metadata={'help': 'num of encoder layers'})
    d_layers: int = field(default=2, metadata={'help': 'num of decoder layers'})
    d_ff: int = field(default=2048, metadata={'help': 'dimension of fcn'})
    moving_avg: int = field(default=25, metadata={'help': 'window size of moving average'})
    distil: bool = field(default=True, metadata={'help': 'whether to use distilling in encoder, using this argument means not using distilling'})
    embed: str = field(default='timeF', metadata={'help': 'time features encoding, options:[timeF, fixed, learned]'})
    activation: str = field(default='gelu', metadata={'help': 'activation function'})
    do_predict: bool = field(default=True, metadata={'help': 'whether to predict unseen future data'})
    pretrained_model: str = field(default=None, metadata={'help': 'path to pretrained model'})

    #
    output_attention: bool = field(default=False, metadata={"help": "Whether to output attention weights in encoder."}) #action='store_true',
    num_spatial: int = field(default=None, metadata={"help": "Number of spatial datapoints (weather_stations)"})
