import torch
import torch.nn as nn
import torch.nn.functional as F

from basicts.modules.embed import TokenEmbedding, \
    TimestampEmbedding as TemporalEmbedding



class my_Layernorm(nn.Module): #TODO: Check for double normalization
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        # for seasonal part normalization
        # make the avg value as zero
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    def __init__(self, multi_correlation, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.multi_correlation = multi_correlation
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # temporal corr
        x = x + self.multi_correlation(x, x, attn_mask)
        x, _ = self.decomp1(x)

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        for attn_layer in self.attn_layers:
            x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x #TODO: what is x? tensor or layer?


class DecoderLayer(nn.Module):
    def __init__(self, multi_correlation, cross_multi_correlation,
                 d_model, c_out, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.multi_correlation = multi_correlation
        self.cross_multi_correlation = cross_multi_correlation
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        ## stage one
        # self multi-correlation
        x = x + self.multi_correlation(x, x, x_mask)
        x, trend1 = self.decomp1(x)

        ## stage two
        # cross multi-correlation
        x = x + self.cross_multi_correlation(x, cross, cross_mask)
        x, trend2 = self.decomp2(x)

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend

class ValueEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        # Project input features to model dimension
        self.value_embedding = nn.Linear(c_in, d_model)

    def forward(self, x):
        return self.value_embedding(x)

class DataEmbedding(nn.Module):
    def __init__(self,
                 c_in,
                 d_model,
                 node_num,
                 num_spatial,
                 embed_type='fixed',
                 timestamps_features=('hourofday', 'dayofweek', 'dayofmonth', 'dayofyear'),
                 dropout=0.1
                 ):
        super(DataEmbedding, self).__init__()
        # spatial
        self.national_embedding = ValueEmbedding(3 * (num_spatial // node_num), d_model)
        # value
        self.value_embedding = TokenEmbedding(num_features=c_in, hidden_size=d_model)
        # temporal
        self.num_temporal_factor = len(timestamps_features)
        self.timestamps_feature_size ={
            "hourofday": 24,
            "dayofweek": 7,
            "dayofmonth": 32,
            "day": 32,
            "monthofyear": 13,
            "month": 13,
            "dayofyear": 366,
        }
        timestamp_sizes = [self.timestamps_feature_size[feature] for feature in timestamps_features]

        self.temporal_embedding = TemporalEmbedding(hidden_size=d_model, timestamp_sizes=timestamp_sizes) \
            if embed_type != 'timeF' else ValueEmbedding(d_model=d_model, c_in=self.num_temporal_factor)
        self.dropout = nn.Dropout(p=dropout)
        # others
        self.node_num = node_num

    def forward(self, x, x_spatial, x_mark):
        # B N T C
        B, L, D = x.shape
        national_position = x_spatial.to(x.device).contiguous() \
            .reshape(self.node_num, x_spatial.shape[0] // self.node_num, 3).view(self.node_num, -1)
        # x = x[:, :, 3:]
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + \
            self.national_embedding(national_position) \
                .unsqueeze(1).unsqueeze(0).repeat(B // self.node_num, 1, L, 1) \
                .view(B, L, -1)
        return self.dropout(x)
