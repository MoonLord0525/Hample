import torch.nn as nn
import torch
import math


class TransformOmics(nn.Module):
    def __init__(self, original_dim, align_dim):
        super(TransformOmics, self).__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv1d(in_channels=original_dim,
                      out_channels=align_dim,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm1d(num_features=align_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, single_omics_features):
        return self.conv_1x1(single_omics_features)


class ChannelAttention(nn.Module):
    def __init__(self, channel_num):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.alpha = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)

        self.gamma = 2
        self.b = 1
        self.k = self.get_channel_numbers(channel_num=channel_num)

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.k,
                                padding=self.k // 2)
        self.sigmoid = nn.Sigmoid()

    def get_channel_numbers(self, channel_num):
        floor = math.floor((math.log2(channel_num) / self.gamma + self.b / self.gamma))
        return floor + (1 - floor % 2)

    def forward(self, x):
        F_avg = self.avg_pool(x)
        F_max = self.max_pool(x)

        # shape: bs, channel_num, 1
        F_add = self.alpha * F_avg + 1 / 2 * (F_avg + F_max) + self.beta * F_max
        # shape: bs, 1, channel_num
        F_add = F_add.permute(0, 2, 1)
        # shape: bs, channel_num, 1
        F_add = self.conv1d(F_add).permute(0, 2, 1)

        y = self.sigmoid(F_add)
        return y


class SpatialAttention(nn.Module):
    def __init__(self, channel_num):
        super(SpatialAttention, self).__init__()
        self.channel_num = channel_num
        self.lambda_ = 0.6

        self.crucial_channel_num = self.get_crucial_channel_numbers(channel_num=channel_num)
        self.subcrucial_channel_num = channel_num - self.crucial_channel_num

        self.shared_conv_layer = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=7,
                                           padding=3, bias=False)
        self.norm_active = nn.Sequential(
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def get_crucial_channel_numbers(self, channel_num):
        floor = math.floor(self.lambda_ * channel_num)
        crucial_channel_num = floor + floor % 2
        return crucial_channel_num

    def get_crucial_and_subcrucial_channels(self, crucial_channel_num, channel_map):
        # channel_map.shape: bs, channel_num, 1
        # topk.shape: bs, crucial_channel_num, 1
        _, topk = torch.topk(channel_map, dim=1, k=crucial_channel_num)

        crucial_channels = torch.zeros_like(channel_map)
        subcrucial_channels = torch.ones_like(channel_map)

        # Tensor.scatter_(dim, index, src, reduce=None) → Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor.
        crucial_channels = crucial_channels.scatter(1, topk, 1)
        subcrucial_channels = subcrucial_channels.scatter(1, topk, 0)

        return crucial_channels, subcrucial_channels

    def get_crucial_and_subcrucial_features(self, crucial_channels, subcrucial_channels, channel_refined_feature):
        crucial_features = crucial_channels * channel_refined_feature
        subcrucial_features = subcrucial_channels * channel_refined_feature
        return crucial_features, subcrucial_features

    def forward(self, x, channel_map):
        crucial_channels, subcrucial_channels = self.get_crucial_and_subcrucial_channels(
            crucial_channel_num=self.crucial_channel_num,
            channel_map=channel_map)
        crucial_features, subcrucial_features = self.get_crucial_and_subcrucial_features(
            crucial_channels=crucial_channels,
            subcrucial_channels=subcrucial_channels,
            channel_refined_feature=x)
        # keepdim = True,   (bs, channel_num, seq_len) -> (bs, 1, seq_len)
        # keepdim = False,  (bs, channel_num, seq_len) -> (bs, seq_len)
        crucial_max_pool, _ = torch.max(crucial_features, dim=1, keepdim=True)
        crucial_avg_pool = torch.mean(crucial_features, dim=1, keepdim=True) * (
                self.channel_num / self.crucial_channel_num)

        subcrucial_max_pool, _ = torch.max(subcrucial_features, dim=1, keepdim=True)
        subcrucial_avg_pool = torch.mean(subcrucial_features, dim=1, keepdim=True) * (
                self.channel_num / self.subcrucial_channel_num)

        crucial_x = torch.cat([crucial_max_pool, crucial_avg_pool], dim=1)
        subcrucial_x = torch.cat([subcrucial_max_pool, subcrucial_avg_pool], dim=1)

        A_S1 = self.norm_active(self.shared_conv_layer(crucial_x))
        A_S2 = self.norm_active(self.shared_conv_layer(subcrucial_x))

        F_S1 = crucial_features * A_S1
        F_S2 = subcrucial_features * A_S2

        spatial_refiend_features = F_S1 + F_S2

        return spatial_refiend_features


class HAMConvBlock(nn.Module):
    def __init__(self, i_dim, o_dim, kernel_size):
        super(HAMConvBlock, self).__init__()
        self.channel_num = o_dim

        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=i_dim,
                      out_channels=o_dim,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(num_features=o_dim),
            nn.ReLU(inplace=True)
        )

        self.channel_attention = ChannelAttention(channel_num=self.channel_num)
        self.spatial_attention = SpatialAttention(channel_num=self.channel_num)

    def forward(self, x):
        conv_layer = self.conv_layer(x)

        channel_map = self.channel_attention(conv_layer)
        channel_refined_features = conv_layer * channel_map

        spatial_refined_features = self.spatial_attention(channel_refined_features, channel_map)

        return spatial_refined_features


class ExpertLayer(nn.Module):
    def __init__(self, omics_feature_num, i_dim, o_dim, kernel_size):
        super(ExpertLayer, self).__init__()
        self.ham_conv_layers = nn.ModuleList(HAMConvBlock(i_dim=i_dim,
                                                          o_dim=o_dim,
                                                          kernel_size=kernel_size) for i in range(omics_feature_num))

    def forward(self, omics_features):
        hidden_features = list()
        for i, layer in enumerate(self.ham_conv_layers):
            hidden_feature = layer(omics_features[i]).unsqueeze(dim=0)
            hidden_features.append(hidden_feature)

        hidden_features = torch.cat(hidden_features, dim=0)

        return hidden_features


class TaskGate(nn.Module):
    def __init__(self, omics_feature_num, feature_dim,
                 expert_num):
        super(TaskGate, self).__init__()
        self.g_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=omics_feature_num * feature_dim,
                      out_features=expert_num),
            nn.Softmax(dim=1)
        )

    def forward(self, omics_features):
        # (omics_features_num, bs, feature_dim, seq_len) -> (bs, omics_features_num, feature_dim, seq_len)
        # (bs, omics_features_num, feature_dim, seq_len) -> (bs, omics_features_num * feature_dim, seq_len)
        omics_features_num, _, feature_dim, seq_len = omics_features.shape
        omics_features = omics_features.permute(1, 0, 2, 3).reshape(-1, omics_features_num * feature_dim, seq_len)
        # 0   - 1/3: sequence signal
        # 1/3 - 2/3: shape signal
        # 2/3 - 1  : epigenome signal
        return self.g_layer(omics_features)


class PLEBasicLayer(nn.Module):
    def __init__(self, omics_feature_num=3,
                 ham_i_dim=4, ham_o_dim=64, ham_kernel_size=15,
                 cell_num=5, mold="middle"):
        super(PLEBasicLayer, self).__init__()
        """
            mold: "middle", "final"
        """
        self.omics_feature_num = omics_feature_num
        self.mold = mold

        self.share_exp_layer = ExpertLayer(omics_feature_num=omics_feature_num,
                                           i_dim=ham_i_dim,
                                           o_dim=ham_o_dim,
                                           kernel_size=ham_kernel_size)
        self.specific_exp_layers = nn.ModuleList(ExpertLayer(omics_feature_num=omics_feature_num,
                                                             i_dim=ham_i_dim,
                                                             o_dim=ham_o_dim,
                                                             kernel_size=ham_kernel_size) for i in range(cell_num))

        if mold == "middle":
            self.share_g_layer = TaskGate(omics_feature_num=omics_feature_num,
                                          feature_dim=ham_i_dim,
                                          expert_num=omics_feature_num * (cell_num + 1))
        self.specific_g_layers = nn.ModuleList(TaskGate(omics_feature_num=omics_feature_num,
                                                        feature_dim=ham_i_dim,
                                                        expert_num=omics_feature_num * 2) for i in range(cell_num))

    def gain_g_signal(self, omics_features, layer, channel_num, seq_len):
        # shape: bs, expert_num
        g_signal = layer(omics_features)
        # shape: expert_num, bs, channel_num
        g_signal = g_signal.permute(1, 0).unsqueeze(dim=2).expand(-1, -1, channel_num)
        # shape: expert_num, bs, channel_num, seq_len
        g_signal = g_signal.unsqueeze(dim=3).expand(-1, -1, -1, seq_len)
        return g_signal

    def gain_g_refined_features(self, expert_num, share_features,
                                specific_features, g_signal):
        # 0   - 1/3: sequence features
        # 1/3 - 2/3: shape features
        # 2/3 - 1  : epigenome features
        _, bs, channel_num, seq_len = share_features.shape
        final_features = list()
        m = 0
        for i in range(self.omics_feature_num):
            for k in range(expert_num):
                if k == 0:
                    final_features.append(share_features[i] * g_signal[m])
                else:
                    if expert_num > 2:
                        final_features.append(specific_features[k - 1][i] * g_signal[m])
                    else:
                        final_features.append(specific_features[i] * g_signal[m])
                m = m + 1
        final_features = torch.cat(final_features, dim=1).unsqueeze(dim=2)
        final_features = final_features.reshape(shape=(bs, self.omics_feature_num, -1, channel_num, seq_len))
        final_features = torch.mean(final_features, dim=2).permute(1, 0, 2, 3)

        return final_features

    def forward(self, omics_features):
        final_omics_features = list()

        share_features = self.share_exp_layer(omics_features[0])
        specific_features = list()
        for p, layer in enumerate(self.specific_exp_layers):
            specific_features.append(layer(omics_features[p]))

        _, bs, channel_num, seq_len = share_features.shape

        if self.mold == "middle":
            share_g_signal = self.gain_g_signal(omics_features=omics_features[0],
                                                layer=self.share_g_layer,
                                                channel_num=channel_num, seq_len=seq_len)
            final_share_features = self.gain_g_refined_features(expert_num=len(specific_features) + 1,
                                                                share_features=share_features,
                                                                specific_features=specific_features,
                                                                g_signal=share_g_signal)
            final_omics_features.append(final_share_features)

        for p, specific_g_layer in enumerate(self.specific_g_layers):
            specific_g_signal = self.gain_g_signal(omics_features=omics_features[p],
                                                   layer=specific_g_layer,
                                                   channel_num=channel_num, seq_len=seq_len)
            final_specific_features = self.gain_g_refined_features(expert_num=2,
                                                                   share_features=share_features,
                                                                   specific_features=specific_features[p],
                                                                   g_signal=specific_g_signal)
            final_omics_features.append(final_specific_features)

        return final_omics_features


class TowerLayer(nn.Module):
    def __init__(self, i_dim, r=2, drop=0.2):
        super(TowerLayer, self).__init__()
        self.pool_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(output_size=1),
            nn.Flatten(start_dim=1)
        )

        self.feed_forward_layer = nn.Sequential(
            nn.Linear(in_features=i_dim, out_features=i_dim // r),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop),
            nn.Linear(in_features=i_dim // r, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.feed_forward_layer(self.pool_layer(x))
        return y


class PLE(nn.Module):
    def __init__(self, omics_feature_num,
                 ham_i_dim, ham_o_dim, ham_kernel_size,
                 ple_num,
                 cell_num):
        super(PLE, self).__init__()
        self.cell_num = cell_num

        self.ple_layers = nn.ModuleList()
        for i in range(ple_num - 1):
            if i == (ple_num - 1):
                mode = "final"
            else:
                mode = "middle"
            ple_layer = PLEBasicLayer(omics_feature_num=omics_feature_num,
                                      ham_i_dim=ham_i_dim[i],
                                      ham_o_dim=ham_o_dim[i],
                                      ham_kernel_size=ham_kernel_size[i],
                                      mold=mode)

            self.ple_layers.append(ple_layer)

        self.prediction_layer = nn.ModuleList([TowerLayer(i_dim=omics_feature_num * ham_o_dim[ple_num - 1])
                                               for i in range(cell_num)])

    def forward(self, sequence, shape, epigenomic):
        # sample pre_processing
        sequence = sequence.unsqueeze(dim=0)
        shape = shape.unsqueeze(dim=0)
        epigenomic = epigenomic.unsqueeze(dim=0)
        # shape (omics_features_num, bs, feature_dim (default=4), seq_len)
        omics_features = torch.cat((sequence, shape, epigenomic), dim=0)
        omics_features = [omics_features for i in range((self.cell_num + 1))]

        for layer in self.ple_layers:
            omics_features = layer(omics_features)

        prediction = list()
        for i, layer in enumerate(self.prediction_layer):
            # (omics_features_num, bs, feature_dim, seq_len) -> (bs, omics_features_num, feature_dim, seq_len)
            # (bs, omics_features_num, feature_dim, seq_len) -> (bs, omics_features_num * feature_dim, seq_len)
            omics_features_num, _, feature_dim, seq_len = omics_features[i].shape
            omics_features[i] = omics_features[i].permute(1, 0, 2, 3).reshape(-1, omics_features_num * feature_dim,
                                                                              seq_len)
            y = layer(omics_features[i])
            prediction.append(y.unsqueeze(dim=0))

        prediction = torch.cat(prediction, dim=0)

        return prediction


class Hample(nn.Module):
    def __init__(self, omics_feature_num=3,
                 sequence_dim=64, shape_dim=4, epigenomic_dim=8,
                 ham_i_dim=(4, 64), ham_o_dim=(64, 64), ham_kernel_size=(15, 9),
                 ple_num=2,
                 cell_num=5):
        super(Hample, self).__init__()
        # dimension processing for omics features
        # align_dim = 4 (default, empiric)
        align_dim = 4
        self.sequence_processing = TransformOmics(original_dim=sequence_dim, align_dim=align_dim)
        self.shape_processing = TransformOmics(original_dim=shape_dim, align_dim=align_dim)
        self.epigenomic_processing = TransformOmics(original_dim=epigenomic_dim, align_dim=align_dim)

        self.ple = PLE(omics_feature_num=omics_feature_num,
                       ham_i_dim=ham_i_dim,
                       ham_o_dim=ham_o_dim,
                       ham_kernel_size=ham_kernel_size,
                       ple_num=ple_num,
                       cell_num=cell_num)

    def forward(self, sequence, shape, epigenomic):
        sequence = self.sequence_processing(sequence)
        shape = self.shape_processing(shape)
        epigenomic = self.epigenomic_processing(epigenomic)

        prediction = self.ple(sequence, shape, epigenomic)

        return prediction


"""
model = Hample()
model(torch.ones(size=(1, 64, 101)), torch.ones(size=(1, 4, 101)), torch.ones(size=(1, 4, 101)))
"""