import os

import pandas as pd
import numpy as np

from torch.utils.data.dataset import Dataset
from Utils.Embedding import HighOrderEncoding, LabelEmbedding


class SampleReader:
    def __init__(self, file_name):
        """
            file_path:
                ATF2
        """

        self.seq_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '\\' + file_name + '\\sequence\\'
        self.shape_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '\\' + file_name + '\\shape\\'
        self.histone_path = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))) + '\\' + file_name + '\\HM_101\\'

    def get_seq(self, order=1, cell_num=5, Test=False):

        if Test is False:
            original_seq = pd.read_csv(self.seq_path + 'Train.csv', sep=',', header=None)
        else:
            original_seq = pd.read_csv(self.seq_path + 'TesT.csv', sep=',', header=None)

        seq_num = original_seq.shape[0]
        seq_len = len(original_seq.loc[0, 1])

        completed_seqs = np.empty(shape=(seq_num, seq_len, 4 ** order))

        completed_labels = np.empty(shape=(seq_num, cell_num))
        for i in range(seq_num):
            completed_seqs[i] = HighOrderEncoding.embedding(
                sequence=original_seq.loc[i, 1], order=order, mapper=HighOrderEncoding.build_mapper(order=order))
            completed_labels[i] = LabelEmbedding(original_label=original_seq.loc[i, 3], cell_num=cell_num)
        completed_seqs = np.transpose(completed_seqs, [0, 2, 1])

        return completed_seqs, completed_labels

    def get_shape(self, shapes, Test=False):

        shape_series = []

        if Test is False:
            for shape in shapes:
                shape_series.append(pd.read_csv(self.shape_path + 'Train' + '_' + shape + '.csv',
                                                header=None))
        else:
            for shape in shapes:
                shape_series.append(pd.read_csv(self.shape_path + 'Test' + '_' + shape + '.csv',
                                                header=None))

        """
            seq_num = shape_series[0].shape[0]
            seq_len = shape_series[0].shape[1]
        """
        completed_shape = np.empty(shape=(shape_series[0].shape[0], len(shapes), shape_series[0].shape[1]))

        for i in range(len(shapes)):
            shape_samples = shape_series[i]
            for m in range(shape_samples.shape[0]):
                completed_shape[m][i] = shape_samples.loc[m]
        completed_shape = np.nan_to_num(completed_shape)

        return completed_shape

    def get_histone(self, Test=False):

        if Test is False:
            histone = pd.read_csv(self.histone_path + 'Train' + '.csv', header=None, index_col=None)
        else:
            histone = pd.read_csv(self.histone_path + 'Test' + '.csv', header=None, index_col=None)

        histone = histone.iloc[:, :]
        histone = histone.fillna(0)
        num = histone.shape[0] // 8
        histone = histone.values
        histone = np.array(np.split(histone, num))
        """
            mask
        """
        # histone[:, 0, :] = 0

        return histone


class SSDataset_690(Dataset):

    def __init__(self, file_name, sequence_order=1, Test=False):
        shapes = ['HelT', 'MGW', 'ProT', 'Roll']

        sample_reader = SampleReader(file_name=file_name)

        self.completed_seqs, self.completed_labels = sample_reader.get_seq(order=sequence_order, Test=Test)
        self.completed_shape = sample_reader.get_shape(shapes=shapes, Test=Test)
        self.completed_histone = sample_reader.get_histone(Test=Test)

    def __getitem__(self, item):
        return self.completed_seqs[item], self.completed_shape[item], self.completed_histone[item], \
               self.completed_labels[item]

    def __len__(self):
        return self.completed_seqs.shape[0]


# domain_dataseT_class = domain_dataseT(TF='ATF3', seq_dim=16, cell_num=5)
# SSDataset_690('USF2', sequence_order=3, Test=True)