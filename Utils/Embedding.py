import numpy as np


class HighOrderEncoding:

    @staticmethod
    def build_mapper(order):
        base_pair = ['A', 'C', 'G', 'T']
        mapper = list([''])

        for _ in range(order):
            mapper_previous = mapper.copy()
            for nucleotide_pre in range(len(mapper_previous)):
                for nucleotide_now in base_pair:
                    mapper.append(mapper_previous[nucleotide_pre] + nucleotide_now)

            for _ in range(len(mapper_previous)):
                mapper.pop(0)

        one_hot = np.eye(len(mapper), dtype=int)
        high_order_code = dict()
        for i in range(len(mapper)):
            high_order_code[mapper[i]] = list(one_hot[i, :])

        return high_order_code

    @staticmethod
    def embedding(sequence, order, mapper):
        code = np.empty(shape=(len(sequence) - order + 1, 4 ** order))
        padding = np.zeros(shape=(4 ** order))

        for loc in range(len(sequence) - order + 1):
            code[loc] = mapper.get(sequence[loc: loc + order])

        lr_round_flag = 0
        for pad in range(order - 1):
            if lr_round_flag == 0:
                code = np.row_stack((padding, code))
                lr_round_flag = 1
            else:
                code = np.row_stack((code, padding))
                lr_round_flag = 0

        return code


def LabelEmbedding(original_label, cell_num=1):
    """
    (1) cell_num=1, original_label  0/1
    (2) cell_num>1, original_label  0, 1, 2, ..., cell_num -1
    """
    if cell_num == 1:
        return original_label
    else:
        new_label = np.zeros(shape=cell_num)
        new_label[original_label] = 1
        return new_label