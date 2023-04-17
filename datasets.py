import os
import pickle
import numpy as np
import torch.utils.data as data
import pickle5 as pickle
from IPython import embed

from utils import get_ur, build_neg_set


class PointData(data.Dataset):
    def __init__(self, data, dims, data_path=None, file2save='raw_sorted_items_per_context'):
        """
        Dataset formatter adapted point-wise algorithms
        """
        super(PointData, self).__init__()

        self.interactions = data
        self.dims = dims

        if not os.path.exists(os.path.join(data_path, "f{file2save}.pkl")):
            from collections import Counter
            dict_2_save = {}
            (triplets, freq) = np.unique(data[:, 1:], return_index=False, return_inverse=False, return_counts=True, axis=0)
            for tri in triplets:
                ids = np.where((data[:, 1:] == tri).all(axis=1))
                c = Counter(data[:, 0][ids])
                sorted_items = [key for key, val in c.most_common()]
                dict_2_save[tuple(tri)] = sorted_items

            a_file = open(os.path.join(data_path, f"{file2save}.pkl"), "wb")
            pickle.dump(dict_2_save, a_file)
            a_file.close()

        self.empty_label = np.asarray([0] * (self.dims[0]))

    def __onehot__(self, item_interaction):
        aux = self.empty_label.copy()
        aux[item_interaction] = 1
        return aux

    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, index):
        return self.interactions[index],  self.__onehot__(self.interactions[index][0])#,  self.c_freq_sample[tuple(self.interactions[index][1:])]


class TestBuilderSet(object):
    def __init__(self, test_x, dims, cand_num=0, data_path='', file2save='raw_test_user_interactions_dims'):
        """
        negative sampling class for some algorithms
        """
        aux_rank = '' if cand_num == 0 else f'cand_num={cand_num}'
        self.dims = dims

        if os.path.exists(os.path.join(data_path, f'{file2save}={dims[-1]}{aux_rank}.pkl')):
            with open(os.path.join(data_path, f'{file2save}={dims[-1]}{aux_rank}.pkl'), "rb") as fp:
                self.user_interactions = pickle.load(fp)
            print('- TEST interactions loaded!')
        else:

            data_gt = get_ur(test_x)
            self.user_interactions = build_neg_set(data_gt, self.dims, context_inference=True)
            with open(os.path.join(data_path, f'{file2save}={dims[-1]}{aux_rank}.pkl'), "wb") as fp:
                pickle.dump(self.user_interactions, fp)
        
    def __getitem__(self, index):
        return self.user_interactions[index]

    def __len__(self):
        return len(self.user_interactions)
