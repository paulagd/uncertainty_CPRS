import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle5 as pickle
import os
from IPython import embed
from collections import Counter
from utils import GetItemsByFrequency


class ItemPop(nn.Module):
    def __init__(self, train_data, noraw, topk, data_path='', contextbased=False):
        super(ItemPop, self).__init__()

        """
        Simple popularity based recommender system
        """
        # Sum the occurences of each item to get is popularity, convert to array and lose the extra dimension
        self.topk = topk
        dict_aux = {i:v for i, v in Counter(train_data.interactions[:, 0]).most_common()}
        popularity = []
        for item in range(train_data.dims[0]):
            if item in dict_aux.keys():
                popularity.append(dict_aux[item])
            else:
                popularity.append(0)
        self.columns_popularity = torch.Tensor(popularity)
        _, self.recommend = torch.topk(self.columns_popularity, topk)
        prepro = '_prepro' if noraw else ''

        if contextbased:
            print('CONTEXT BASED ITEM POP!')
            if not os.path.exists(os.path.join(data_path, f"sorted_items_per_context{prepro}.pkl")):
                from collections import defaultdict
                aux_dict = defaultdict(list)
                for row in train_data.interactions:
                    aux_dict[tuple(row[1:])].append(row[0])

                self.context_dict = {}
                for k, v in aux_dict.items():
                    self.context_dict[tuple(k)] = GetItemsByFrequency(v)
                # Store data (serialize)
                with open(os.path.join(data_path, f"sorted_items_per_context{prepro}.pkl"), 'wb') as handle:
                    pickle.dump(self.context_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                a_file = open(os.path.join(data_path, f"sorted_items_per_context{prepro}.pkl"), "rb")
                self.context_dict = pickle.load(a_file)
        else:
            self.context_dict = None
            
    def forward(self):
        pass

    def predict(self, interactions):
        if self.context_dict:
            itemsXcontext = []
            for interact in interactions[:, 1:].numpy():
                if tuple(interact) in self.context_dict:
                    itemsXcontext.append(self.context_dict[tuple(interact)])
                else:
                    itemsXcontext.append(list(self.recommend.numpy()))
            recommend = []
            for mostpop in itemsXcontext:
                if len(mostpop) < self.topk: 
                    mostpop.extend(list(self.recommend.numpy()))
                    recommend.append(list(dict.fromkeys(mostpop))[:self.topk])
                else:
                    recommend.append(mostpop[:self.topk])
            recommend = torch.Tensor(np.stack(recommend))
        else:
            recommend = torch.Tensor(np.repeat([self.recommend.numpy()], len(interactions), axis=0))
        return recommend, []


class RandomModel(nn.Module):
    def __init__(self, dims, topk):
        super(RandomModel, self).__init__()
        """
        Simple random based recommender system
        """
        self.all_items = list(range(dims[0]))
        self.topk = topk

    def forward(self):
        pass

    def predict(self, interactions):
        rand_list = [random.sample(self.all_items, self.topk) for i in range(len(interactions))]
        return torch.Tensor(np.stack(rand_list)) , []


class EmbeddingClassifier(nn.Module):
    def __init__(self, num_features, num_factors, t, sigmoid=False, s_param=1):
        super(EmbeddingClassifier, self).__init__()
        """
        Matrix factorization - extended to Tensor factorization 
        ===================================================
        num_features: number of input features,
        k: number of hidden factors,
        sigmoid: wether add a sigmoid activation at the end or not,
        """
        self.item_idx = torch.LongTensor(list(range(num_features[0]))).cuda()
        self.k = num_factors
        self.dims = num_features
        self.embeddings = nn.Embedding(num_features[0], self.k)
        self.embeddings_context = nn.Embedding(num_features[-1] - num_features[0], self.k)
        nn.init.normal_(self.embeddings_context.weight, std=0.01)

        self.sigmoid = sigmoid
        self.s_param = s_param
        self.act = nn.Sigmoid()
        self.T = t


    def forward(self, features):
        self.items_emb = self.embeddings(self.item_idx)
        embeddings_feat = self.embeddings_context(features[:, 1:]-self.dims[0])
        y_pred = torch.matmul(embeddings_feat, self.items_emb.T.unsqueeze(0)).prod(1)  
        return F.softmax(y_pred/self.T, dim=1), y_pred

    def predict(self, features):
        pred, logits = self.forward(features.cuda())
        return pred.cpu(), logits.cpu()


class MLP(nn.Module):
    def __init__(self, dims, num_factors, T=1):
        super(MLP, self).__init__()
        
        self.dims = dims
        self.k = num_factors
        self.num_features = (dims[-1]-dims[0])
        self.input_fc = nn.Linear(self.num_features, self.k)
        self.hidden_fc = nn.Linear(self.k, self.k*2)
        self.output_fc = nn.Linear(self.k*2, dims[0])

        self.T = T

    def forward(self, features, act=True):
        x = torch.nn.functional.one_hot(features[:, 1:] - self.dims[0], num_classes=self.num_features).sum(dim=1)
        h_1 = F.relu(self.input_fc(x.float()))
        h_2 = F.relu(self.hidden_fc(h_1))
        y_pred = self.output_fc(h_2)

        return F.softmax(y_pred/self.T, dim=1), y_pred

    def predict(self, features):
        pred, logits = self.forward(features.cuda())
        return pred.cpu(), logits.cpu()
