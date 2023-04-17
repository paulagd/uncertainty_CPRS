import os, re
import argparse
import scipy
import torch
import numpy as np
import pickle5 as p
import pandas as pd

from scipy import optimize, stats, special
from tqdm import trange, tqdm
from collections import defaultdict
from scipy.special import softmax

from IPython import embed


def parse_args():
    parser = argparse.ArgumentParser(description='RECOMMENDER PARAMS')
    # common settings
    parser.add_argument('--seed', type=int, default=1234, help='pre-fixed seed for experiments')
    parser.add_argument('--model', type=str, default='mlp', help='model to select: [embc, mlp, random, itempop, context_itempop]')
    parser.add_argument('--dataset', type=str, default='non-logged', help='dataset to select')
    parser.add_argument('--aleatoric_ranker', action="store_true", default=False, help='apply aleatoric ranker')
    parser.add_argument('--dirichlet', action="store_true", default=False, help='apply epistemic ranker')
    parser.add_argument('--coin', type=int, default=1, help='repeat samples')

    parser.add_argument('--inference', action="store_true", default=False)
    parser.add_argument('--topk', type=int, default=5, help='top number of recommend list')
    parser.add_argument('--k_ranges', default=[5])
    parser.add_argument('--loss_type', type=str,  default='ce', help='loss function type: ce, mse')
    parser.add_argument('--save', action="store_true", default=False, help='activate to save weights')
    parser.add_argument('--T', type=float, default=1, help='softmax temperature')
    parser.add_argument('--n_train_weeks', type=int, default=12, help='number training weeks previous to testweek')
    parser.add_argument('--testweek', type=int, default=48, help='testweek from dataset')
    parser.add_argument('--noraw', action="store_true", default=False, help='activate data human postprocessed data')

    # algorithm settings
    parser.add_argument('--k', type=int, default=64, help='latent factors numbers in the model')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument("--do", default='[0.3, 0.2]', help="dropout rate for FM and MLP")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument("--batch_norm", default=True, help="use batch_norm or not")
    args = parser.parse_args()
    return args


def get_weights_path(args):
    nb = f'weights/seed_{args.seed}'
    testweek = f'testweek={args.testweek}'
    ruleta = '_ruleta' if args.aleatoric_ranker else ''
    dirichlet = '_dirichlet' if args.dirichlet else ''
    coin = '_coin' if args.coin else ''
    return f'{nb}/{args.model}/{testweek}{coin}{ruleta}{dirichlet}_do={eval(args.do)}_lr={args.lr}_bs={args.batch_size}'


def split_data_testweek(data, N_train_weeks=10, testweek=52):
    test_set = data[data[:, -1] == testweek][:, :-1]
    training_weeks = [i for i in range(testweek - N_train_weeks, testweek)]
    train_set = data[np.isin(data[:, -1], training_weeks)][:, :-1]
    return train_set, test_set


def get_data(args, filename='raw_split_data.npz'):
    if args.noraw:
        filename='noraw_split_data.npz'

    data_path = os.path.join('data', args.dataset, f'test_week={args.testweek}')
    if not os.path.exists(os.path.join(data_path, filename)):
        if args.noraw:
            df = pd.read_csv(f"data/{args.dataset}/post_processed_contextual_dataset_PSM.csv")
        else:
            df = pd.read_csv(f"data/{args.dataset}/contextual_dataset_PSM.csv")
        df = df[['item', 'device', 'daytime', 'weekday', 'sorted_week']]
        data = df.astype('int32').to_numpy()
        add_dims = 0
        for i in range(data.shape[1] - 1):  # do not affect to sorted_WEEK
            # MAKE IT START BY 0
            data[:, i] -= np.min(data[:, i])
            # RE-INDEX
            data[:, i] += add_dims
            add_dims = np.max(data[:, i]) + 1

        dims = np.max(data[:, :-1], axis=0) + 1
        train_x, test_x = split_data_testweek(data, N_train_weeks=args.n_train_weeks, testweek=args.testweek)
        os.makedirs(data_path, exist_ok=True)
        np.savez(os.path.join(data_path, filename), train_x=train_x, test_x=test_x,
                 dims=dims, data_path=data_path)
        return dims, train_x, test_x, data_path
    else:
        print('- data loaded!')
        data = np.load(os.path.join(data_path, filename))
        print('- split done!')
        # plot_merit([data['train_x'][:, 0], data['test_x'][:, 0]], data['dims'][0],
        #            name=f'data_merits_tstweek={args.testweek}', labels=['train', 'test'])
        return data['dims'], data['train_x'], data['test_x'], str(data['data_path'])


def GetItemsByFrequency(my_list):
    # returning items by popularity from + to - frequency appearence
    items, freq = np.unique(my_list, return_counts=True)
    zipped =  list(zip(items, freq))
    zipped_sorted = sorted(zipped, key = lambda x: x[1])

    return [k for k,v in zipped_sorted][::-1]


def get_ur(data):
    ur = defaultdict(set)
    for x in tqdm(data, desc=f"Building user-items dicctionary..."):
        item = x[0]
        ur[tuple(np.delete(x, 0, None))].add(int(item))

    return ur


def build_neg_set(test_gt, dims, context_inference=True):
    user_interactions = []
    for u in tqdm(test_gt.keys()):
        for gt_item in test_gt[u]:
            if context_inference:
                user_interactions.append(np.stack([gt_item] + list(u)))
            else:
                # TODO: build interaccions d'items que no hagin interaccionat sota x context
                # cand_items = [[item] + list(u) for item in range(dims[0]) if (not(item in seen_data[u])
                cand_items = [[item] + list(u) for item in range(dims[0]) if item !=gt_item]
                cand_items.insert(0, [gt_item] + list(u))
                user_interactions.append(np.stack(cand_items))
    if context_inference:
        return np.stack(user_interactions)
    else:
        return user_interactions


# We obtain the empirical parameters of the Dirichlet distribution
def get_Dirichlet_params_optimized(samples):
    nloglike = lambda a_0,a,s: - np.sum(special.gammaln(np.sum(a_0*a)) - np.sum(special.gammaln(a_0*a))
                                        + ((a_0*a).reshape(1,-1) * np.log(s)).sum(axis=1))

    return optimize.minimize(nloglike, 2., bounds=[[1e-6,1e20]],tol=1e-100,
                                   args=(samples.mean(axis=0),samples))['x'][0] * samples.mean(axis=0)

def get_Dirichlet_params(samples):
    nloglike = lambda a_0,a,s: - np.sum(scipy.special.gammaln(np.sum(a_0*a)) - np.sum(scipy.special.gammaln(a_0*a)) +
                                        ((a_0*a).reshape(1,-1) * np.log(s)).sum(axis=1))
    bb = scipy.optimize.minimize(nloglike, 2., bounds=[[1e-6,1e20]],
                                 tol=1e-100,args=(samples.mean(axis=0),samples))['x'][0] * samples.mean(axis=0)

    nloglike = lambda a_0,s: - np.sum(scipy.special.gammaln(np.sum(a_0)) - np.sum(scipy.special.gammaln(a_0)) +
                                      ((a_0).reshape(1,-1) * np.log(s)).sum(axis=1))
    return scipy.optimize.minimize(nloglike, bb,  bounds=[[1e-6, 1e20]]*bb.shape[0],tol=1e-1,args=(samples))['x']


def get_predictions_per_context_DL(alphas, freq_context, topk, ruleta=False):
    if freq_context == 1:

        scores = stats.dirichlet.rvs(alphas, size=freq_context)[0]
        return np.random.choice(list(range(len(scores))), topk, p=scores, replace=False)
    else:
        scores = stats.dirichlet.rvs(alphas, size=freq_context)
        return [np.random.choice(list(range(len(s))), topk, p=s, replace=False) for s in scores]


def getDLmodel(args, optimized=True):
    flag = 'old_' if optimized else ''
    if os.path.exists(f'data/{args.dataset}/test_week={args.testweek}/raw_ensambled_df_{args.model}.csv'):
        df = pd.read_csv(f'data/{args.dataset}/test_week={args.testweek}/raw_ensambled_df_{args.model}.csv')
    else:
        print(f'THERE IS NO ENSAMBLED DF SAVED FOR TESTWEEK={args.testweek} ...')
        exit()
    aux = pd.Categorical(df['context'].astype(str)).codes
    dict_context = {b: tuple(np.fromstring(a[1:][:-1], dtype=int, sep=' ')) for a, b in zip(df['context'].values, aux)}
    df['context'] = aux
    alphas_model = defaultdict(list)
    grouped = df.groupby('context')
    for name, group in tqdm(grouped):
        probs = softmax(group.iloc[:, 1:].values, axis=1)
        if optimized:
            alphas = get_Dirichlet_params_optimized(probs)
        else:
            alphas = get_Dirichlet_params(probs)
        alphas_model[dict_context[name]].append(alphas)

    apriori_probs = softmax(df.iloc[:, 1:].values, axis=1)
    if optimized:
        apriori_alphas = get_Dirichlet_params_optimized(apriori_probs)
    else:
        apriori_alphas = get_Dirichlet_params(apriori_probs)
    alphas_model['apriori'].append(apriori_alphas)
    # with open(f'data/{args.dataset}/test_week={args.testweek}/{flag}dirichlet_model_{args.model}.pkl', 'wb') as handle:
    #     p.dump(alphas_model, handle, protocol=p.HIGHEST_PROTOCOL)
    return alphas_model


def build_frequency_dict(items, num_items):
    frequency_dict = {}
    for item in items:
        if item in frequency_dict:
            frequency_dict[item] += 1
        else:
            frequency_dict[item] = 1
    for i in range(num_items):
        if i not in frequency_dict:
            frequency_dict[i] = 0
    total = sum(frequency_dict.values())
    norm =  {k: v / total for k, v in frequency_dict.items()}
    sorted_dict = {key: norm[key] for key in sorted(norm)}
    return sorted_dict


def gumbel_sampling(w, R, T):
    n = w.shape[1] # len(w)
    U = np.random.uniform(0,1,size=(R,n))
    G = w - np.log(- np.log(U))
    res = np.argsort(-G, axis=1)
    return res[:,:T]

