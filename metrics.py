import numpy as np
import os, pickle
from collections import defaultdict
from tqdm import trange
import pandas as pd
import pickle
from IPython import embed


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0

def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = np.where(gt_item == pred_items)[0][0]
        return np.reciprocal(np.log2(index+2))
    return 0


def get_meritdict_per_context(itemslist):
    items, freq = np.unique(itemslist, return_index=False, return_counts=True)
    item_merits = dict(zip(items, freq))
    return {k:v/sum(item_merits.values()) for k,v in item_merits.items()}


def get_fairness_metric(tst_interactions, train_interactions, recommended_items, entire=False, args='',
                        plot_merits=False):
    if not entire:  # plot per each context, not for the entire dataset
        tr_items = train_interactions[:, 0]
        rec_items = np.concatenate(recommended_items)
        M = get_meritdict_per_context(tr_items)
        P = get_meritdict_per_context(rec_items)
        if plot_merits:
            ru = '_ruleta' if args.aleatoric_ranker else ''
            gce = '_gce' if args.gce else ''
            with open(f'M_{args.model}{ru}{gce}.pkl', 'wb') as f:
                pickle.dump(M, f)
            with open(f'P_{args.model}{ru}{gce}.pkl', 'wb') as f:
                pickle.dump(P, f)
            exit()
        a = pd.DataFrame([M,P]).T.reset_index()
        a.rename(columns={"index": "items", 0: "M", 1:"P"}, inplace=True)
        pond_foe = np.abs(a.P.fillna(0) - a.M.fillna(0))
    else:
        train_merits = defaultdict(list)
        for row in train_interactions:
            train_merits[tuple(row[1:])].append(row[0])

        test_recs = defaultdict(list)
        for i, row in enumerate(tst_interactions):
            test_recs[tuple(row[1:])].append(recommended_items.astype(int)[i])

        pond_foe = []
        contexts, freq_context = np.unique(tst_interactions[:, 1:], axis=0, return_index=False, return_counts=True)
        pond_contexts = {tuple(c):f/sum(freq_context) for c, f in zip(contexts, freq_context)}
        for c, rec_list in test_recs.items():
            M = get_meritdict_per_context(train_merits[c])
            P = get_meritdict_per_context(np.concatenate(rec_list))
            if plot_merits:
                with open('M0.pkl', 'wb') as f:
                    pickle.dump(M, f)
                with open('P0.pkl', 'wb') as f:
                    pickle.dump(P, f)
                exit()
            a = pd.DataFrame([M,P]).T.reset_index()
            a.rename(columns={"index": "items", 0: "M", 1:"P"}, inplace=True)
            foe = np.sum(np.abs(a.P.fillna(0) - a.M.fillna(0)))
            pond_foe.append(foe*pond_contexts[c])
    return np.sum(pond_foe)
    

def personalized_cov(HR, gt_items, n_items):
    ok_gt_items = [gt for hr, gt in zip(HR, gt_items) if hr > 0]
    return len(np.unique(ok_gt_items)) / n_items


def coverage_at_k(pred_items, n_items, k=None):
    """
    Coverage --> analyses the diversity among recommended items
    """
    if k:
        top_items = np.concatenate([items[:k] for items in pred_items])
    else:
        top_items = np.concatenate(pred_items)
    return len(np.unique(top_items)) / n_items, top_items


def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))


def ISP(cand, rec, dims, k=10):

    prob = get_prob_i_k(cand, rec, dims, k)
    return prob, 1-gini_coefficient(prob)


def get_hr_givenusers(users, gt_items, rec_items, k):
    gt = np.asarray(gt_items)[users]
    rec = np.asarray(rec_items)[users]
    HR = []
    for idx, (g, recommends) in enumerate(zip(gt, rec)):
        HR.append(hit(g, recommends[:k]))
    return np.mean(HR)


def get_HR_users_error(args, gt_items, recommended_items, k):
    a_file = open(os.path.join('data', args.dataset, "user_av_bins.pkl"), "rb")
    user_pop_bins = pickle.load(a_file)

    fashion = get_hr_givenusers([key for key, val in user_pop_bins.items() if val == 1], gt_items, recommended_items, k)
    middle = get_hr_givenusers([key for key, val in user_pop_bins.items() if val == 2], gt_items, recommended_items, k)
    diverse = get_hr_givenusers([key for key, val in user_pop_bins.items() if val == 3], gt_items, recommended_items, k)

    return fashion, middle, diverse


def find_gt_useridx(item, gt):
    return [i for i, g in enumerate(gt) if g == item]


def get_prob_i_k(candidates, recommended, dims, k):
    isp = [np.sum([1 if item in rec[:k] else 0 for rec in recommended])/
           np.sum([1 if item in cand else 0.000001 for cand in candidates])
           for item in trange(dims[0], dims[1])]
    return isp
