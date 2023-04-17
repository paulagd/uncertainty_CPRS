import torch
import os
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm
# from plots import plot_bars
from utils import get_predictions_per_context_DL, gumbel_sampling
from metrics import hit, ndcg, coverage_at_k, ISP, personalized_cov, get_fairness_metric, get_meritdict_per_context
from IPython import embed


def train_run(model, loader, device, model_name, criterion=None, optimizer=None):
    """
    Function to log the training loss
    """
    model.train()
    av_loss = []

    for iter, (x, y) in enumerate(loader):

        x, y = x.to(device), y.to(device)
        model.zero_grad()
        pred, logits = model(x.long())
        loss = criterion(pred, y.float())
        
        av_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(av_loss)


def inference_run(epoch, model, test_loader, args, dims, train_loader):
    """
    Function to log the validation loss
    """
    if not args.dirichlet:
        model.eval()
    recommended_items, recommend_user, all_logits = [], [], []
    gt_items, cand_per_user = [], []
    preds, tst_interactions = [], []
    mockedmodel = True if args.model == 'random' or args.model == 'itempop' or args.model == 'context_itempop' else False

    with torch.no_grad():
        for iter, interactions in enumerate(tqdm(test_loader, desc='inference for each user...')):
            gt_items.append(interactions[:, 0].numpy())
            if args.dirichlet:
                # We obtain 'size' samples of distributions
                rankings = []
                for a in interactions[:, 1:]:
                    dl_context = model['apriori'][0] if len(model[tuple(a.cpu().numpy())]) == 0 \
                        else model[tuple(a.cpu().numpy())][0]
                    rankings.append(get_predictions_per_context_DL(dl_context, args.coin, args.topk, ruleta=args.aleatoric_ranker))
                # rankings = [get_predictions_per_context_DL(model[tuple(a.cpu().numpy())][0], args.coin,
                #                                            args.topk, ruleta=args.aleatoric_ranker) for a in interactions[:, 1:]]
                recommended_items.append(np.vstack(rankings))
            else:
                prediction, logits = model.predict(interactions)
                all_logits.append(logits)
                if mockedmodel:
                    recommended_items.append(prediction.numpy())
                else:
                    if args.aleatoric_ranker:
                        # aux = [np.random.choice(name_items, args.topk, p=list(p_i/p_i.sum()), replace=False) for p_i
                        #        in prediction.numpy().astype('float64')]
                        # recommended_items.append(np.stack(aux))
                        recommended_items.append(gumbel_sampling(logits, len(logits), args.topk))
                    else:
                        _, indices = torch.topk(logits, args.topk)
                        recommended_items.append(indices.numpy())
                    preds.append(prediction)
            tst_interactions.append(interactions.numpy())

    # IDEA: for DL preparation
    if not args.dirichlet and not mockedmodel:
        all_logits = np.vstack(all_logits) if not mockedmodel else []
        contexts, idx, freq = np.unique(np.vstack(tst_interactions)[:, 1:], axis=0, return_index=True, return_counts=True)
        mu_pred = np.asarray(all_logits[idx])
        name_mu = [f'mu_{item}' for item in range(mu_pred.shape[1])]
        mu_df = pd.DataFrame(all_logits[idx], columns=name_mu)
        df_seed = pd.concat([pd.DataFrame({'context': list(contexts)}), mu_df], axis=1)

        tr_contexts, _, tr_freq = np.unique(train_loader.dataset.interactions[:, 1:], axis=0, return_index=True, return_counts=True)
        total_df = None
    else:
        df_seed, total_df = None, None

    tst_interactions = np.vstack(tst_interactions)
    ponderated_fairnes_metric = get_fairness_metric(tst_interactions, train_loader.dataset.interactions,
                                                    np.vstack(recommended_items), args=args, entire=True)
    ####################
    # IDEA: plot EoE
    # len_fairnes_metric = len(ponderated_fairnes_metric)
    # x = list(range(1, len_fairnes_metric+1))
    # unique = [len(v) for k, v in info_list.items()]

    # import matplotlib.pyplot as plt

    # plt.clf()
    # fig, ax1 = plt.subplots(figsize=(14,6))
    # color = 'tab:blue'
    # ax1.set_xlabel('context')
    # ax1.set_ylabel('unfairness', color = color)
    # ax1.bar([i - 0.3 for i in x], fairnes_metric, color=color, width=0.3)
    # ax1.tick_params(axis ='y', labelcolor = color)
    #
    # # Adding Twin Axes to plot using dataset_2
    # ax2 = ax1.twinx()
    # color = 'tab:red'
    # ax2.set_ylabel('unique items inside context', color=color)
    # ax2.bar(x, unique, color=color, width=0.3)
    # ax2.tick_params(axis ='y', labelcolor = color)
    #
    # plt.tight_layout()
    # plt.savefig(f'unique_context_{args.model}.jpg')

    ####################
    # IDEA: CHECK WHICH ITEMS ARE RECOMMENDING
    metrics_at_k = []
    gt_items = np.concatenate(gt_items)
    recommended_items = np.concatenate(recommended_items)
    from collections import defaultdict
    pond = defaultdict(list)

    for k in args.k_ranges:
        HR, NDCG, pred_k_items = [], [], []
        for idx, (gt, recommends, c) in tqdm(enumerate(zip(gt_items, recommended_items, tst_interactions[:, 1:])), desc='computing HR, NDCG...'):
            pond[tuple(c)].append(hit(gt, recommends[:k]))
            HR.append(hit(gt, recommends[:k]))
            NDCG.append(ndcg(gt, recommends[:k]))
            pred_k_items.append(recommends[:k])
        cov, top_items = coverage_at_k(pred_k_items, dims[0] , k)
        custom_cov = personalized_cov(HR, gt_items, dims[0])

        HR = np.mean(HR)
        NDCG = np.mean(NDCG)
        metrics_at_k.append([HR, NDCG, ponderated_fairnes_metric, cov, custom_cov])

        if k == 5:
            print(f"[epoch {epoch + 1}] HR@{k}:{np.mean(HR):.4f} | NDCG@{k}: {np.mean(NDCG):.4f} |"
                  f" fairness@{k}: {ponderated_fairnes_metric:.4f} | Coverage@{k}: {cov:.4f} | custom_cov@{k}: {custom_cov:.4f}")

    return metrics_at_k, recommended_items, df_seed, total_df


def evaluation(model, weights_path, args, test_loader, dims, train_loader, load=True):
    if os.path.exists(os.path.join(os.path.dirname(weights_path), f'weights.pkl')) and load and not args.dirichlet:
        checkpoint = torch.load(os.path.join(os.path.dirname(weights_path), f'weights.pkl'))
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Weights loaded FROM {os.path.join(os.path.dirname(weights_path), f'weights.pkl')}!")
    else:
        print(f'NO weights loaded FROM {weights_path}, using {args.model.upper()} RND model for inference!')

    metrics, recommended_items, df_seed, total_df = inference_run(-1, model, test_loader, args, dims, train_loader)
    print(f'-------------- INFERENCE METRICS ------------')
    for k in range(len(args.k_ranges)):
        print(f"HR@{args.k_ranges[k]}:{metrics[k][0]:.4f} | NDCG@{args.k_ranges[k]}: {metrics[k][1]:.4f} "
              f"| fairness@{args.k_ranges[k]}: {metrics[k][2]:.2f} |  coverage@{args.k_ranges[k]}: {metrics[k][3]:.4f}"
              f" | custom_cov@{k}: {metrics[k][4]:.4f}")
    return metrics, recommended_items, df_seed, total_df