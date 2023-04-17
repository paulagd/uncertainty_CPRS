# Import common packages
import numpy as np
import os, time, pickle
import pandas as pd
from tqdm import tqdm
from IPython import embed
from datetime import datetime

# Import Pytorch functions
import torch
import joblib
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.sparse import identity
#from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.special import softmax


# Import Utility Functions
from utils import *
from plots import *
from models import ItemPop, RandomModel, EmbeddingClassifier, MLP
from inference import evaluation, inference_run, train_run
from datasets import PointData, TestBuilderSet


def main_loop(seed=1234, testweek=48):

    # Selecting arguments
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Fixing seeds
    args.testweek = testweek
    args.seed = seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ################################################################################
    # Selecting data
    ################################################################################
    dims, train_data, test_data, data_path = get_data(args)
    print(f'DIMS = {dims}')
    rawflag = 'no' if args.noraw else ''
    train_dataset = PointData(train_data, dims, data_path=data_path, file2save=f'{rawflag}raw_sorted_items_per_context')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_set = TestBuilderSet(test_data, dims, data_path=data_path, file2save=f'{rawflag}raw_test_user_interactions_dims')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    ################################################################################
    # Defining model
    ################################################################################
    if args.dirichlet:
        model = getDLmodel(args)
        args.inference=True
    elif args.model == 'embc':
        model = EmbeddingClassifier(dims, args.k, args.T).to(device)
    elif args.model == 'mlp':
        model = MLP(dims, args.k, args.T).to(device)
    elif args.model == 'itempop' or args.model == 'context_itempop':
        args.inference = True
        model = ItemPop(train_dataset, args.noraw, args.topk, data_path, contextbased=args.model.split('_')[0] == 'context').to(device)
    elif args.model == 'random':
        args.inference = True
        model = RandomModel(dims, args.topk).to(device)

    # Defining weights path either to load or to save the model
    weights_path = get_weights_path(args)

    ################################################################################
    # Defining inference or training
    ################################################################################
    s_time = time.time()
    if args.inference:
        metrics, recommended_items, df_seed, total_df = evaluation(model, weights_path, args, test_loader, dims, train_loader)
        # plot_merit([train_data, np.concatenate(recommended_items)], 0, dims[0], name=f'data_merits_{args.model}',
        #            labels=['train', str(args.model)])
        # ru = '_ruleta' if args.aleatoric_ranker else ''
        # plot_merit_pie(np.concatenate(recommended_items), dims[0], name=f'pie_data_merits_{args.model}{ru}',
        #                model_name=f'{args.model}')
        # plot_merit_pie(train_data[:, 0], dims[0], name=f'pie_data_merits_GT', model_name=f'GT', gt=True)
        # plot_merit_pie(test_data[:, 0], dims[0], name=f'pie_data_merits_GT_test', model_name=f'GT_test', gt=True)

    else:
        # Defining optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

        if args.loss_type == 'mse':
            criterion = nn.MSELoss(reduction='mean')
        elif args.loss_type == 'ce':  # log_loss
            # criterion = nn.BCEWithLogitsLoss(reduction='none' if args.prop_score else 'sum')
            criterion = nn.BCELoss(reduction='sum')

        # set k_eval to 10 in order to measure metrics@10 during training
        early_stopping_counter = 0
        previous_loss = 9999999

        for epoch in range(args.epochs):
            loss = train_run(model, train_loader, device, args.model, criterion, optimizer)
            print(f"Epoch[{epoch}] Loss: {loss}")

            if int(round(loss)) < int(round(previous_loss)):
                previous_loss = loss
                best_epoch = epoch
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter == 5:
                    print('Satisfy early stop mechanism')
                    break
            metrics, recommended_items, df_seed, total_df = inference_run(epoch, model, test_loader, args, dims, train_loader)

        print(f'--------------BEST METRICS on epoch {best_epoch} ------------')
        if args.save:
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            best_checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(best_checkpoint, os.path.join(os.path.dirname(weights_path), f'weights.pkl'))
        args.inference = True
        evaluation(model, weights_path, args, test_loader, dims, train_loader, load=False)

    ################################################################################
    # Compute the elapsed training and inference time
    ################################################################################

    elapsed_time = time.time() - s_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'TOTAL ELAPSED TIME: {hours:.2f} hours, {minutes:.2f} min, {seconds:.4f}seconds')
    return metrics, df_seed, total_df, args


def get_all_logits(model='mlp', dataset='non-logged', percontext=False, top=False):
    if percontext:
        dfs = []
        train_data = []
        for testweek in list(range(12, 52)):
            # with open(f"data/non-logged/test_week={testweek}/raw_sorted_items_per_context.pkl", 'rb') as filehandle:
            #     a = pickle.load(filehandle)
            data = np.load(os.path.join(f'data/{dataset}/test_week={testweek}', f'raw_split_data.npz'))
            train_data.append(data['train_x'])
        #     if testweek == 12:
        #         df_week_context = pd.DataFrame(columns=[f'test_week={testweek}'],
        #                           index=pd.MultiIndex.from_tuples(a.keys()))
        #         df_week_context[f'test_week={testweek}'] = [len(i) for i in a.values()]
        #     else:
        #         df_week_context[f'test_week={testweek}'] = df_week_context.index.map({k:len(v) for k,v in a.items()})
        #
            dfs.append(pd.read_csv(f'data/{dataset}/test_week={testweek}/raw_ensambled_df_{model}.csv'))
        df = pd.concat(dfs)

        # df.to_csv(f'data/{dataset}/ensambled_df_{model}.csv', index=None)
        # # df_week_context.to_csv(f'data/{dataset}/df_week_context_{model}.csv')
        #
        # with open(f"data/non-logged/total_train_data.pkl", 'wb') as filehandle:
        #     pickle.dump(train_data, filehandle)

        aux = pd.Categorical(df['context'].astype(str)).codes
        dict_context = {b: tuple(np.fromstring(a[1:][:-1], dtype=int, sep=' ')) for a, b in zip(df['context'].values, aux)}
        df['context'] = aux
        incertesa_per_context = {}#defaultdict(list)
        grouped = df.groupby('context')
        for name, group in grouped:
            probs = softmax(group.iloc[:, 1:].values, axis=1)

            # A function to be applied to the array
            def get_entropy(row):
                sorted = np.sort(row)[::-1]
                # return np.abs(np.sum(sorted[0])  - np.sum(sorted[1]))
                # return np.sum(sorted[:10])  - np.sum(sorted[10:])
                return -np.sum(sorted*np.log(sorted))
            # Apply add() function to array.
            aux = [get_entropy(row) for row in probs]
            incertesa_per_context[tuple(dict_context[name])] = np.sum(aux)
            
            # incertesa_per_context[tuple(dict_context[name])] = np.sum(np.sort(probs.mean(axis=0))[::-1][:5]) \
            #                                                    - np.sum(np.sort(probs.mean(axis=0))[::-1][5:])

            # incertesa_per_context[tuple(dict_context[name])] = abs(np.sort(probs.mean(axis=0))[::-1][0] -
            #                                                        np.sort(probs.mean(axis=0))[::-1][1])

        with open(f"data/non-logged/one_sum_results_incertesa_context_{model}.pkl", 'wb') as filehandle:
            pickle.dump(incertesa_per_context, filehandle)

    else:
        incertesa = defaultdict(list)
        def get_uncertanty(row, i, top):
            if top:
                return np.sum(np.sort(row)[::-1][i]) - np.sum(np.sort(row)[::-1][i+1])
            else:
                return np.sum(np.sort(row)[::-1][:i]) - np.sum(np.sort(row)[::-1][i:])
        
        for testweek in list(range(12, 52)):
            df = pd.read_csv(f'data/{dataset}/test_week={testweek}/ensambled_df_{model}.csv')
            probs = softmax(df.iloc[:, 1:].values, axis=1)
            # incertesa.append(np.sum(np.sort(probs.mean(axis=0))[::-1][:5]) - np.sum(np.sort(probs.mean(axis=0))[::-1][5:]))

            for i in range(1, 6):
                incertesa[i].append(np.sum([get_uncertanty(row, i, top) for row in probs]))
                # if top:
                #     embed()
                #     incertesa[i].append(np.sum(np.sort(probs.mean(axis=0))[::-1][i]) -
                #                         np.sum(np.sort(probs.mean(axis=0))[::-1][i+1]))
                # else:
                #     incertesa[i].append(np.sum(np.sort(probs.mean(axis=0))[::-1][:i]) -
                #                         np.sum(np.sort(probs.mean(axis=0))[::-1][i:]))


            # incertesa.append(abs(df.mean().sort_values()[0] -  df.mean().sort_values()[1]))
            # df.meane().sort_values()[df.mean().sort_values() >df.mean().sort_values().mean()].shape
        t = 'top_' if top else ''
        with open(f"data/non-logged/{t}results_incertesa_{model}.pkl", 'wb') as filehandle:
            pickle.dump(incertesa, filehandle)
    exit()


if __name__ == '__main__':

    # IDEA: get_all_logits(percontext=False, top=False)
    slided_results, total_dfs = [], []
    for testweek in list(range(12, 52)):  #IDEA: cemb with gcn badd in week [ 13, 35, 36, 37, 39, 40, 42, 43]  41
        print('-------------------------------------')
        print(f'week {testweek}')
        print('-------------------------------------')
        results, dfs = [], []
        for seed in [1234, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]:
            print(f'Seed: {seed}')
            mlp_output, df_seed, total_df, args = main_loop(seed, testweek)
            results.append(mlp_output)
            dfs.append(df_seed)
            if total_df is not None:
                total_dfs.append(total_df)
        results = np.vstack(results)
        if df_seed is not None:
            df = pd.concat(dfs)
            df.to_csv(f'data/{args.dataset}/test_week={args.testweek}/raw_ensambled_df_{args.model}.csv', index=False)
        else:
            print('DIRICHLET!!')
        slided_results.append(results)
    # aux = pd.concat(total_dfs)
    # #
    # # aux = [df.set_index('context') for df in total_dfs]
    # # total_df = pd.concat(aux, axis=1)
    # aux.to_csv(f'data/{args.dataset}/raw_total_df_{args.model}.csv')

    # results = np.vstack(slided_results)
    results = [np.mean(a, axis=0) for a in slided_results]

    print('AVERAGE SEEDS [1234, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]:')
    print('')
    print(f'HR --> {np.round(np.mean(results, axis=0)[0], 3)} +- {np.round(np.std(results, axis=0)[0], 3)}')
    print(f'NDCG --> {np.round(np.mean(results, axis=0)[1], 3)} +- {np.round(np.std(results, axis=0)[1], 3)}')
    print(f'FAIRNESS --> {np.round(np.mean(results, axis=0)[2], 3)} +- {np.round(np.std(results, axis=0)[2], 3)}')
    print(f'COV --> {np.round(np.mean(results, axis=0)[3], 3)} +- {np.round(np.std(results, axis=0)[3], 3)}')
    print(f'GT_COV --> {np.round(np.mean(results, axis=0)[4], 3)} +- {np.round(np.std(results, axis=0)[4], 3)}')

    # ru = '_ruleta' if args.aleatoric_ranker else ''
    # gce = '_gce' if args.gce else ''
    # DL = '_DL' if args.dirichlet else ''
    # with open(f"data/non-logged/results_{args.model}{DL}{gce}{ru}.pkl", 'wb') as filehandle:
    #     pickle.dump(slided_results, filehandle)
