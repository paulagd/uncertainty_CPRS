import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import random


def plot_figure(xlabel, ylabel, title, x, namefig, y=None, legend=None, histogram=False, color='paleturquoise',
                nb=False, scatter=False, keep_plot=False):
    if not keep_plot:
        plt.clf()
        plt.rcParams.update({'font.size': 17})
        fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_title(title)
        plt.grid(False)
        ax.yaxis.grid(color='black', linestyle='--', linewidth=1, alpha=0.4)  # grid lines
        ax.xaxis.label.set_color('black')        # setting up X-axis label color to yellow
        # ax.yaxis.label.set_color('black')          # setting up Y-axis label color to blue
        ax.tick_params(axis='x', colors='black')    # setting up X-axis tick color to red
        ax.tick_params(axis='y', colors='black')  # setting up Y-axis tick color to black
        ax.spines['left'].set_color('black')        # setting up Y-axis tick color to red
        ax.spines['top'].set_color('black')         # setting up above X-axis tick color to red

    if histogram:
        x.plot(kind="hist", density=False, bins=25, color=color, alpha=1)
        plt.yscale('log', base=10)
        plt.ylabel(ylabel)
        plt.grid(True, which="major", axis='y', color='black', ls='--', linewidth=1, alpha=0.4)
    elif scatter:
        # ax.xaxis.grid(color='black', linestyle='--', linewidth=1, alpha=0.4)  # grid lines
        # plt.plot(x, x, ':r', alpha=0.3)  # dotted red
        plt.scatter(x, y, alpha=0.8, cmap='viridis', color=color)
        # plt.axis([0, 0.4, 0, 0.4])
        # plt.xticks(np.unique(x))
        plt.ylim([0, 1])

    else:
        plt.bar(x, y, color=color, alpha=1)
        plt.xticks(x, legend)
        plt.ylim([0, 60])

    plt.tight_layout()

    if not nb:
        plt.show()
        plt.savefig(namefig)


def plot_merit(data_list, nitems, name='data_merits', labels=[], column=0):
    plt.clf()
    fig = plt.figure(figsize=(10, 3), dpi=300)
    # X_axis = np.arange(nitems)
    # offset = [-0.2, 0.2]
    colors = ['darkviolet', 'olive', 'pink']

    ind = np.arange(nitems)
    width = 0.25
    for i, data in enumerate(data_list):
        df_initial = pd.DataFrame(data).reset_index()
        df = df_initial.groupby(column)['index'].count().reset_index()
        df.rename({0: 'item', 'index': 'freq'}, axis='columns', inplace=True)
        missing_items = list(set(range(nitems)) - set(np.unique(df_initial.groupby(column)['index'].count().index)))
        aux_df = pd.DataFrame(np.stack((missing_items, np.zeros(len(missing_items))), axis=1), columns=df.columns)
        full_df = pd.concat([df, aux_df]).set_index('item').sort_index()
        full_df.reset_index(drop=True, inplace=True)
        full_df = full_df/np.sum(full_df.freq)
        plt.bar(ind+width*i, list(full_df.freq.values), width, label=labels[i], color=colors[i], alpha=0.9)

    plt.xlim([0, nitems+1])
    fig.gca().set_xticks(list(range(0, len(full_df), 10)))
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{name}.pdf')


def plot_merit_pie(data, nitems, name='data_merits', model_name='', column=0, gt=False):

    df_initial = pd.DataFrame(data).reset_index()
    df = df_initial.groupby(column)['index'].count().reset_index()
    df.rename({0: 'item', 'index': 'freq'}, axis='columns', inplace=True)
    missing_items = list(set(range(nitems)) - set(np.unique(df_initial.groupby(column)['index'].count().index)))
    aux_df = pd.DataFrame(np.stack((missing_items, np.zeros(len(missing_items))), axis=1), columns=df.columns)
    full_df = pd.concat([df, aux_df]).set_index('item').sort_index()
    full_df.reset_index(drop=True, inplace=True)
    full_df = full_df/np.sum(full_df.freq)

    full_df = full_df[~full_df['freq'].isin([0.0])]


    # # plt.bar(ind+width*i, list(full_df.freq.values), width, label=labels[i], color=colors[i], alpha=0.9)
    # fig.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
    #         shadow=True, startangle=90)
    # fig.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # colors = list(mcolors.XKCD_COLORS.values())
    colors = random.choices(list(mcolors.XKCD_COLORS.values()), k=len(full_df))
    plt.clf()
    plt.figure(figsize=(10, 3), dpi=300)
    if gt:
        full_df.plot.pie(y='freq', figsize=(10, 10), title=f'MERIT_{model_name}', colors=colors)
        plt.legend(loc=0, title='Nº ITEM')
    else:
        full_df.plot.pie(y='freq', figsize=(10, 10), labels=['' for i in range(len(full_df))], autopct='%1.1f%%',
                         title=f'MERIT_{model_name}', colors=colors)

        plt.legend(loc=1, labels=full_df.index, title='Nº ITEM')
    # plt.tight_layout()
    plt.savefig(f'{name}.png')


def plot_ponderated_HR(tst_interactions, model, pond):
    c, idx, f = np.unique(tst_interactions[:, 1:], axis=0, return_index=True, return_counts=True)
    f = f/f.sum()
    x = list(range(len(c)))
    ponderatedHR = []
    for c_i in c:
        ponderatedHR.append(np.mean(pond[tuple(c_i)]))

    plt.clf()
    fig, ax1 = plt.subplots(figsize=(14,6))
    color = 'tab:blue'
    ax1.set_xlabel('contexts')
    ax1.set_ylabel('context_freq', color = color)
    ax1.bar([i - 0.3 for i in x], f, color=color, width=0.3)
    ax1.tick_params(axis ='y', labelcolor = color)

    # Adding Twin Axes to plot using dataset_2
    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('HR per context', color=color)
    ax2.bar(x, ponderatedHR, color=color, width=0.3)
    ax2.tick_params(axis ='y', labelcolor = color)

    plt.tight_layout()
    plt.savefig(f'ponderated_HR_{model}.jpg')


def plot_bars(dict1, dict2, dict3, bar_width=0.1, ylim=0.2, xlim=525):
    plt.clf()
    #fig = plt.figure(figsize=(15, 5))
    #ax = fig.add_subplot(1, 1, 1)

    # Get the keys and values from the dictionaries
    # keys1 = list(dict1.keys())
    values1 = list(dict1.values())
    if dict2 is not None:
        values2 = list(dict2.values())
    values3 = list(dict3.values())
    # Set the position of the bars
    r1 = [i for i in range(len(dict1))]
    if dict2 is not None:
        r2 = [i + 2*bar_width for i in range(len(dict2))]
    r3 = [i + bar_width for i in range(len(dict3))]

    # Create the bars
    plt.bar(r1, values1, width=bar_width, label='train freq')
    if dict2 is not None:
        plt.bar(r2, values2, width=bar_width, label='rec freq')    # bottom=values1
    plt.bar(r3, values3, width=bar_width, label='test freq')    # bottom=values1

    # Add labels and a title
    plt.xlabel('Item')
    plt.ylabel('Frequency')
    # Set the x-axis limits to [0, 4]
    plt.xlim([-bar_width*3, xlim])

    # Set the y-axis limits to [0, 25]
    plt.ylim([0, ylim])
    # Add a legend
    plt.legend()
    plt.savefig('aux.pdf')
