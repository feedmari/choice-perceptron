#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from scipy.io import loadmat
from glob import glob
import choice


def pad(array, length):
    """Pad vector with zeros up to given length."""
    assert array.ndim == 1
    full = np.zeros(length, dtype=array.dtype)
    full[:len(array)] = array
    return full

def zafz(loss, time):
    found = False
    for i in range(len(loss)):
        if found:
            loss[i] = 0.0
            time[i] = 0.0
        if loss[i] <= 0:
            found = True
    return loss, time

def zafz_matrix(lm, tm):
    for i in range(len(lm)):
        lm[i], tm[i] = zafz(lm[i], tm[i])
    return lm, tm

def get_style(info):
    method = info['method']

    if method == 'choice':
        k = info['set_size']
        color = {
            2: '#DC322F',
            3: '#D33682',
            4: '#6C71C4',
        }[k]

        label = 'CP-{k}'.format(**locals())
        marker = 'o'

    elif method == 'setmargin':
        k = info['k']
        color = {
            2: '#47e18a',
            3: '#2AA198',
            4: '#268BD2',
        }[k]

        label = 'SM-{k}'.format(**locals())
        marker = 's'

    elif method == 'viappiani':
        qs = info['qss']
        color = {
            'EUS': '#e1f500',
            'QI': '#f78c3d'
        }[qs]

        label = 'VB-{}'.format(qs)
        marker = {'EUS': '<', 'QI': '>'}[qs]

    else:
        raise NotImplementedError()

    return label, marker, color

def load(path, whichregret):
    regret_index = {'min': 0, 'avg': 1, 'max': 2}[whichregret]
    basedir, extension = os.path.splitext(path)
    basename = os.path.basename(path)
    if extension == '.pickle':
        try:
            d = choice.load(path) # choice
            method = 'choice'
        except UnicodeDecodeError:
            d = choice.load(path, encoding='bytes') # setmargin
            method = 'setmargin'
        if method == 'choice':
            max_iters = d['args']['max_iters']
            loss_matrix, time_matrix = [], []
            for trace in d['traces']:
                trace = np.array(trace)
                loss_matrix.append(pad(trace[:,regret_index], max_iters))
                time_matrix.append(pad(trace[:,-1], max_iters))
            info = {**{'method': 'choice'}, **d['args']}
        else:
            args = basename.split('__')
            k = int(args[1].split('=')[-1])
            max_iters = int(args[11])
            loss_matrix, time_matrix = [], []
            for trace in d:
                trace = np.array(trace)
                loss_matrix.append(pad(trace[:,1], max_iters))
                time_matrix.append(pad(trace[:,-1], max_iters))
            info = {'method': 'setmargin', 'max_iters': max_iters, 'k': k}
    elif extension == '.txt': # viappiani & boutilier 2010
        args = basename.split('_')
        qss = args[3]
        d = pd.read_csv(path).values
        loss_matrix = d[:,2].reshape((20, 101))[:,:100]
        time_matrix = d[:,3].reshape((20, 101))[:,:100]
        loss_matrix, time_matrix = zafz_matrix(loss_matrix, time_matrix)
        info = {'method': 'viappiani', 'max_iters': 100, 'qss': qss}
    else:
        raise NotImplementedError()
    return np.array(loss_matrix), np.array(time_matrix), info

def draw(args):

    plt.style.use('ggplot')

    data = []
    for path in args.pickles:
        data.append(load(path, args.regret))

    loss_fig, loss_ax = plt.subplots(1, 1)
    time_fig, time_ax = plt.subplots(1, 1)

    max_regret, max_time, max_iters, = -np.inf, -np.inf, -np.inf
    for loss_matrix, time_matrix, info in data:
        label, marker, color = get_style(info)

        max_iters = args.max_iters or max(max_iters, info['max_iters'])
        xs = np.arange(1, (args.max_iters or info['max_iters']) + 1)

        if args.avg_cum_regret:
            cumloss_matrix = (loss_matrix.cumsum(axis=1) /
                              (np.arange(loss_matrix.shape[1]) + 1))
            ys = np.median(cumloss_matrix, axis=0)
            yerrs = np.std(cumloss_matrix, axis=0) / np.sqrt(loss_matrix.shape[0])
        else:
            ys = np.median(loss_matrix, axis=0)
            yerrs = np.std(loss_matrix, axis=0) / np.sqrt(loss_matrix.shape[0])
        ys, yerrs = ys[:max_iters], yerrs[:max_iters]
        max_regret = args.max_regret or max(max_regret, ys.max())

        loss_ax.plot(xs, ys, linewidth=2, label=label, marker=marker,
                     markersize=6, color=color)
        loss_ax.fill_between(xs, ys - yerrs, ys + yerrs, color=color,
                             alpha=0.35, linewidth=0)

        cumtime_matrix = time_matrix.cumsum(axis=1)
        ys = np.mean(cumtime_matrix, axis=0)
        yerrs = np.std(cumtime_matrix, axis=0)
        ys, yerrs = ys[:max_iters], yerrs[:max_iters]
        max_time = args.max_time or max(max_time, ys.max())

        time_ax.plot(xs, ys, linewidth=2, label=label, marker=marker,
                     markersize=6, color=color)
        time_ax.fill_between(xs, ys - yerrs, ys + yerrs, color=color,
                             alpha=0.35, linewidth=0)

    def prettify(ax, max_iters):
        xtick = 5 if max_iters <= 50 else 10
        xticks = np.hstack([[1], np.arange(xtick, max_iters + 1, xtick)])
        loss_ax.set_xticks(xticks)

        ax.xaxis.label.set_fontsize(18)
        ax.yaxis.label.set_fontsize(18)
        ax.grid(True)
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            line.set_linestyle('-.')

    #loss_ax.set_xlabel('# queries')
    loss_ax.set_ylabel('avgerage regret' if args.avg_cum_regret else 'regret')
    loss_ax.set_xlim([1, max_iters])
    loss_ax.set_ylim([0, 1.05 * max_regret])
    prettify(loss_ax, max_iters)

    #time_ax.set_xlabel('# queries')
    time_ax.set_ylabel('cumulative time (s)')
    time_ax.set_xlim([1, max_iters])
    time_ax.set_ylim([0, 1.05 * max_time])
    prettify(time_ax, max_iters)

    loss_ax.set_title(args.title, fontsize=18)
    legend = loss_ax.legend(loc='upper right', fancybox=False, shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-large')
    for label in legend.get_lines():
        label.set_linewidth(2)
    mod = {True: 'avgcum', False: ''}[args.avg_cum_regret]
    loss_fig.savefig((args.png_basename +
                      '_loss_{}{}regret.png'.format(mod, args.regret)),
                     bbox_inches='tight', pad_inches=0, dpi=120)

    legend = loss_ax.legend(loc='upper right', fancybox=False, shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-large')
    for label in legend.get_lines():
        label.set_linewidth(2)
    time_fig.savefig(args.png_basename + '_time.png', bbox_inches='tight',
                     pad_inches=0, dpi=120)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('png_basename', type=str,
                        help='basename of the loss/time PNG plots')
    parser.add_argument('pickles', type=str, nargs='+',
                        help='comma-separated list of pickled results')
    parser.add_argument('-T', '--title', type=str, default='Title',
                        help='plot title')
    parser.add_argument('--avg-cum-regret', action='store_true', default=False,
                        help='whether to plot the average cumulative regret')
    parser.add_argument('-r', '--regret', type=str, default='min',
                        help='either min, avg, or max')
    parser.add_argument('--max-iters', type=int, default=None,
                        help='max iters')
    parser.add_argument('--max-regret', type=int, default=None,
                        help='max regret')
    parser.add_argument('--max-time', type=int, default=None,
                        help='max time')
    args = parser.parse_args()

    draw(args)
