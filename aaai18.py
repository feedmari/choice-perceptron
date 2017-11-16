#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
import pickle
import logging
import choice
import threading

from sklearn.utils import check_random_state


PROBLEMS = {
    'cartesian': choice.Cartesian,
    'rectangles': choice.Rectangles,
    'pc': choice.PC,
    'travel': choice.Travel
}


USERS = {
    'noiseless': choice.NoiselessUser,
    'pl': choice.PlackettLuceUser,
}


def _get_results_path(args):
    causal_args = [
        args['problem'], args['num_users'], args['learner'], args['max_iters'],
        args['seed'], args['set_size'], args['hyperparam'], args['tradeoff'],
        args['tradeoff_schedule'], args['dist_norm'], args['dist_method'],
        args['user_model'], args['noise'], args['sampling_mode'],
        args['density'], args['non_negative'], args['min_regret'],
        args['qsargmax']
    ]
    return os.path.join('results', '_'.join(map(str, causal_args)) + '.pickle')


def _load_weights(path):
    try:
        return choice.load(path)
    except pickle.UnpicklingError:
        return pd.read_csv(path, header=None).values


def gen_users(args):
    rng = check_random_state(args['seed'])
    nopargs = choice.subdict(args, nokeys={'problem'})
    problem = PROBLEMS[args['problem']](**nopargs)

    users = []
    for uid in range(1, args['num_users'] + 1):
        user_weights = choice.sample_users(problem, rng=rng, **nopargs)
        users.append(user_weights)
    return users


def generate_users(args):
    if not args['weights']:
        raise ValueError('Argument weights must be given.')
    users = gen_users(args)
    with open(args['weights'], 'wb') as f:
        pickle.dump(users, f)


def experiment(args):
    rng = check_random_state(args['seed'])
    nopargs = choice.subdict(args, nokeys={'problem'})
    problem = PROBLEMS[args['problem']](rng=rng, **nopargs)

    if args['weights']:
        weights = _load_weights(args['weights'])
    else:
        weights = gen_users(args)
    users = [USERS[args['user_model']](problem, weights[uid], rng=rng, uid=uid,
                                       lmbda=args['noise'], **nopargs)
             for uid in range(args['num_users'])]

    start_user = args['start_user']
    pd = args['parallel']
    pdl = (len(users)) // pd + 1
    batches = [users[u : u + pdl] for u in range(start_user, len(users), pdl)]

    traces = []
    def _exp(_users):
        for user in _users:
            traces.append(choice.rp(problem, user, rng=rng, **nopargs))

    threads = []
    for batch in batches:
        t = threading.Thread(target=_exp, args=(batch,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    choice.dump(_get_results_path(args), {'args': args, 'traces': traces})


FUNCTIONS = {
    'gen': generate_users,
    'exp': experiment
}


if __name__ == '__main__':
    import argparse

    np.seterr(all='raise')

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument('function', choices=FUNCTIONS.keys(),
                        help='The function to execute')
    parser.add_argument('problem', type=str,
                        help='the problem, any of {}'.format(list(PROBLEMS.keys())))
    parser.add_argument('-N', '--num-users', type=int, default=20,
                        help='number of users in the experiment')
    parser.add_argument('-s', '--start-user', type=int, default=0,
                        help='user to start with')
    parser.add_argument('-T', '--max-iters', type=int, default=100,
                        help='number of trials')
    parser.add_argument('-r', '--seed', type=int, default=0,
                        help='RNG seed')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='enable debug spew')
    parser.add_argument('--keep', action='store_true',
                        help='keep mzn files around')

    group = parser.add_argument_group('Learning')
    group.add_argument('-L', '--learner', type=str, default='perceptron',
                       help='perceptron or svm')
    group.add_argument('-C', '--hyperparam', type=float, default=1,
                       help='perceptron step size')
    group.add_argument('-X', '--cv-hyperparams', nargs='+', type=float, default=None,
                       help='list of crossvalidation hyperparameters')

    group = parser.add_argument_group('Query Selection')
    group.add_argument('-k', '--set-size', type=int, default=2,
                       help='set size')
    group.add_argument('-c', '--tradeoff', type=float, default=0.1,
                       help='distance-quality trade-off')
    group.add_argument('--tradeoff-schedule', type=str, default='uniform',
                       help='distance-quality trade-off at diff iterations, '\
                            'either uniform, invlin or invsqrt')
    group.add_argument('-n', '--dist-norm', type=str, default='l1',
                       help='distance norm')
    group.add_argument('-D', '--dist-method', type=str, default='firstvsall',
                       help='distance maximization method')
    group.add_argument('-M', '--qsargmax', action='store_true',
                       help='Precalculate argmax in query selection')

    group = parser.add_argument_group('User Simulation')
    group.add_argument('-W', '--weights', type=str, default=None,
                       help='path to pickle or txt file with weight matrix')
    group.add_argument('-U', '--user-model', type=str, default='pl',
                       help='user response model for choice queries')
    group.add_argument('-E', '--noise', type=float, default=1.0,
                       help='amount of user noise (pl user model only)')
    group.add_argument('-S', '--sampling-mode', type=str, default='normal',
                       help='user weight sampling mode')
    group.add_argument('-d', '--density', type=float, default=1,
                       help='percentage of non-zero user weights')
    group.add_argument('--non-negative', action='store_true', default=False,
                       help='whether the weights should be non-negative')
    group.add_argument('-p', '--parallel', type=int, default=1,
                        help=('The parallelism degree over the users'))
    group.add_argument('--min-regret', type=float, default=0,
                       help='minimum regret for satisfaction')


    args = parser.parse_args()

    handlers = []
    if args.verbose:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.DEBUG, handlers=handlers,
                        format='%(levelname)-6s %(name)-14s: %(message)s')

    FUNCTIONS[args.function](vars(args))

