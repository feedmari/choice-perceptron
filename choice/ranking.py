import numpy as np
from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold
from sklearn.utils import check_random_state
from time import time
from textwrap import dedent

from .utils import get_logger


def _delta(problem, query_phi, istar):
    set_size = len(query_phi)
    return query_phi[istar] - \
           query_phi[np.arange(set_size) != istar].sum(axis=0) / (set_size - 1)


def perceptron(dataset, step_size, w1, lastw=None):
    if lastw is None:
        return w1 + step_size * np.sum(dataset, axis=0)
    else:
        return lastw + step_size * dataset[-1]


def ep(dataset, step_size, w1, lastw=None):
    if lastw is None:
        nextw = w1 * np.exp(step_size * np.sum(dataset, axis=0))
    else:
        nextw = lastw * np.exp(step_size * dataset[-1])
    return nextw / np.sum(nextw)


def pa(dataset, _, w1, lastw=None):
    # XXX proof-of-concept, the margin M is wrong
    assert lastw is not None
    delta = dataset[-1]
    step_size = (10 - np.dot(lastw, delta)) / np.dot(delta, delta)
    print(step_size)
    return lastw + step_size * dataset[-1]


def svm(dataset, C, w1, lastw=None):
    n = dataset.shape[0]
    X = np.vstack([dataset, -dataset])
    y = np.hstack([[1] * n, [-1] * n])
    return LinearSVC(penalty='l2', loss='hinge', C=C, random_state=0) \
               .fit(X, y).coef_.ravel()


def crossvalidate(dataset, learn, hyperparams, w1):
    log = get_logger('rp')

    kfold = KFold(dataset.shape[0], n_folds=min(5, dataset.shape[0]))

    hyperparam_to_perf = {}
    for hyperparam in hyperparams:
        perfs = []
        for tr_index, ts_index in kfold:
            tr_dataset = dataset[tr_index,:]
            ts_dataset = dataset[ts_index,:]
            w = learn(tr_dataset, hyperparam, w1)
            utils = np.dot(ts_dataset, w)
            acc = np.sum(utils[utils > 0])
            perfs.append(acc.mean())
        hyperparam_to_perf[hyperparam] = np.array(perfs).mean()

    best_hyperparam = sorted(hyperparam_to_perf.items(),
                             key=lambda pair: pair[-1],
                             reverse=True)[0][0]

    log.info('CROSSVALIDATION perf={hyperparam_to_perf} best={best_hyperparam}', **locals())

    return best_hyperparam


def rp(problem, user, max_iters=100, learner='perceptron', hyperparam=1,
       cv_hyperparams=None, tradeoff_schedule='uniform', rng=None, **kwargs):
    rng = check_random_state(rng)
    log = get_logger('rp')

    if learner == 'ep':
        w = np.ones(problem.num_features) / problem.num_features
    else:
        w = rng.randint(-1, 2, size=problem.num_features)
    w1 = np.array(w)

    w_star = user.w_star

    log.info('w* = {w_star}', **locals())

    learn = {
        'perceptron': perceptron,
        'ep': ep,
        'pa': pa,
        'svm': svm,
    }[learner]

    if tradeoff_schedule == 'uniform':
        tradeoffs = None
    elif tradeoff_schedule == 'invlin':
        tradeoffs = 1 / np.linspace(1, max_iters + 1, num=max_iters)
    elif tradeoff_schedule == 'invsqrt':
        tradeoffs = 1 / np.sqrt(np.linspace(1, max_iters + 1, num=max_iters))
    else:
        raise ValueError('invalid tradeoff_schedule, got {}' \
                             .format(tradeoff_schedule))

    stop_on_satisfied = len(problem.cal_x()) <= 1
    trace, dataset = [], []
    for it in range(max_iters):
        t0 = time()
        tradeoff = None if tradeoffs is None else tradeoffs[it]

        # Receive context
        x = user.draw_x(it)

        # Query selection
        query_set = problem.select_query(w, x, tradeoff=tradeoff)
        t0 = time() - t0

        query_phi = np.array([problem.phi(x, y) for y in query_set])
        regrets = np.array([user.regret(x, y) for y in query_set])
        stats = regrets.min(), regrets.mean(), regrets.max()
        assert all(s >= 0 for s in stats)

        is_satisfied = any(user.is_satisfied(x, y) for y in query_set)
        istar = user.query_choice(x, query_set)

        t1 = time()
        delta = _delta(problem, query_phi, istar)
        alpha = np.dot(w_star, delta)
        beta = np.dot(w, delta)
        hamming = np.linalg.norm(delta, ord=1)
        dataset.append(delta)
        next_w = learn(np.array(dataset), hyperparam, w1, w)
        t1 = time() - t1

        sorted_query_set = [
            sorted(x.items(), key=lambda item: item[0])
            for x in query_set
        ]
        log.info('''
            ITERATION {it:3d}
            w = {w}
            x = {x}
            query_set = {sorted_query_set}
            query_phi = {query_phi}
            regret={stats} a={alpha:5.3f} b={beta:5.3f} |delta|={hamming}
            is_satisfied = {is_satisfied}
            istar = {istar}
            delta = {delta}
            next_w = {next_w}
        ''', **locals())

        trace.append(stats + (t0 + t1,))

        if is_satisfied and stop_on_satisfied:
            log.info('user satisfied in {} iterations'.format(it))
            break

        w = next_w

        if it >= 2 and cv_hyperparams is not None:
            hyperparam = crossvalidate(np.array(dataset), learn,
                                       cv_hyperparams, w1)
    else:
        log.info('user not satisfied after {} iterations'.format(it))

    return trace

