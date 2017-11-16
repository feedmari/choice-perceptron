import numpy as np
from scipy.misc import logsumexp
from sklearn.utils import check_random_state
from . import freeze


def _sparsify(w, problem, density=1.0, rng=None):
    if not(0 < density <= 1):
        raise ValueError('density must be in (0, 1], got {}'.format(density))
    rng = check_random_state(rng)
    perm = rng.permutation(w.shape[0])
    w[perm[:round((1 - density) * problem.num_features)]] = 0
    return w


def sample_users(problem, mode='normal', density=1.0, non_negative=False,
                 rng=None, **kwargs):
    rng = check_random_state(rng)
    num_features = (problem.num_features if problem.cost_matrix is None else
                    sum(problem.cost_matrix.shape))
    if mode == 'uniform':
        w = rng.uniform(1, 100, size=num_features)
    elif mode == 'normal':
        w = rng.normal(25, 25 / 3, size=num_features)
    else:
        raise ValueError('invalid sampling mode, got {}'.format(mode))
    if non_negative:
        w = np.abs(w)
    return _sparsify(w, problem, density, rng)


class User(object):
    def __init__(self, problem, w_star, min_regret=0, uid=0, rng=None, **kwargs):

        if problem.cost_matrix is not None:
            num_dep, num_indep = problem.cost_matrix.shape
            w_indep, w_dep = w_star[:num_indep], w_star[-num_dep:]
            w_star = w_indep + np.dot(problem.cost_matrix.T, w_dep)

        self.problem = problem
        self.w_star = w_star
        self.min_regret = min_regret
        self.uid = uid
        self.rng = check_random_state(rng)
        self.y_stars = {}
        self.u_stars = {}

    def draw_x(self, it):
        return self.problem.draw_x(it)

    def utility(self, x, y):
        return np.dot(self.w_star, self.problem.phi(x, y))

    def regret(self, x, y):
        _frx = freeze(x)
        if _frx in self.u_stars:
            u_star = self.u_stars[_frx]
        else:
            y_star = self.problem.infer(self.w_star, x)
            self.y_stars[_frx] = y_star
            u_star = self.utility(x, y_star)
            self.u_stars[_frx] = u_star
        reg = u_star - self.utility(x, y)
        assert reg >= 0
        return reg

    def is_satisfied(self, x, y):
        return self.regret(x, y) <= self.min_regret

    def query_choice(self, x, query_set):
        raise NotImplementedError()


class NoiselessUser(User):

    def query_choice(self, x, query_set):
        if len(query_set) < 2:
            err = 'Expected set_size >= 2, got {}'
            raise ValueError(err.format(len(query_set)))
        utils = np.array([self.utility(x, y) for y in query_set])
        pvals = np.array([u == utils.max() for u in utils]) / utils.sum()
        return np.argmax(self.rng.multinomial(1, pvals=(pvals / pvals.sum())))


class PlackettLuceUser(User):

    def __init__(self, problem, w_star, lmbda=1.0, **kwargs):
        super().__init__(problem, w_star, **kwargs)
        self.lmbda = lmbda

    def query_choice(self, x, query_set):
        if len(query_set) < 2:
            err = 'Expected set_size >= 2, got {}'
            raise ValueError(err.format(len(query_set)))
        utils = np.array([self.lmbda * self.utility(x, y) for y in query_set])
        sumexputils = logsumexp(utils)
        pvals = np.exp(utils - sumexputils)
        return np.argmax(self.rng.multinomial(1, pvals=pvals))

