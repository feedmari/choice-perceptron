import numpy as np
from pymzn import MiniZincModel, minizinc
from . import get_logger, freeze

import re


def _indexset(attr_type):
    return re.match('array\[([^\]]+)\].+', attr_type).group(1)


class Problem(object):
    def __init__(self, template, set_size=2, tradeoff=0.5, dist_norm='l2',
                 dist_method='allvsall', keep=False, qsargmax=False, **kwargs):
        if set_size <= 1:
            raise ValueError('set_size must be >= 2, got {}'.format(set_size))
        if not (0 <= tradeoff <= 1):
            raise ValueError('tradeoff must be in [0, 1], got {}'.format(tradeoff))
        self.template = template
        self.set_size = set_size
        self.tradeoff = tradeoff
        self.dist_norm = dist_norm
        self.dist_method = dist_method
        self.keep = keep
        self._phis = {}
        self._num_features = None
        self.cost_matrix = None
        self.log = get_logger(__name__)
        self.qsargmax = qsargmax

    @property
    def num_features(self):
        if not self._num_features:
            features, _ = self._mzn_features()
            self._num_features = len(features)
        return self._num_features

    def _mzn_attributes(self, prefix=''):
        raise NotImplementedError()

    def _mzn_features(self, prefix=''):
        raise NotImplementedError()

    def _constraints(self, prefix=''):
        raise NotImplementedError()

    def draw_x(self, it):
        return {}

    def cal_x(self):
        return [{}]

    def utility(self, x, y, w):
        return np.dot(w, self.phi(x, y))

    def phi(self, x, y):
        """Returns the feature vector of (x, y)."""
        _frx = freeze({**x, **y})
        if _frx in self._phis:
            return self._phis[_frx]

        model = MiniZincModel(self.template)
        for attr, value in y.items():
            model.par(attr, value)
        features, phi_type = self._mzn_features()
        model.par('FEATURES', set(range(1, self.num_features + 1)))
        model.var('array[FEATURES] of {}'.format(phi_type), 'phi', features)
        model.solve('satisfy')
        _phi = minizinc(model, data={**x}, output_vars=['phi'],
                        keep=self.keep)[0]['phi']
        self._phis[_frx] = np.array(_phi, dtype=float)
        return self._phis[_frx]

    def infer(self, w, x):
        model = MiniZincModel(self.template)
        model.par('w', w)
        attributes = self._mzn_attributes()
        for attr, attr_type in attributes.items():
            if attr_type.startswith('array'):
                model.var(attr_type, attr + ' :: output_array([{}])'.format(_indexset(attr_type)))
            else:
                model.var(attr_type, attr)
        features, phi_type = self._mzn_features()
        model.par('FEATURES', set(range(1, self.num_features + 1)))
        model.var('array[FEATURES] of var {}'.format(phi_type), 'phi', features)
        constraints = self._constraints()
        for constr in constraints:
            model.constraint(constr)
        model.var('var float',  'utility', 'sum(i in FEATURES)(w[i] * phi[i])')
        model.solve('maximize utility')
        return minizinc(model, data={**x}, output_vars=attributes.keys(),
                        keep=self.keep)[0]

    def select_query(self, w, x, tradeoff=None, discretize=True):
        FACTOR = 10

        phi_dist_util_type = 'float'
        if discretize:
            w = (w * FACTOR).astype(int)
            phi_dist_util_type = 'int'

        tradeoff = tradeoff or self.tradeoff

        if tradeoff >= 0.9:
            tradeoff = 0.9
        elif tradeoff <= 0.1:
            tradeoff = 0.1

        self.log.debug('computing query {} (set_size={} tradeoff={} dist_norm={} dist_method={})' \
                       .format(w, self.set_size, tradeoff, self.dist_norm, self.dist_method))

        max_y = self.infer(w, x)
        max_phi = self.phi(x, max_y)

        model = MiniZincModel(self.template)
        model.par('w', w)
        model.par('FEATURES', set(range(1, self.num_features + 1)))
        if self.qsargmax:
            model.par('y1_phi', max_phi.astype(int) if discretize else max_phi)
        else:
            model.par('max_phi', max_phi.astype(int) if discretize else max_phi)

        output_vars = []
        start = 2 if self.qsargmax else 1
        for i in range(start, self.set_size + 1):
            prefix = 'y{}_'.format(i)
            attributes = self._mzn_attributes(prefix)
            features, phi_type = self._mzn_features(prefix)
            for attr, attr_type in attributes.items():
                output_vars.append(attr)
                if attr_type.startswith('array'):
                    model.var(attr_type, attr + ' :: output_array([{}])'.format(_indexset(attr_type)))
                else:
                    model.var(attr_type, attr)
            model.var('array[FEATURES] of var {}'.format(phi_type),
                      '{}phi'.format(prefix), features)

        if self.dist_norm == 'l2':
            distij = ('sqrt(sum(i in FEATURES)('
                        '{prefix_i}phi[i] * {prefix_i}phi[i] - '
                        '2 * {prefix_i}phi[i] * {prefix_j}phi[i] + '
                        '{prefix_j}phi[i] * {prefix_j}phi[i]))')
        else:
            distij = ('sum(i in FEATURES)('
                        'abs({prefix_i}phi[i] - {prefix_j}phi[i]))')

        delta, ijpairs = [], []
        if self.dist_method == 'allvsall':
            for i in range(1, self.set_size + 1):
                for j in range(i + 1, self.set_size + 1):
                    prefix_i = 'y{}_'.format(i)
                    prefix_j = 'y{}_'.format(j)
                    dist = distij.format(**locals())
                    model.var('var float', 'dist{}{}'.format(i, j), dist)
                    delta.append('dist{}{}'.format(i, j))
                    ijpairs.append((i, j))
            dist_normalization = 2 / (self.set_size * (self.set_size - 1))
        else:
            for i in range(1, self.set_size + 1):
                for j in range(i + 1, self.set_size + 1):
                    prefix_i = 'y{}_'.format(i)
                    prefix_j = 'y{}_'.format(j)
                    dist = distij.format(**locals())
                    model.var('var int', 'dist{}{}'.format(i, j), dist)
                    if i == 1:
                        delta.append('dist{}{}'.format(1, j))
                    ijpairs.append((i, j))
            dist_normalization = 1 / self.set_size

        model.var('var ' + phi_dist_util_type, 'delta', ' + '.join(delta))

        # All configurations must be different
        for i, j in ijpairs:
            model.constraint('dist{}{} > 0'.format(i, j))

        # Utilities
        utils = []
        for i in range(start, self.set_size + 1):
            prefix = 'y{}_'.format(i)
            util = '{}util'.format(prefix)
            model.var('var ' + phi_dist_util_type, util,
                      'sum(i in FEATURES)(w[i] * {}phi[i])'.format(prefix))
            utils.append(util)

        model.var('var ' + phi_dist_util_type, 'mu', ' + '.join(utils))

        if not self.qsargmax:
            # The utility of the first optimal object is known
            max_util_diff = '0' if discretize else '1e-9'
            model.constraint('sum(i in FEATURES)(w[i] * (max_phi[i] - y1_phi[i])) <= {}' \
                                .format(max_util_diff))

        # Constraints on the xs
        for i in range(start, self.set_size + 1):
            prefix = 'y{}_'.format(i)
            constraints = self._constraints(prefix)
            for constr in constraints:
                model.constraint(constr)

        # NOTE if w == 0, normalizing leads to a constantly-zero objective
        # function == 0; however, when w == 0 there is no need to balance
        # between the distance and utility terms (the second one is zero),
        # so we simply remove ther norm-of-w from the objective.
        norm = 1 if (w == 0).all() else np.linalg.norm(w)

        l1factor = 1 if self.dist_norm == 'l2' else np.sqrt(self.num_features)

        a = tradeoff * norm * dist_normalization
        b = ((1 - tradeoff) * l1factor) / self.set_size
        if discretize:
            a, b = int(FACTOR * a), int(FACTOR * b)

        model.var('var ' + phi_dist_util_type, 'objective',
                  '{} * delta + {} * mu'.format(a, b))
        model.solve('maximize objective')

        assignment = minizinc(model, data=x, keep=self.keep, parallel=0,
			      timeout=20, suppress_segfault=True,
                              output_vars=output_vars)[0]

        attributes = self._mzn_attributes()
        query_set = [max_y] if self.qsargmax else []
        for i in range(start, self.set_size + 1):
            prefix = 'y{}_'.format(i)
            y = {}
            for attr in attributes:
                y[attr] = assignment[prefix + attr]
            query_set.append(y)

        return query_set

