import numpy as np
from . import Problem
from sklearn.utils import check_random_state


def random_rectangles(canvas_size=10, num_rects=10, rng=None):
    rng = check_random_state(rng)
    return rng.randint(1, canvas_size + 1, size=(num_rects, 4))


TEMPLATE = """\
int: CANVAS_SIZE = {canvas_size};
set of int: CANVAS = 1 .. CANVAS_SIZE;
"""

class Rectangles(Problem):
    def __init__(self, rects=None, canvas_size=10, num_rects=10, **kwargs):
        if rects:
            assert np.all([v >= 1 and v <= canvas_size for v in rects])
            self.rects = rects
        else:
            self.rects = random_rectangles(canvas_size, num_rects)
        self.canvas_size = canvas_size
        template = TEMPLATE.format(**locals())
        super().__init__(template, **kwargs)

    def cal_x(self):
        return [{}]

    def _mzn_attributes(self, prefix=''):
        return {'{}x'.format(prefix): 'var CANVAS',
                '{}y'.format(prefix): 'var CANVAS'}

    def _mzn_features(self, prefix=''):
        features = []
        for (a, b, c, d) in self.rects:
            min_x, max_x = min(a, b), max(a, b)
            min_y, max_y = min(c, d), max(c, d)
            x = '{}x'.format(prefix)
            y = '{}y'.format(prefix)
            is_inside = ('{x} >= {min_x} /\ {x} <= {max_x} /\ '
                         '{y} >= {min_y} /\ {y} <= {max_y}').format(**locals())
            features.append('2 * ({}) - 1'.format(is_inside))
        return features, 'int'

    def _constraints(self, prefix=''):
        return []

