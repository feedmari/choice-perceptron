import numpy as np
from . import Problem


TEMPLATE = """\
set of int: TYPES = 1 .. 3;
set of int: MANUFACTURERS = 1 .. 8;
set of int: CPUS = 1 .. 37;
set of int: MONITORS = 1 .. 8;
set of int: RAMS = 1 .. 10;
set of int: HDS = 1 .. 10;
"""

class PC(Problem):
    _ATTRIBUTES = [
        ('cpu', 37),
        ('hd', 10),
        ('manufacturer', 8),
        ('ram', 10),
        ('monitor', 8),
        ('pctype', 3),
    ]

    _ATTR_TO_COSTS = {
        'pctype': [50, 0, 80],
        'manufacturer': [100, 0, 100, 50, 0, 0, 50, 50],
        'cpu' : [
            1.4*100, 1.4*130, 1.1*70, 1.1*90, 1.2*80, 1.2*50, 1.2*60, 1.2*80,
            1.2*90, 1.2*100, 1.2*110, 1.2*120, 1.2*130, 1.2*140, 1.2*170,
            1.5*50, 1.5*60, 1.5*80, 1.5*90, 1.5*100, 1.5*110, 1.5*130, 1.5*150,
            1.5*160, 1.5*170, 1.5*180, 1.5*220, 1.4*27, 1.4*30, 1.4*40, 1.4*45,
            1.4*50, 1.4*55, 1.4*60, 1.4*70, 1.6*70, 1.6*73,
        ],
        'monitor': [
            0.6*100, 0.6*104, 0.6*120, 0.6*133, 0.6*140, 0.6*150, 0.6*170,
            0.6*210
        ],
        'ram': [
            0.8*64, 0.8*128, 0.8*160, 0.8*192, 0.8*256, 0.8*320, 0.8*384,
            0.8*512, 0.8*1024, 0.8*2048
        ],
        'hd': [
            4*8, 4*10, 4*12, 4*15, 4*20, 4*30, 4*40, 4*60, 4*80, 4*120
        ],
    }

    def __init__(self, **kwargs):
        super().__init__(TEMPLATE, **kwargs)
        self.cost_matrix = np.hstack([
            np.array(self._ATTR_TO_COSTS[attr], dtype=float)
            for attr, _ in self._ATTRIBUTES
        ]).reshape(1, -1) / 2754.4

    def _mzn_attributes(self, prefix=''):
        return {
            '{}pctype'.format(prefix): 'var TYPES',
            '{}manufacturer'.format(prefix): 'var MANUFACTURERS',
            '{}cpu'.format(prefix): 'var CPUS',
            '{}monitor'.format(prefix): 'var MONITORS',
            '{}ram'.format(prefix): 'var RAMS',
            '{}hd'.format(prefix): 'var HDS',
        }

    def cal_x(self):
        return [{}]

    def _mzn_features(self, prefix=''):
        features = []
        for attr, dom in self._ATTRIBUTES:
            for v in range(1, dom + 1):
                features.append('{}{} == {}'.format(prefix, attr, v))
        return features, 'int'

    def _constraints(self, prefix=''):
        constraints = []

        pctype = '{}pctype'.format(prefix)
        manufacturer = '{}manufacturer'.format(prefix)
        cpu = '{}cpu'.format(prefix)
        monitor = '{}monitor'.format(prefix)
        ram = '{}ram'.format(prefix)
        hd = '{}hd'.format(prefix)

        # Manufacturer -> Type
        constraints.append('({manufacturer} = 2) -> ({pctype} in {{1, 2}})'.format(**locals()))
        constraints.append('({manufacturer} = 4) -> ({pctype} = 1)'.format(**locals()))
        constraints.append('({manufacturer} = 6) -> ({pctype} = 2)'.format(**locals()))
        constraints.append('({manufacturer} = 7) -> ({pctype} in {{1, 3}})'.format(**locals()))

        # Manufacturer -> CPU
        constraints.append('({manufacturer} = 1) -> ({cpu} in 28 .. 37)'.format(**locals()))
        constraints.append('({manufacturer} in {{2, 7}}) -> ({cpu} in 1 .. 4 union 6 .. 27)'.format(**locals()))
        constraints.append('({manufacturer} = 4) -> ({cpu} in 5 .. 27)'.format(**locals()))
        constraints.append('({manufacturer} in {{3, 5, 8}}) -> ({cpu} in 6 .. 27)'.format(**locals()))
        constraints.append('({manufacturer} = 6) -> ({cpu} in 16 .. 27)'.format(**locals()))

        # Type -> RAM
        constraints.append('({pctype} = 1) -> ({ram} <= 9)'.format(**locals()))
        constraints.append('({pctype} = 2) -> ({ram} in {{2, 5, 8, 9}})'.format(**locals()))
        constraints.append('({pctype} = 3) -> ({ram} in {{5, 8, 9, 10}})'.format(**locals()))

        # Type -> HD
        constraints.append('({pctype} = 1) -> ({hd} <= 6)'.format(**locals()))
        constraints.append('({pctype} >= 2) -> ({hd} >= 5)'.format(**locals()))

        # Type -> Monitor
        constraints.append('({pctype} = 1) -> ({monitor} <= 6)'.format(**locals()))
        constraints.append('({pctype} >= 2) -> ({monitor} >= 6)'.format(**locals()))

        return constraints

