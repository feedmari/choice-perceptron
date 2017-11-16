from . import Problem

TEMPLATE = """\
int: DOMAIN_SIZE = {domain_size};
set of int: DOMAIN = 1 .. DOMAIN_SIZE;
"""

class Cartesian(Problem):
    def __init__(self, domain_size=4, **kwargs):
        self.domain_size = self.num_attributes = domain_size
        template = TEMPLATE.format(**locals())
        super().__init__(template, **kwargs)

    def cal_x(self):
        return [{}]

    def _mzn_attributes(self, prefix=''):
        attributes = {}
        for n in range(1, self.num_attributes + 1):
            attr = '{}a{}'.format(prefix, n)
            attributes[attr] = 'var DOMAIN'
        return attributes

    def _mzn_features(self, prefix=''):
        features = []
        for n in range(1, self.num_attributes + 1):
            for v in range(1, self.domain_size + 1):
                features.append('{}a{} == {}'.format(prefix, n, v))
        return features, 'int'

    def _constraints(self, prefix=''):
        return []

