import pymzn
import pickle

from itertools import combinations
from . import Problem

from sklearn.utils import check_random_state


TEMPLATE = """\
include "globals.mzn";

int: T = {horizon};
set of int: TIME = 1..T;
set of int: TIME1 = 1..T-1;

int: NUM_LOCATIONS = {num_locations};
set of int: LOCATIONS = 1..NUM_LOCATIONS;
set of int: LOCATIONS1 = 1..NUM_LOCATIONS+1;
int: NO_LOCATION = NUM_LOCATIONS+1;

int: N_REGIONS = {num_regions};
set of int: REGIONS = 1..N_REGIONS;
set of int: REGIONS1 = 1..N_REGIONS+1;
int: NO_REGION = N_REGIONS+1;
array[LOCATIONS] of REGIONS: LOCATION_REGION = {location_region};

int: NUM_ACTIVITIES = {num_activities};
set of int: ACTIVITIES = 1..NUM_ACTIVITIES;

array[LOCATIONS1, ACTIVITIES] of 0..1: LOCATION_ACTIVITIES = {location_activities};
array[LOCATIONS1] of int: LOCATION_COST = {location_cost};
array[LOCATIONS, LOCATIONS] of int: TRAVEL_TIME = {travel_time};

int: NUM_NECESSARY_LOCATIONS;
array[1..NUM_NECESSARY_LOCATIONS] of int: necessary_locations;
"""



class Travel(Problem):

    def __init__(self, horizon=10, dataset='10', rng=None, **kwargs):

        self._horizon = horizon
        self.rng = check_random_state(rng)

        with open("datasets/travel_tn{}.pickle".format(dataset), "rb") as fp:
            dataset = pickle.load(fp)

        self._location_activities = dataset["location_activities"]
        self._num_locations = dataset["location_activities"].shape[0] - 1
        self._num_activities = dataset["location_activities"].shape[1]
        self._location_cost = dataset["location_cost"]
        self._travel_time = dataset["travel_time"]
        self._regions = dataset["regions"]
        self._num_regions = dataset["num_regions"]
        self._cal_x = None

        template = TEMPLATE.format(
                horizon=self._horizon,
                num_locations=self._num_locations,
                num_activities=self._num_activities,
                location_activities=pymzn.dzn_value(self._location_activities),
                location_cost=pymzn.dzn_value(self._location_cost),
                travel_time=pymzn.dzn_value(self._travel_time),
                location_region=pymzn.dzn_value(self._regions),
                num_regions=self._num_regions)
        super().__init__(template, **kwargs)
        self._drawn_xs = []

    def _mzn_attributes(self, prefix=''):
        return {
            '{prefix}location'.format(**locals()): 'array[TIME] of var LOCATIONS1',
            '{prefix}duration'.format(**locals()): 'array[TIME] of var int',
            '{prefix}travel'.format(**locals()): 'array[TIME1] of var int',
            '{prefix}location_counts'.format(**locals()): 'array[LOCATIONS] of var int',
            '{prefix}n_different_locations'.format(**locals()): 'var int',
            '{prefix}travel_time'.format(**locals()): 'var int',
            '{prefix}regions'.format(**locals()): 'array[TIME] of var REGIONS1',
            '{prefix}region_counts'.format(**locals()): 'array[REGIONS1] of var 0..NUM_LOCATIONS',
            '{prefix}n_different_regions'.format(**locals()): 'var int'
        }

    def draw_x(self, it):
        if len(self._drawn_xs) == 0:
            self._drawn_xs = self.rng.choice(range(len(self.cal_x())), 100)
        idx = self._drawn_xs[it]
        cal_x = self.cal_x()
        x = list(cal_x[idx])
        return {'NUM_NECESSARY_LOCATIONS': len(x), 'necessary_locations': x}

    def cal_x(self):
        if self._cal_x:
            return self._cal_x
        self._cal_x = []
        for i in range(2, 4):
            self._cal_x += list(combinations(range(1, self._num_locations), i))
        return self._cal_x

    def _mzn_features(self, prefix=''):
        features = []

        # Number of time slots spent in a location
        for location in range(1, self._num_locations + 1):
            feature = 'sum(i in TIME)({prefix}location[i] = {location})'.format(**locals())
            features.append(feature)

        # Number of time slots with access to an activity
        for activity in range(1, self._num_activities + 1):
            feature = 'sum(i in TIME)(LOCATION_ACTIVITIES[{prefix}location[i], {activity}])'.format(**locals())
            features.append(feature)

        # Number of distinct locations
        features.append('{prefix}n_different_locations'.format(**locals()))

        # Total time spent traveling
        features.append('sum(i in TIME1)({prefix}travel[i])'.format(**locals()))

        # Total cost
        features.append('sum(i in TIME)(LOCATION_COST[{prefix}location[i]])'.format(**locals()))

        for region in range(1, self._num_regions + 1):
            features.append('2 * ({prefix}region_counts[{region}] + {prefix}travel_time == T) - 1'.format(**locals()))

        # Number of different regions
        features.append('{prefix}n_different_regions'.format(**locals()))

        for location1, location2 in combinations(range(1, self._num_locations + 1), 2):
            features.append('2 * ({prefix}location_counts[{location1}] = 0 \/ {prefix}location_counts[{location2}] > 0) - 1'.format(**locals()))
            features.append('2 * ({prefix}location_counts[{location2}] = 0 \/ {prefix}location_counts[{location1}] > 0) - 1'.format(**locals()))

        return features, 'int'

    def _constraints(self, prefix=''):
        constraints = []

        # Hard context
        constraints.append('global_cardinality_low_up({prefix}location, necessary_locations, [1 | i in 1..NUM_NECESSARY_LOCATIONS], [T | i in 1..NUM_NECESSARY_LOCATIONS])'.format(**locals()))

        # count number of distinct location
        constraints.append('global_cardinality({prefix}location, [i | i in LOCATIONS], {prefix}location_counts)'.format(**locals()))
        constraints.append('{prefix}n_different_locations = among({prefix}location_counts, LOCATIONS)'.format(**locals()))

        # bounds on duration
        constraints.append('forall(i in TIME)({prefix}duration[i] >= 0)'.format(**locals()))
        constraints.append('forall(i in TIME)({prefix}duration[i] <= T)'.format(**locals()))

        # bounds on travel
        constraints.append('forall(i in TIME1)({prefix}travel[i] >= 0)'.format(**locals()))
        constraints.append('forall(i in TIME1)({prefix}travel[i] <= T)'.format(**locals()))

        # trip must fit into available time
        constraints.append('(sum(i in TIME)({prefix}duration[i]) + sum(i in TIME1)({prefix}travel[i])) = T'.format(**locals()))

        # location implies durations and conversely
        constraints.append('forall(i in TIME)({prefix}location[i] = NO_LOCATION <-> {prefix}duration[i] = 0)'.format(**locals()))

        # null travels are only at the end
        constraints.append('forall(i in TIME1)({prefix}location[i+1] = NO_LOCATION <-> {prefix}travel[i] = 0)'.format(**locals()))

        # consecutive locations must be different
        constraints.append('forall(i in TIME1 where {prefix}location[i] != NO_LOCATION)({prefix}location[i] != {prefix}location[i+1])'.format(**locals()))

        # null locations are only at the end
        constraints.append('forall(i in TIME1 where {prefix}location[i] = NO_LOCATION)({prefix}location[i+1] = NO_LOCATION)'.format(**locals()))

        # traveling from one location to another takes time
        constraints.append('forall(i in TIME1 where {prefix}location[i+1] != NO_LOCATION)({prefix}travel[i] = TRAVEL_TIME[{prefix}location[i], {prefix}location[i+1]])'.format(**locals()))

        constraints.append('{prefix}travel_time = sum({prefix}travel)'.format(**locals()))

        # regions
        constraints.append('{prefix}regions = [if {prefix}location[t] == NO_LOCATION then NO_REGION else LOCATION_REGION[{prefix}location[t]] endif | t in TIME]'.format(**locals()))
        constraints.append('global_cardinality({prefix}regions, [i | i in REGIONS1], {prefix}region_counts)'.format(**locals()))
        constraints.append('{prefix}n_different_regions = among({prefix}region_counts, 1..N_REGIONS)'.format(**locals()))

        constraints.append('{prefix}location[1] = 1'.format(**locals()));

        return constraints

