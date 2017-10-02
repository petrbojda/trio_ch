import numpy as np
import data_containers as dc
import tracking_filters as tflt

class FilterManager(list):

    def __init__(self, gate=None, flt_type='G-H', Tsampling=50.0e-3):
        super().__init__()
        self._Tsampling = Tsampling
        self._gate_ = gate
        self._flt_type = flt_type
        self._n_of_Trackers = np.array([0])

        if gate is None:
            self._gate = {'x': 3, 'y': 1, 'dx': 0.65, 'dy': 0.3}
        else:
            self._gate = gate

    def create_new_filter(self):
        self._n_of_Trackers = np.append(self._n_of_Trackers, self._n_of_Trackers[-1] + 1)
        self.append(tflt.Tracker(self._n_of_Trackers[-1]))


