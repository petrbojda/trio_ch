import numpy as np
import data_containers as dc

class TrackManager(object):

    def __init__(self):
        super().__init__()
        # TODO: add list of tracks
        self._Tsampling = (0,0)
        self._Vx_max = (0,0)
        self._Vy_max = (0,0)
        self._velx_interval = (0,0)
        self._flt_type = (0,0)


    def calc_gate(center_point,d_x,d_y):
        gate = np.array([0,0,0,0])
        gate[0] = center_point.x - .5 * d_x
        gate[1] = center_point.x + .5 * d_x
        gate[2] = center_point.y - .5 * d_y
        gate[3] = center_point.y + .5 * d_y
        return gate

    def calc_two_point_projection(start_point,end_point,ts):
        projected_point = TrackPoint(None,None,None,None,None)
        projected_point.mcc = end_point.mcc + 1
        projected_point.x = 2*end_point.x - start_point.x
        projected_point.y = 2*end_point.y - start_point.y
        projected_point.dx = (end_point.x - start_point.x) / ts
        projected_point.dy = (end_point.x - start_point.x) / ts
        return projected_point

    # TODO: add methods to assign detection to an existing track from the list or to start a new track
