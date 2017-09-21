import numpy as np
import data_containers as dc

class TrackManager(list):

    def __init__(self, flt_type="G-H", Tsampling=50.0e-3, gate_dx=5, gate_dy=5):
        super().__init__()

        self._Tsampling = Tsampling
        self._gate_dx = gate_dx
        self._gate_dy = gate_dy

        self._flt_type = flt_type
        self._n_of_Tracks = np.array([0])


    def create_new_track(self):
        self.append(dc.Track(self._n_of_Tracks))
        self._n_of_Tracks = np.append(self._n_of_Tracks,self._n_of_Tracks[-1]+1)


    def test_det_in_gate(gate,detection):

        is_in_x = (gate.x-gate.dx < detection._x) and (gate.x+gate.dx > detection._x)
        is_in_y = (gate.y-gate.dy < detection._y) and (gate.y+gate.dy > detection._y)

        return is_in_x and is_in_y

    def calc_two_point_projection(start_point,end_point,ts):
        projected_point = dc.TrackPoint(None,None,None,None,None)
        projected_point.mcc = end_point.mcc + 1
        projected_point.x = 2*end_point.x - start_point.x
        projected_point.y = 2*end_point.y - start_point.y
        projected_point.dx = (end_point.x - start_point.x) / ts
        projected_point.dy = (end_point.x - start_point.x) / ts
        return projected_point

    def test_detections(self,mcc,noDet,lst_detections):


