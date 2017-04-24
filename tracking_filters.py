import numpy as np
import data_containers as dc


class TrackPoint(object):
    def __init__(self,mcc,x,y,dx,dy):
        self.mcc = mcc
        self.x = x
        self.dx = dx
        self.y = y
        self.dy = dy

    def get_array(self):
        x = np.array([self.x,self.dx,self.y,self.dy])
        return x.reshape(4,1)

class Tracker(object):
    """ Class to encapsulate an estimator - tracker is defined here"""
    def __init__(self,start_mcc, **kwarg):

        if 'beta' in kwarg:
            beta = float(kwarg['beta'])
        else:
            beta = 0.5

        if 'ts' in kwarg:
            self._ts = float(kwarg['ts'])
        else:
            self._ts = 0.05

        if 'v_max' in kwarg:
            self._vx_max = float(kwarg['v_max'])
        else:
            self._vx_max = 138.0

        if 'vc_max' in kwarg:
            self._vy_max = float(kwarg['vc_max'])
        else:
            self._vy_max = 49.0

        #Qtmp = [(dT^2)/3 dT/2; dT/2 1]*dT*StateVar
        #Rtmp = 1*ObserVar
        #xtmp = [0;0]
        #Ptmp = Qtmp*10
        g = 1-beta**2
        h = (1-beta)**2
        K = np.array([g,h])
        K = K.reshape(2,1)
        Phi = np.array([[1,self._ts],[0,1]])
        H = np.array([0,1])

        self._list_det = dc.DetectionList()  # Initialize a list of detections
        self._list_tpts = []                # Initialize a list of tracking points
        self._start_mcc = start_mcc

        self._K = np.kron(np.eye(2),K)
        self._Phi = np.kron(np.eye(2),Phi)
        self._H = np.kron(np.eye(2),H)

        self._selection_gate = np.array([0,0,0,0])
        self._expected_TP = TrackPoint(None,None,None,None,None)

    def append_detection(self,detection):
        self._list_det.append(detection)

    def extend_detection_list(self,radar_data_list, **kwarg):
        self._list_det.extend_with_selection(radar_data_list, **kwarg)

    def update_track(self):
        if self._list_det:
            last_det = self._list_det[-1]
            if len(self._list_tpts) >= 2:
                if last_det._mcc > self._list_tpts[-1]:
                    x_m = np.array([last_det._x,0,last_det._y,0]).reshape(4,1)
                    x_n = self._list_tpts[-1].get_array()
                    x_tpt = calc_state_fading_mem_fltr(x_n,x_m,self._Phi,self._K,self._H,last_det._mcc)
                    self._list_tpts.append(x_tpt)
                    self._expected_TP = calc_two_point_projection(self._list_tpts[-2],
                                                                  self._list_tpts[-1])
            elif len(self._list_tpts) == 1:
                if last_det._mcc != self._list_tpts[-1]:
                    dx = (last_det._x - self._list_tpts[-1].x) / self._ts
                    dy = (last_det._y - self._list_tpts[-1].y) / self._ts
                    self._list_tpts.append(TrackPoint(last_det._mcc,last_det._x,last_det._y,dx,dy))
                    self._expected_TP = calc_two_point_projection(self._list_tpts[-2],
                                                                  self._list_tpts[-1],
                                                                  self._ts)
            else:
                self._list_tpts.append(TrackPoint(last_det._mcc,last_det._x,last_det._y,0,0))
                self._expected_TP = self._list_tpts[-1]

            self._selection_gate = calc_gate(self._expected_TP,
                                             self._vx_max * self._ts,
                                             self._vy_max * self._ts)
        else:
            print("List of detections is empty")
            return None
        return self._list_tpts[-1].mcc

    def get_predicted_gate(self):
        gate = {'gate':self._selection_gate,'center':self._expected_TP}
        return gate


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

def calc_state_fading_mem_fltr(x_n,x_m,Phi,K,H,mcc):
    z = np.matmul(H,x_m)
    x_mod = np.matmul(Phi,x_n)
    x = x_mod + np.matmul(K,z - x_m)
    next_point = TrackPoint(mcc,x[0],x[2],x[1],x[3])
    return next_point

def my_fltr_range(lst_det, mcc_start, mcc_end):
    for i_mcc in range(mcc_start, mcc_end):
        dets = lst_det.get_array_detections_selected(mcc = (i_mcc,i_mcc))
        no_det = np.size(dets["mcc"])

        for i_det in range(0, no_det):
            print("Mcc", dets["mcc"][i_det],
                  "Detection:", i_det,
                  "range", dets["range"][i_det],
                  "velocity", dets["velocity"][i_det],
                  "azimuth", 180 / np.pi * dets["azimuth"][i_det],
                  "number of detections", no_det)
