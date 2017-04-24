import numpy as np
import data_containers as dc


class TrackPoint(object):
    def __init__(self,mcc,x,y):
        self.mcc = mcc
        self.x = x
        self.y = y

class Tracker(object):
    """ Class to encapsulate an estimator - tracker is defined here"""
    def __init__(self,start_mcc, **kwarg):

        if 'beta' in kwarg:
            beta = float(kwarg['beta'])
        else:
            beta = 0.5

        if 'ts' in kwarg:
            ts = float(kwarg['ts'])
        else:
            ts = 0.05

        if 'v_max' in kwarg:
            v_max = float(kwarg['v_max'])
        else:
            v_max = 138.0

        if 'vc_max' in kwarg:
            vc_max = float(kwarg['vc_max'])
        else:
            vc_max = 49.0

        #Qtmp = [(dT^2)/3 dT/2; dT/2 1]*dT*StateVar
        #Rtmp = 1*ObserVar
        #xtmp = [0;0]
        #Ptmp = Qtmp*10
        g = 1-beta**2
        h = (1-beta)**2
        K = np.array([g,h])
        K = K.reshape(2,1)
        self._ts = ts
        self._list_detections = dc.DetectionList()
        self._list_trackpoints = []
        self._start_mcc = start_mcc
        self._mx_K = np.kron(np.eye(2),K)
        self._mx_phi = np.array([[1,ts,0,0],[0,1,0,0],[0,0,1,ts],[0,0,0,1]])
        H = np.array([0,1])
        self._mx_H = np.kron(np.eye(2),H)

    def append_detection(self,detection):
        self._list_detections.append(detection)

    def append_detection_list(self,radar_data_list, **kwarg):
        self._list_detections.append_list_detections_selection(radar_data_list, **kwarg)

    def update_track(self):
        if self._list_detections:
            mcc_last_det = max([elem.mcc for elem in self._list_detections])
            print(mcc_last_det)
        else:
            print("List of detections is empty")

        if self._list_trackpoints:
            mcc_last_track = max([elem.mcc for elem in self._list_trackpoints])
            print(mcc_last_track)
        else:
            print("Track is empty")

    def get_predicted_window(self,mcc):


        return(gate)







def g_h_constant_fltr():
    return ()


def g_h_benedbordner_fltr():
    return ()


def g_h_critdamp_fltr():
    return ()


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
