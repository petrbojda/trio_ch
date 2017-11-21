import numpy as np
import data_containers as dc

class TrackManager(list):

    def __init__(self, gate = None, flt_type='G-H', Tsampling=50.0e-3):
        super().__init__()
        self._Tsampling = Tsampling
        self._gate = gate
        self._flt_type = flt_type
        self._n_of_Tracks = np.array([0])
        self._lst_not_assigned_detections = dc.UnAssignedDetectionList(
                                                            self._Tsampling,
                                                            self._gate)
        if gate is None:
            self._gate = {'x': 3, 'y': 1, 'dx': 0.65, 'dy': 0.3}
        else:
            self._gate = gate

    def create_new_track(self):
        self._n_of_Tracks = np.append(self._n_of_Tracks,self._n_of_Tracks[-1]+1)
        self.append(dc.Track(self._n_of_Tracks[-1]))

    def new_detection(self,lst_detections):
        print ("track_mgmt: Detections to assign, mcc:",lst_detections[0].get_mcc(),"Agreed on:", len(lst_detections))
        print ("track_mgmt: Detections to assign", lst_detections)
        aim=[]
        for det in lst_detections:
            if self:
                for elem in self:
                    aim.append(elem.test_det_in_gate(det))
                if max(aim):
                    self[aim.index(max(aim))].append(det)
                    unassigned = False
                else:
                    print("track_mgmt: Some track exists but detection doesn't fit in.")
                    unassigned = True
            else:
                unassigned = True
            if unassigned:
                print("track_mgmt: No track started yet!")
                # test unassigned detections
                if self._lst_not_assigned_detections.new_detection(det):
                    # a new track is started with a detection "det"
                    self.create_new_track()
                else:
                    # a new detestion "det" is stored in a list of unassigned detections
                    self._lst_not_assigned_detections.append(det)

        aim.clear()




