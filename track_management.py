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

    def append_detection_to_track(self, trackID, detection):
        for elem in self:
            if elem.trackID == trackID:
                elem.append_detection(detection)

    def new_detection(self,mcc,noDet,lst_detections):
        print ("track_mgmt: Detections to assign, mcc:",mcc,"with NoDet",noDet,"Agreed on:", len(lst_detections))
        print ("track_mgmt: Detections to assign", lst_detections)
        for i1 in range(0, noDet):
            if self._n_of_Tracks[-1]:
                for elem in self:
                    if self.test_det_in_gate(elem.get_prediction(),lst_detections[i1]):
                        selTrackID = elem.trackID
                        print ("Detection at mcc",mcc ,"is assigned to:",selTrackID)
                        self.append_detection_to_track(selTrackID,lst_detections[i1])

                lst_detections[i1]._trackID = 999   # more reasonable value will be assigned in the future
            #     TODO: assign list of trackIDs to mark the case when a detection is assigned to more than just one track

            if lst_detections[i1]._trackID == 0:
                print ("track_mgmt: not assigned detections are in a list:",self._lst_not_assigned_detections)
                self._lst_not_assigned_detections.append_detection(lst_detections[i1])

                self.create_new_track()
                print ("track_mgmt: A new track is created. ID:",self._n_of_Tracks)


