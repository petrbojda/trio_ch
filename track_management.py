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

    def create_new_track(self,track):
        self._n_of_Tracks = np.append(self._n_of_Tracks,self._n_of_Tracks[-1]+1)
        self.append(track)

    def new_detection(self,lst_detections):
        print ("track_mgmt: Detections to assign, mcc:",lst_detections[0].get_mcc(),"Agreed on:", len(lst_detections))
        print ("track_mgmt: Detections to assign", lst_detections)
        aim=[]
        self._lst_not_assigned_detections.remove_detections([0, lst_detections[0].get_mcc() - 10])
        for det in lst_detections:
            if self:
                for elem in self:
                    aim.append(elem.test_detection_in_gate(det))
                print("track_mgmt: aim is",aim)
                if max(aim):
                    print("track_mgmt: max(aim) is",max(aim), "number of tracks",
                          len(self), "pointing at",aim.index(max(aim)),"Object:",
                          self[aim.index(max(aim))])
                    self[aim.index(max(aim))].append_detection(det)
                    print("track_mgmt: The detection was assigned to a track at", aim.index(max(aim)))
                    unassigned = False
                else:
                    print("track_mgmt: Some track exists but detection doesn't fit in. "
                          "Number of tracks in list:",len(self))
                    unassigned = True
            else:
                unassigned = True
            aim.clear()
            if unassigned:
                print("track_mgmt: Processing detection at mcc:",det._mcc,"No track started now!")
                # test unassigned detections
                newly_formed_track = self._lst_not_assigned_detections.new_detection(det)
                if newly_formed_track:
                    # a new track is started with a detection "det"
                    self.create_new_track(newly_formed_track)
                    print("track_mgmt: A new track was created. Currently ",len(self), "tracks is in the list.")

    def port_data(self,requested_data_type):
        if requested_data_type == "track_init":
            if self:
                print("track_mgmt: porting track_init data. Number of tracks: ", len(self), "The last track ported.")
                return self._lst_not_assigned_detections, self[-1]
            else:
                print("track_mgmt: porting track_init data. No track in the list, None track ported.")
                return self._lst_not_assigned_detections, None



