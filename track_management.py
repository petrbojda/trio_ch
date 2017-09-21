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
        self._lst_not_assigned_detections = dc.DetectionList()


    def _create_new_track(self):
        self._n_of_Tracks = np.append(self._n_of_Tracks,self._n_of_Tracks[-1]+1)
        self.append(dc.Track(self._n_of_Tracks[-1]))

    def append_detection_to_track(self, trackID, detection):
        track_sel = [elem for elem in self if (elem.trackID == trackID)]
        track_sel.append_point_from_detection(detection)


    def _test_det_in_gate(self,gate,detection):

        is_in_x = (gate.x-self._gate_dx < detection._x) and (gate.x+self._gate_dx > detection._x)
        is_in_y = (gate.y-self._gate_dx < detection._y) and (gate.y+self._gate_dx > detection._y)

        return is_in_x and is_in_y

    def calc_two_point_projection(self,start_detection,end_detection):
        projected_point = dc.TrackPoint(end_detection._mcc + 1,
                                        0,0,
                                        0,0)

        projected_point.x = 2*end_detection._x - start_detection._x
        projected_point.y = 2*end_detection._y - start_detection._y
        projected_point.dx = (end_detection._x - start_detection._x) / self._Tsampling
        projected_point.dy = (end_detection._y - start_detection._y) / self._Tsampling
        return projected_point

    def test_detections(self,mcc,noDet,lst_detections):

        for i1 in range(0, noDet):
            if self._n_of_Tracks[-1]:
                selTrackID = [elem.trackID for elem in self if self._test_det_in_gate(elem.get_prediction(),lst_detections[i1])]
                print ("Detection at mcc",mcc ,"is assigned to:",selTrackID)
                self.append_detection_to_track(selTrackID,lst_detections[i1])
                lst_detections[i1]._trackID = selTrackID
            else:
                self._lst_not_assigned_detections.append(lst_detections[i1])
                print ("Detection at mcc",mcc ,"is not assigned to an existing track")
                self._create_new_track()
                print ("A new track is created. ID:",self._n_of_Tracks)

            # if lst_detections[i1]._trackID == 0:
            #     self._lst_not_assigned_detections.append(lst_detections[i1])
            #     print ("Detection at mcc",mcc ,"is not assigned to an existing track")
            #     self._create_new_track()
            #     print ("A new track is created. ID:",self._n_of_Tracks)
            # else: pass

