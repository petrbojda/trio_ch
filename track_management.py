import numpy as np
import itertools as it
import data_containers as dc

class TrackManager(list):

    def __init__(self, flt_type="G-H", Tsampling=50.0e-3, gate_dx=5, gate_dy=5):
        super().__init__()

        self._Tsampling = Tsampling
        self._gate_dx = gate_dx
        self._gate_dy = gate_dy

        self._flt_type = flt_type
        self._n_of_Tracks = np.array([0])
        self._lst_not_assigned_detections = []


    def create_new_track(self):
        self._n_of_Tracks = np.append(self._n_of_Tracks,self._n_of_Tracks[-1]+1)
        self.append(dc.Track(self._n_of_Tracks[-1]))

    def append_detection_to_track(self, trackID, detection):
        for elem in self:
            if (elem.trackID == trackID):
                elem.append_detection(detection)



    def test_det_in_gate(self,gate,detection):
        is_in_x = (gate.x-self._gate_dx < detection['x']) and (gate.x+self._gate_dx > detection['x'])
        is_in_y = (gate.y-self._gate_dx < detection['y']) and (gate.y+self._gate_dx > detection['y'])

        return is_in_x and is_in_y



    def test_detections(self,mcc,noDet,lst_detections):
        print ("Detections to assign, mcc:",mcc,"with NoDet",noDet)
        print ("Detections to assign", lst_detections)
        for i1 in range(0, noDet):
            # detection = {'x':lst_detections['x'][i1],'y':lst_detections['y'][i1],'beam':lst_detections['beam'][i1],
            #              'mcc':lst_detections['mcc'][i1],'trackID':lst_detections['trackID'][i1],
            #              'Razimuth': lst_detections['Razimuth'][i1], 'Rvelocity': lst_detections['Rvelocity'][i1]}
            if self._n_of_Tracks[-1]:
                for elem in self:
                    if self.test_det_in_gate(elem.get_prediction(),lst_detections[i1]):
                        selTrackID = elem.trackID
                        print ("Detection at mcc",mcc ,"is assigned to:",selTrackID)
                        self.append_detection_to_track(selTrackID,lst_detections[i1])

                lst_detections[i1]._trackID = 999   # more reasonable value will be assigned in the future
            #     TODO: assign list of trackIDs to mark the case when a detection is assigned to more than just one tracks

            if lst_detections[i1]._trackID == 0:
                print ("not assigned detections are in a list:",self._lst_not_assigned_detections)
                self._lst_not_assigned_detections.append(detection)
                # TODO: rewrite '_lst_not_assigned_detections' to a structure (class) with a combination of linear predictions
                print ("Detection at mcc",mcc ,"is not assigned to an existing track")

            if len(self._lst_not_assigned_detections['mcc']) >= 2:
                for elem1, elem2 in it.combinations(self._lst_not_assigned_detections['x'], 2):
                    self.calc_two_point_projection(elem1,elem2)
                    # TODO: finish comparison of elements, rewrite method 'calc_two_point_projection'
                    self.create_new_track()
                    print ("A new track is created. ID:",self._n_of_Tracks)


        # TODO: write a function to test third detection in a row to fit a prediction based on detection at mcc-2 and mcc-1


        # TODO: '_lst_not_assigned_detections' is of a special class inherited from the DetectionList. An array of 3-point predictions will be its parameter updated with every new appended detection
