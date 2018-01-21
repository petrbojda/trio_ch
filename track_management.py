import numpy as np
import logging
import data_containers as dc

class TrackManager(list):

    def __init__(self, gate = None, tracker_type={'filter_type': 'kalman_filter', 'dim_x': 4, 'dim_z': 2}, Tsampling=50.0e-3):
        super().__init__()
        self._Tsampling = Tsampling
        self._gate = gate
        self._tracker_type = tracker_type
        self._n_of_Tracks = np.array([0])
        self._lst_not_assigned_detections = dc.UnAssignedDetectionList(
                                                            self._Tsampling,
                                                            self._gate)
        if gate is None:
            self._gate = {'x': 3, 'y': 1, 'dx': 0.65, 'dy': 0.3}
        else:
            self._gate = gate

    def append_track(self,track):
        """ A new track is being appended to the list of tracks, a new tracking filter is also created
        alongside the track and is assigned to it.

        :param track:
        :type track: Track
        """
        self._n_of_Tracks = np.append(self._n_of_Tracks,self._n_of_Tracks[-1]+1)
        self.append(track)

    def new_detections(self,lst_detections):
        logger = logging.getLogger(__name__)
        logger.debug("TrackManager.new_detections: Tested new %s detections",
                     len(lst_detections))
        logger.debug("TrackManager.new_detections: \t \t with MCCs from %s to %s",
                    lst_detections.get_mcc_interval()[0],
                    lst_detections.get_mcc_interval()[1])

        aim=[]
        logger.debug("TrackManager.new_detections: In a _lst_not_assigned_detections is %s detections.",
                     len(self._lst_not_assigned_detections))
        self._lst_not_assigned_detections.remove_detections([0, lst_detections[0].get_mcc() - 10])
        logger.debug("TrackManager.new_detections: \t after 10 mccs removal: %s detections.",
                     len(self._lst_not_assigned_detections))

        # track update loop - each new detection as assigned to an existing track
        # triggers the update cycle of the track
        for det in lst_detections:
            if self:
                print("track_mgmt: Currently some tracks exist in a list. Will be scrutinized. Number of tracks:",
                          len(self))
                for elem in self:
                    if elem._active and elem._last_update != det._mcc:
                        aim.append(elem.test_detection_in_gate(det))
                print("track_mgmt: The vector of all distances from each track's gate center, the aim, is:",aim)
                if aim:
                    print("track_mgmt: max(aim) is",max(aim),"pointing at the track number:",aim.index(max(aim)))
                    self[aim.index(max(aim))].append_detection(det)
                    print("track_mgmt: The detection was assigned to a track number:", aim.index(max(aim)))
                    self[aim.index(max(aim))].update_tracker()
                    print("track_mgmt:track updated")
                    unassigned = False
                else:
                    print("track_mgmt: Some track exists but the detection doesn't fit in.")
                    unassigned = True
            else:
                unassigned = True
            aim.clear()

            if unassigned:
                # The detection 'det' was not assigned to an existing track, will be passed to
                # the list of unassigned detections.
                print("track_mgmt: Processing detection at mcc:",det._mcc,"No track started now!")
                # test unassigned detections
                newly_formed_track = self._lst_not_assigned_detections.new_detection(det)
                if newly_formed_track:
                    # a new track is started with a detection "det"
                    self.append_track(newly_formed_track)
                    print("track_mgmt: A new track was created. Currently ",len(self), "tracks is in the list.")
                    self[-1].init_tracker(type=self._tracker_type['filter_type'],
                                          dim_x=self._tracker_type['dim_x'],
                                          dim_z=self._tracker_type['dim_z'],
                                          dt=self._Tsampling,
                                          init_x=self[-1][0].get_xy_array())
                    print("track_mgmt: tracker initialized for the new track:",self[-1]._tracker)
                    self[-1].start_tracker()
                    print("track_mgmt: new track's first 3 points:",self[-1])
            else:
                # TODO: tracker update to finish here
                # The detection 'det' was assigned to an existing track and its appropriate tracker
                # needs to update.
                pass


    def port_data(self,requested_data):
        if requested_data == "track_init":
            if self:
                print("track_mgmt: porting track_init data. Number of tracks: ", len(self), "The last track ported.")
                return self._lst_not_assigned_detections, self[-1]
            else:
                print("track_mgmt: porting track_init data. No track in the list, None track ported.")
                return self._lst_not_assigned_detections, None
        if requested_data == "tracks_array":
            if self:
                list_of_tracks = []
                print("track_mgmt: porting tracks_aray data. Number of tracks: ", len(self), "The last track ported.")
                for elem in self:
                    list_of_tracks.append(elem.get_array_trackpoints())
                return list_of_tracks

            else:
                print("track_mgmt: porting tracks_aray data. No track in the list, None ported.")
                return None

    def predict(self,mcc):
        for elem in self:
            if elem._last_update < mcc-10:
                elem.deactivate()
            else:
                elem.predict()



