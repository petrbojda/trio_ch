import numpy as np
import logging
import data_containers as dc
import radar_plots as rp

class TrackManager(list):

    def __init__(self, gate = None, tracker_type={'filter_type': 'kalman_filter', 'dim_x': 4, 'dim_z': 2}, Tsampling=50.0e-3):
        super().__init__()
        if gate is None:
            self._gate = dc.Gate(beam=[], x=0, y=0, diffx=0.5, diffy=0.3, dx=0, dy=0, diffdx=0.65, diffdy=0.3,
            rvelocity=0, d_rvelocity = 0, razimuth=0, d_razimuth=0, rrange=0, d_rrange=0)
            #{'x': 3, 'y': 1, 'dx': 0.65, 'dy': 0.3}
        else:
            self._gate = gate
        self._Tsampling = Tsampling
        self._tracker_type = tracker_type
        self._n_of_Tracks = np.array([0])
        logging.getLogger(__name__).debug("__init__: A new track manager will be created with a gate:")
        logging.getLogger(__name__).debug("__init__: \t \t %s", self._gate)
        logging.getLogger(__name__).debug("__init__: \t \t tracker_type %s,",  self._tracker_type)
        logging.getLogger(__name__).debug("__init__: \t \t Tsampl %s, number of tracks %s",
                                                                self._Tsampling, self._n_of_Tracks)
        self._lst_not_assigned_detections = dc.UnAssignedDetectionList(self._Tsampling, self._gate)
        logging.getLogger(__name__).debug("__init__: \t just created, number of unassigned dets %s",
                                                            len(self._lst_not_assigned_detections))


    def append_track(self,track):
        """ Appends an existing track to the list of tracks, a new tracking filter is also created
        alongside the track and is assigned to it.

        :param track:
        :type track: Track
        """
        self._n_of_Tracks = np.append(self._n_of_Tracks,self._n_of_Tracks[-1]+1)
        self.append(track)

    def new_detections(self,lst_detections):
        logger = logging.getLogger(__name__)
        logger.debug("new_detections: Tested will be new %s detections",
                     len(lst_detections))
        logger.debug("new_detections: \t \t with MCCs from %s to %s",
                    lst_detections.get_mcc_interval()[0],
                    lst_detections.get_mcc_interval()[1])

        aim=[]
        logger.debug("new_detections: In a _lst_not_assigned_detections is %s detections.",
                     len(self._lst_not_assigned_detections))
        self._lst_not_assigned_detections.remove_detections_by_mcc([0, lst_detections[0].get_mcc() - 10])
        logger.debug("new_detections: \t after 10 mccs removal: %s detections.",
                     len(self._lst_not_assigned_detections))

        # track update loop - each new detection as assigned to an existing track
        # triggers the update cycle of the track
        for det in lst_detections:
            if self:
                logger.debug("new_detections, tracks exist: Currently some tracks exist in a list. Will be scrutinized. Number of tracks: %d",
                              len(self))
                for elem in self:
                    if elem._active and elem._last_update != det._mcc:
                        aim.append(elem.test_detection_in_gate(det))
                        logger.debug("new_detections, tracks exist: The vector of all distances from each track's gate center, the aim, is: %5.3f", aim[-1])
                    else:
                        logger.debug("new_detections, tracks exist: none of tracks is active or they have been updated in this mcc")
                        aim.append(0)
                if aim[-1]:
                    logger.debug("new_detections, tracks exist: max(aim) is %5.3f pointing at the track number: %d",max(aim),aim.index(max(aim)))
                    self[aim.index(max(aim))].append_detection(det)
                    logger.debug("new_detections, tracks exist: The detection was assigned to a track number: %d", aim.index(max(aim)))
                    self[aim.index(max(aim))].update_tracker()
                    logger.debug("new_detections, tracks exist: track updated")
                    unassigned = False
                else:
                    logger.debug("new_detections, tracks exist: currently tested detection doesn't fit in.")
                    unassigned = True
            else:
                unassigned = True
            aim.clear()

            if unassigned:
                # The detection 'det' was not assigned to an existing track, will be passed to
                # the list of unassigned detections.
                logger.debug("new_detections, no track exists yet. Processing detection at mcc: %d" ,det._mcc)
                # test unassigned detections
                newly_formed_track = self._lst_not_assigned_detections.new_detection(det)
                if newly_formed_track:
                    title = 'A new track created at {0}. Incomming {1} new detections, {2} unassigned '.format(det._mcc,
                                                                                                               len(lst_detections),
                                                                                                               len(self._lst_not_assigned_detections)
                                                                                                               )
                    rp.static_track_init(3,
                                         lst_detections,
                                         self._lst_not_assigned_detections,
                                         det,
                                         newly_formed_track['best_fit_gate'],
                                         newly_formed_track['new_track'].get_array_trackpoints(),
                                         title)

                     # a new track is started with a detection "det"
                    self.append_track(newly_formed_track['new_track'])
                    logger.debug("new_detections, no tracks: A new track was created. Currently %d tracks is in the list.",len(self))
                    self[-1].init_tracker(type=self._tracker_type['filter_type'],
                                          dim_x=self._tracker_type['dim_x'],
                                          dim_z=self._tracker_type['dim_z'],
                                          dt=self._Tsampling,
                                          init_x=self[-1][0].get_xy_array())
                    logger.debug("new_detections, no tracks: tracker initialized for the new track: %s",self[-1]._tracker)
                    self[-1].start_tracker()
                    logger.debug("new_detections, no tracks: new track's first 3 points: %s",self[-1])
                else:
                    title = 'No track created at {0}. Incomming {1} new detections, {2} unassigned '.format(det._mcc,
                                                                                                            len(lst_detections),
                                                                                                            len(self._lst_not_assigned_detections)
                                                                                                            )
                    rp.static_track_init(3,
                                         lst_detections,
                                         self._lst_not_assigned_detections,
                                         None,
                                         None,
                                         None,
                                         title)
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



