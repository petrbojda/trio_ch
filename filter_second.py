#!/usr/bin/env python

import data_containers as dc
import tracking_filters as tf
import radar_plots as rplt
import numpy as np


def main(conf_data):

    selection = {"beam_tp":conf_data["beams_tp"],
                 "mcc_tp":None, "x_tp":None, "y_tp":None,
                 "rng_tp":None, "vel_tp":None, "az_tp":None}

    # Load Data from .mat files
    if conf_data["radar_tp"] == "L":
        filename_Radar = conf_data["filename_LeftRadar"]
    elif conf_data["radar_tp"] == "R":
        filename_Radar = conf_data["filename_RightRadar"]
    else:
        print("Both radars will not be tracked together")
        quit()

    l = []
    l.append(conf_data["path_data_folder"])
    l.append(filename_Radar)
    radar_path = ''.join(l)

    lst_det = dc.DetectionList()
    lst_det.append_from_m_file(radar_path, True, conf_data["EGO_car_width"])

    ############  Select an arrea to track (filter out unwanted detections)
    lst_det_s = dc.DetectionList()
    lst_det_s.append_list_detections_selection(lst_det, selection=selection)

    mcc_interval = lst_det_s.get_mcc_interval()
    mcc_start = mcc_interval[0] +100
    mcc_end = mcc_start + 15

    trackers = []

    ############ Filtering loop
    trk_counter = 0
    for i1 in range(mcc_start, mcc_end):
        print("A new burst starts here, mcc:", i1)
        lst_det_i = [elem for elem in lst_det_s if elem._mcc == i1]
        print("detections in this run",len(lst_det_i))
        no_det = len(lst_det_i)
        for i2 in list(range(0,no_det)):
            print("counter",trk_counter)
            print("trackers",len(trackers))
            trackers.append(tf.Tracker(i1))
            trackers[trk_counter].update_track()
            trk_counter += 1
    print("counter",trk_counter)
    print("trackers",len(trackers))
    print("detections in selection",len(lst_det_s))



if __name__ == "__main__":
    conf_data = dc.parse_CMDLine("./analysis.cnf")
    if conf_data:
        main(conf_data)
