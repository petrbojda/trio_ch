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

    ##  Select an area to track (In terms of any parameter - beam, x, y, vel, ...)
    lst_det_s = dc.DetectionList()
    lst_det_s.extend_with_selection(lst_det, selection=selection)

    ##  Define the range of MCCs to do the filtering
    mcc_interval = lst_det_s.get_mcc_interval()
    mcc_start = mcc_interval[0] +100
    mcc_end = mcc_start + 15

    trackers = []       # List of trackers - empty for now

    ##  Filtering loop
    trk_counter = 0
    for i1 in range(mcc_start, mcc_end):    # Iterate through the set of MCCs within a range
        print("A new burst starts here, mcc:", i1)
        lst_det_i=[elem for elem in lst_det_s if elem._mcc == i1] # Get list of detections for MCC
        print("detections in this run",len(lst_det_i))
        no_det = len(lst_det_i)             # Number of detections in the list
        for i2 in list(range(0,no_det)):    # Iterate through the set of detections in a single MCC
            print("detection at x:",lst_det_i[i2]._x,"y:",lst_det_i[i2]._y,"vel:",lst_det_i[i2]._vel)
            print("beam of detection:",lst_det_i[i2]._beam)
            if trk_counter == 0:
                trackers.append(tf.Tracker(i1))
                trackers[0].append_detection(lst_det_i[i2])
                trk_counter += 1
            else:
                for i3 in list(range(0,trk_counter)):   # Iterate through trackers to assign the detection
                    gate = trackers[i3].get_predicted_gate()
                    print("Tracker number:",i3,"selection gate",gate)



    print("counter",trk_counter)
    print("trackers",len(trackers))
    print("detections in the selection",len(lst_det_s))


if __name__ == "__main__":
    conf_data = dc.parse_CMDLine("./analysis.cnf")
    if conf_data:
        main(conf_data)
