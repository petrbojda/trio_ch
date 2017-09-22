#!/usr/bin/env python

import data_containers as dc
import track_management as tm
import radar_plots as rplt
import numpy as np


def main(conf_data):

    selection = {"beam_tp":conf_data["beams_tp"],
                 "mcc_tp":None, "x_tp":None, "y_tp":None,
                 "rng_tp":None, "vel_tp":(-5,5), "az_tp":None,
                 "trackID_tp":None,}

    tracks_left = tm.TrackManager()
    tracks_right = tm.TrackManager()

    # Load Data from .mat files
    if conf_data["filename_LeftRadar"]:
        l = []
        l.append(conf_data["path_data_folder"])
        l.append(conf_data["filename_LeftRadar"])
        leftradar_path = ''.join(l)

        lst_det_left = dc.DetectionList()
        lst_det_left.append_from_m_file(leftradar_path, True, conf_data["EGO_car_width"])
        mcc_interval_left = lst_det_left.get_mcc_interval()
        print("MCC Left starts at: ", mcc_interval_left[0],
              "and ends at: ", mcc_interval_left[1])

    if conf_data["filename_RightRadar"]:
        l = []
        l.append(conf_data["path_data_folder"])
        l.append(conf_data["filename_RightRadar"])
        rightradar_path = ''.join(l)

        lst_det_right = dc.DetectionList()
        lst_det_right.append_from_m_file(rightradar_path, False, conf_data["EGO_car_width"])
        mcc_interval_right = lst_det_right.get_mcc_interval()
        print("MCC Right starts at: ", mcc_interval_right[0], "and ends at: ", mcc_interval_right[1])

    # Calculate valid mcc interval for detections to be presented
    if conf_data["filename_LeftRadar"] and conf_data["filename_RightRadar"]:
        mcc_start = min(mcc_interval_left[0], mcc_interval_right[0])
        mcc_end = max(mcc_interval_left[1], mcc_interval_right[1])
    elif conf_data["filename_LeftRadar"]:
        mcc_start = mcc_interval_left[0]
        mcc_end = mcc_interval_left[1]
    else:
        mcc_start = mcc_interval_right[0]
        # mcc_end = mcc_interval_right[1]
        mcc_end = mcc_start + 100
    print("MCC starts at: ", mcc_start, "MCC ends at: ", mcc_end)
    mcc_step = 1

    ############ Filtering loop
    i_prev = mcc_start
    for i in range(mcc_start, mcc_end, mcc_step):  # number of frames

        selection["mcc_tp"] = (i_prev,i)
        #################### Left radar filter
        if lst_det_left:
            LR_data = lst_det_left.get_array_detections_selected(selection=selection)
            if LR_data["mcc"].any():
                LR_data_exists = True
                number_of_dets_left = np.size(LR_data["mcc"])
                tracks_left.test_detections(i,number_of_dets_left,LR_data)

        i_prev = i
#     TODO: graphical representation of the results

if __name__ == "__main__":
    conf_data = dc.parse_CMDLine("./analysis.cnf")
    if conf_data:
        main(conf_data)
