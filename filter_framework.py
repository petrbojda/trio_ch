#!/usr/bin/env python

import data_containers as dc
import track_management as tm
import radar_plots as rplt
import numpy as np


def main(config_data):
    selection = {"beam_tp": config_data["beams_tp"],
                 "mcc_tp": None, "x_tp": None, "y_tp": None,
                 "rng_tp": None, "vel_tp": (-5, 5), "az_tp": None,
                 "trackID_tp": None, }
    # Structure 'selection' constrains  data to use as input to the tracker.
    # Specific 'mcc', 'azimuth', 'range' ... etc can be specified here to block
    # unwanted data to enter.

    track_LR = tm.TrackManager()
    track_RR = tm.TrackManager()

    # Load Data from .mat files
    if config_data["filename_LeftRadar"]:
        l = []
        l.append(config_data["path_data_folder"])
        l.append(config_data["filename_LeftRadar"])
        leftradar_path = ''.join(l)     

        lst_det_LR = dc.DetectionList()
        lst_det_LR.append_data_from_m_file(leftradar_path, True, config_data["EGO_car_width"])
        mcc_interval_LR = lst_det_LR.get_mcc_interval()
        print("filter_framework: MCC Left starts at: ", mcc_interval_LR[0],
              "and ends at: ", mcc_interval_LR[1])

    if config_data["filename_RightRadar"]:
        l = []
        l.append(config_data["path_data_folder"])
        l.append(config_data["filename_RightRadar"])
        rightradar_path = ''.join(l)

        lst_det_RR = dc.DetectionList()
        lst_det_RR.append_data_from_m_file(rightradar_path, False, config_data["EGO_car_width"])
        mcc_interval_RR = lst_det_RR.get_mcc_interval()
        print("filter_framework: MCC Right starts at: ", mcc_interval_RR[0], "and ends at: ", mcc_interval_RR[1])

    # Calculate valid mcc interval for detections to be presented
    if config_data["filename_LeftRadar"] and config_data["filename_RightRadar"]:
        mcc_start = min(mcc_interval_LR[0], mcc_interval_RR[0])
        mcc_end = max(mcc_interval_LR[1], mcc_interval_RR[1])
    elif config_data["filename_LeftRadar"]:
        mcc_start = mcc_interval_LR[0]
        mcc_end = mcc_interval_LR[1]
    else:
        mcc_start = mcc_interval_RR[0]
        # mcc_end = mcc_interval_RR[1]
        mcc_end = mcc_start + 100
    print("filter_framework: MCC starts at: ", mcc_start, "MCC ends at: ", mcc_end)
    mcc_step = 1

    #----------------- Filtering loop
    i_prev = mcc_start
    for i in range(mcc_start, mcc_end, mcc_step):  # number of frames

        selection["mcc_tp"] = (i_prev, i)
        #-------------- Left radar filter
        if lst_det_LR:
            lst_det_per_loop_cycle_LR = lst_det_LR.get_lst_detections_selected(selection=selection)
            #   TODO: Is it correct to assign this for every iteration? Potential to write more effective code.
            if lst_det_per_loop_cycle_LR:
                print('filter_framework: Number of detections for a LR mcc ', i, 'is: ', len(lst_det_per_loop_cycle_LR))
                track_LR.new_detection(lst_det_per_loop_cycle_LR)
            else:
                print('filter_framework: There is no detection for current LR mcc ',i)

        i_prev = i + 1
        # TODO: This line is redundant if only one mcc is being processed per loop cycle. However if mcc_step is different than 1, it might be good to keep it here.


#     TODO: graphical representation of the results

if __name__ == "__main__":
    config_data = dc.parse_CMDLine("./analysis.cnf")
    if config_data:
        main(config_data)
