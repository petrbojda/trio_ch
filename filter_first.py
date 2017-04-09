#!/usr/bin/env python

import data_containers as dc
import tracking_filters as tf
import radar_plots as rplt
import numpy as np


def main(conf_data):
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
    mcc_interval = lst_det.get_mcc_interval()
    print("MCC starts at: ", mcc_interval[0],
          "and ends at: ", mcc_interval[1])
    # mcc_start = mcc_interval[0]
    # mcc_end = mcc_interval[1]

    mcc_start = mcc_interval[0]
    mcc_end = mcc_interval[0] + 10

    mcc_history_depth = 3

    ############ Filtering loop
    i_prev = mcc_start
    for i in range(mcc_start + mcc_history_depth, mcc_end):  # number of frames frames
        print("A new burst starts here")
        tf.my_fltr_range(lst_det, i_prev, i)
        # rplt.GridPlot_hist(lst_det_left,lst_det_right,conf_data["beams_tp"],i_prev,i+1,None)

        i_prev += 1


if __name__ == "__main__":
    conf_data = dc.parse_CMDLine("./analysis.cnf")
    if conf_data:
        main(conf_data)
