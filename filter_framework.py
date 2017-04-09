#!/usr/bin/env python

import data_containers as dc
import tracking_filters as tf
import radar_plots as rplt
import numpy as np


def main(conf_data):
    # Load Data from .mat files
    if conf_data["filename_LeftRadar"]:
        l = []
        l.append(conf_data["path_data_folder"])
        l.append(conf_data["filename_LeftRadar"])
        leftradar_path = ''.join(l)

        lst_det_left = dc.detection_set()
        lst_det_left.AppendFromFileMat(leftradar_path, True, conf_data["EGO_car_width"])
        mcc_interval_left = lst_det_left.GetMCCInterval()
        print("MCC Left starts at: ", mcc_interval_left[0],
              "and ends at: ", mcc_interval_left[1])

    if conf_data["filename_RightRadar"]:
        l = []
        l.append(conf_data["path_data_folder"])
        l.append(conf_data["filename_RightRadar"])
        rightradar_path = ''.join(l)

        lst_det_right = dc.detection_set()
        lst_det_right.AppendFromFileMat(rightradar_path, False, conf_data["EGO_car_width"])
        mcc_interval_right = lst_det_right.GetMCCInterval()
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
        mcc_end = mcc_interval_right[1]
    print("MCC starts at: ", mcc_start, "MCC ends at: ", mcc_end)
    mcc_step = 1

    ############ Filtering loop
    i_prev = mcc_start
    for i in range(mcc_start, mcc_end, mcc_step):  # number of frames frames
        if conf_data["output_folder"]:
            fname_det = '_tmp%08d.png' % i
            l = []
            l.append(conf_data["output_folder"])
            l.append(fname_det)
            output_path = ''.join(l)
        else:
            output_path = None

        # Here is a place where a filter and/or plot module belong(s)
        # tf.g_h_constant_fltr(lst_det_left,lst_det_right,conf_data["beams_tp"],i_prev,i,output_path)
        # rplt.GridPlot_hist(lst_det_left,lst_det_right,conf_data["beams_tp"],i_prev,i,output_path)

        i_prev = i


if __name__ == "__main__":
    conf_data = dc.parse_CMDLine("./analysis.cnf")
    if conf_data:
        main(conf_data)
