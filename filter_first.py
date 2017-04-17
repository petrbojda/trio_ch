#!/usr/bin/env python

import data_containers as dc
import tracking_filters as tf
import radar_plots as rplt
import numpy as np


def main(conf_data):

    selection = {"beam_tp":conf_data["beams_tp"],
                 "mcc_tp":None, "x_tp":(10,55), "y_tp":(-30,55),
                 "rng_tp":(25,30), "vel_tp":None, "az_tp":None}

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

    lst_det_s = dc.DetectionList()
    lst_det_s.append_list_detections_selection(lst_det, selection=selection)

    mcc_interval = lst_det_s.get_mcc_interval()
    mcc_start = mcc_interval[0]
    mcc_end = mcc_interval[0] + 15

    mcc_history_depth = 3

    ############ Filtering loop
    i_prev = mcc_start
    for i in range(mcc_start + mcc_history_depth, mcc_end):  # number of frames frames
        print("A new burst starts here, mcc:", i_prev, i)
        tf.my_fltr_range(lst_det_s, i_prev, i)
        # rplt.GridPlot_hist(lst_det_left,lst_det_right,conf_data["beams_tp"],i_prev,i+1,None)
        i_prev += 1

    if conf_data["output_folder"]:
        l = []
        l.append(conf_data["output_folder"])
        fname_det = '_tmp%s.png' % conf_data["scenario"]
        l.append(fname_det)
        output_path = ''.join(l)
    else:
        output_path = None

    selection_pl = {"beam_tp":conf_data["beams_tp"],
                 "mcc_tp":None, "x_tp":None, "y_tp":None,
                 "rng_tp":None, "vel_tp":None, "az_tp":None}

    rplt.static_plot_selections(lst_det_s, None, selection_pl, output_path)


if __name__ == "__main__":
    conf_data = dc.parse_CMDLine("./analysis.cnf")
    if conf_data:
        main(conf_data)
