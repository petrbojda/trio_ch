#!/usr/bin/env python

import data_containers as dc
import radar_plots as rplt

import cProfile, pstats, io

def main(conf_data):
    # Load Data from .mat file
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
    else:
        lst_det_left = None

    if conf_data["filename_RightRadar"]:
        l = []
        l.append(conf_data["path_data_folder"])
        l.append(conf_data["filename_RightRadar"])
        rightradar_path = ''.join(l)

        lst_det_right = dc.DetectionList()
        lst_det_right.append_from_m_file(rightradar_path, False, conf_data["EGO_car_width"])
        mcc_interval_right = lst_det_right.get_mcc_interval()
        print("MCC Right starts at: ", mcc_interval_right[0], "and ends at: ", mcc_interval_right[1])
    else:
        lst_det_right = None

    if conf_data["filename_LeftDGPS"]:
        l = []
        l.append(conf_data["path_data_folder"])
        l.append(conf_data["filename_LeftDGPS"])
        leftDGPS_path = ''.join(l)

        lst_ref_left = dc.ReferenceList()
        lst_ref_left.append_from_m_file(leftDGPS_path)
        mcc_intervalDGPS_left = lst_ref_left.get_mccL_interval()
        print("MCC Left DGPS starts at: ", mcc_intervalDGPS_left[0], "and ends at: ", mcc_intervalDGPS_left[1])
    else:
        lst_ref_left = None

    if conf_data["filename_RightDGPS"]:
        l = []
        l.append(conf_data["path_data_folder"])
        l.append(conf_data["filename_RightDGPS"])
        rightDGPS_path = ''.join(l)

        lst_ref_right = dc.ReferenceList()
        lst_ref_right.append_from_m_file(rightDGPS_path)
        mcc_intervalDGPS_right = lst_ref_right.get_mccR_interval()
        print("MCC Left DGPS starts at: ", mcc_intervalDGPS_right[0], "and ends at: ", mcc_intervalDGPS_right[1])
    else:
        lst_ref_right = None

    if conf_data["filename_BothDGPS"]:
        l = []
        l.append(conf_data["path_data_folder"])
        l.append(conf_data["filename_BothDGPS"])
        bothDGPS_path = ''.join(l)

        lst_ref_both = dc.ReferenceList()
        lst_ref_both.append_from_m_file(bothDGPS_path)
        mcc_intervalDGPS_both = lst_ref_both.get_mccB_interval()
        print("MCC Both DGPS starts at: ", mcc_intervalDGPS_both[0], "and ends at: ", mcc_intervalDGPS_both[1])
    else:
        lst_ref_both = None


    if conf_data["output_folder"]:
        l = []
        l.append(conf_data["output_folder"])
        fname_det = '_tmp%s.png' % conf_data["scenario"]
        l.append(fname_det)
        output_path = ''.join(l)
    else:
        output_path = None

    selection = {"beam_tp":conf_data["beams_tp"],
                 "mcc_tp":None, "x_tp":None, "y_tp":None,
                 "rng_tp":None, "vel_tp":None, "az_tp":None}

    #rplt.static_plot_selections(lst_det_left, lst_det_right, selection, output_path)
    rplt.static_plotREF_selections(lst_det_left, lst_det_right,
                                   lst_ref_left,lst_ref_right,lst_ref_both,
                                   selection, output_path)

pr = cProfile.Profile()
pr.enable()

if __name__ == "__main__":
    conf_data = dc.parse_CMDLine("./analysis.cnf")

if conf_data:
        main(conf_data)

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
#print (s.getvalue())
