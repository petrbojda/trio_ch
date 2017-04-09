import numpy as np


# import data_containers as dc

def g_h_constant_fltr():
    return ()


def g_h_benedbordner_fltr():
    return ()


def g_h_critdamp_fltr():
    return ()


def my_fltr_range(lst_det, mcc_start, mcc_end):
    for i_mcc in range(mcc_start, mcc_end):
        dets = lst_det.get_array_detections_MCCsel(i_mcc, i_mcc)
        no_det = np.size(dets["mcc"])
        for i_det in range(0, no_det):
            print("Detection:", i_det,
                  "mcc", dets["mcc"][i_det],
                  "range", dets["range"][i_det],
                  "velocity", dets["velocity"][i_det],
                  "azimuth", 180 / np.pi * dets["azimuth"][i_det],
                  "number of detections", no_det)
