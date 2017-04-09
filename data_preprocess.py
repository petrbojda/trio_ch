#!/usr/bin/env python
import data_containers as dc
import numpy as np
import scipy.io as sio


def main(conf_data):
    if conf_data["filename_LeftRadar"]:
        l = []
        l.append(conf_data["path_data_folder"])
        l.append(conf_data["filename_LeftRadar"])
        leftradar_path = ''.join(l)

        lst_det_left = dc.DetectionList()
        lst_det_left.append_from_m_file(leftradar_path, True, conf_data["EGO_car_width"])

        LR0_data = lst_det_left.get_array_detections_selected(beam = [0])
        LR1_data = lst_det_left.get_array_detections_selected(beam = [1])
        LR2_data = lst_det_left.get_array_detections_selected(beam = [2])
        LR3_data = lst_det_left.get_array_detections_selected(beam = [3])
        print("LR Beam 0: N of Det: ", len(LR0_data["mcc"]), "starting MCC: ", min(LR0_data["mcc"]), "ending MCC: ",
              max(LR0_data["mcc"]))
        print("LR Beam 1: N of Det: ", len(LR1_data["mcc"]), "starting MCC: ", min(LR1_data["mcc"]), "ending MCC: ",
              max(LR1_data["mcc"]))
        print("LR Beam 2: N of Det: ", len(LR2_data["mcc"]), "starting MCC: ", min(LR2_data["mcc"]), "ending MCC: ",
              max(LR2_data["mcc"]))
        print("LR Beam 3: N of Det: ", len(LR3_data["mcc"]), "starting MCC: ", min(LR3_data["mcc"]), "ending MCC: ",
              max(LR3_data["mcc"]))

        print("LR Beam 0: min vel: ", min(LR0_data["velocity"]), "max vel: ", max(LR0_data["velocity"]))
        print("LR Beam 1: min vel: ", min(LR1_data["velocity"]), "max vel: ", max(LR1_data["velocity"]))
        print("LR Beam 2: min vel: ", min(LR2_data["velocity"]), "max vel: ", max(LR2_data["velocity"]))
        print("LR Beam 3: min vel: ", min(LR3_data["velocity"]), "max vel: ", max(LR3_data["velocity"]))

        print("LR Beam 0: min az: ", min(LR0_data["azimuth"]), "max az: ", max(LR0_data["azimuth"]))
        print("LR Beam 1: min az: ", min(LR1_data["azimuth"]), "max az: ", max(LR1_data["azimuth"]))
        print("LR Beam 2: min az: ", min(LR2_data["azimuth"]), "max az: ", max(LR2_data["azimuth"]))
        print("LR Beam 3: min az: ", min(LR3_data["azimuth"]), "max az: ", max(LR3_data["azimuth"]))

        print("LR Beam 0: min rng: ", min(LR0_data["range"]), "max rng: ", max(LR0_data["range"]))
        print("LR Beam 1: min rng: ", min(LR1_data["range"]), "max rng: ", max(LR1_data["range"]))
        print("LR Beam 2: min rng: ", min(LR2_data["range"]), "max rng: ", max(LR2_data["range"]))
        print("LR Beam 3: min rng: ", min(LR3_data["range"]), "max rng: ", max(LR3_data["range"]))

        max_detections_per_mcc_left, at = lst_det_left.get_max_of_detections_per_mcc()
        print("Max detections in a mcc's", max_detections_per_mcc_left, "at", at)

        mcc_min, mcc_max = lst_det_left.get_mcc_interval()
        number_of_mccs_left = mcc_max - mcc_min
        print("Number of mcc samples", number_of_mccs_left)
        m = max_detections_per_mcc_left if max_detections_per_mcc_left > 20 else 20
        array_of_detections_left = np.empty((2, m, number_of_mccs_left))
        array_of_detections_left.fill(np.nan)

        for i in range(0, number_of_mccs_left):
            det = [elem._x for elem in lst_det_left if (elem._mcc == i + mcc_min)]
            n_o_el = np.size(det)
            array_of_detections_left[0, 0:n_o_el, i] = [elem._x for elem in lst_det_left if (elem._mcc == i + mcc_min)]
            array_of_detections_left[1, 0:n_o_el, i] = [elem._y for elem in lst_det_left if (elem._mcc == i + mcc_min)]
            print("n o Elements in a mcc", i, "is", n_o_el)

        L = {"detections": array_of_detections_left}

        print("Matrix size:", array_of_detections_left.shape)
        sio.savemat("LR_detections.mat", L)


    if conf_data["filename_RightRadar"]:
        l = []
        l.append(conf_data["path_data_folder"])
        l.append(conf_data["filename_RightRadar"])
        rightradar_path = ''.join(l)

        lst_det_right = dc.DetectionList()
        lst_det_right.append_from_m_file(rightradar_path, True, conf_data["EGO_car_width"])

        RR0_data = lst_det_right.get_array_detections_selected(beam = [0])
        RR1_data = lst_det_right.get_array_detections_selected(beam = [1])
        RR2_data = lst_det_right.get_array_detections_selected(beam = [2])
        RR3_data = lst_det_right.get_array_detections_selected(beam = [3])
        print("RR Beam 0: N of Det: ", len(RR0_data["mcc"]), "starting MCC: ", min(RR0_data["mcc"]), "ending MCC: ",
              max(RR0_data["mcc"]))
        print("RR Beam 1: N of Det: ", len(RR1_data["mcc"]), "starting MCC: ", min(RR1_data["mcc"]), "ending MCC: ",
              max(RR1_data["mcc"]))
        print("RR Beam 2: N of Det: ", len(RR2_data["mcc"]), "starting MCC: ", min(RR2_data["mcc"]), "ending MCC: ",
              max(RR2_data["mcc"]))
        print("RR Beam 3: N of Det: ", len(RR3_data["mcc"]), "starting MCC: ", min(RR3_data["mcc"]), "ending MCC: ",
              max(RR3_data["mcc"]))

        print("RR Beam 0: min vel: ", min(RR0_data["velocity"]), "max vel: ", max(RR0_data["velocity"]))
        print("RR Beam 1: min vel: ", min(RR1_data["velocity"]), "max vel: ", max(RR1_data["velocity"]))
        print("RR Beam 2: min vel: ", min(RR2_data["velocity"]), "max vel: ", max(RR2_data["velocity"]))
        print("RR Beam 3: min vel: ", min(RR3_data["velocity"]), "max vel: ", max(RR3_data["velocity"]))

        print("RR Beam 0: min az: ", min(RR0_data["azimuth"]), "max az: ", max(RR0_data["azimuth"]))
        print("RR Beam 1: min az: ", min(RR1_data["azimuth"]), "max az: ", max(RR1_data["azimuth"]))
        print("RR Beam 2: min az: ", min(RR2_data["azimuth"]), "max az: ", max(RR2_data["azimuth"]))
        print("RR Beam 3: min az: ", min(RR3_data["azimuth"]), "max az: ", max(RR3_data["azimuth"]))

        print("RR Beam 0: min rng: ", min(RR0_data["range"]), "max rng: ", max(RR0_data["range"]))
        print("RR Beam 1: min rng: ", min(RR1_data["range"]), "max rng: ", max(RR1_data["range"]))
        print("RR Beam 2: min rng: ", min(RR2_data["range"]), "max rng: ", max(RR2_data["range"]))
        print("RR Beam 3: min rng: ", min(RR3_data["range"]), "max rng: ", max(RR3_data["range"]))

        max_detections_per_mcc_right, at = lst_det_right.get_max_of_detections_per_mcc()
        print("Max detections in a mcc's", max_detections_per_mcc_right, "at", at)

        mcc_min, mcc_max = lst_det_right.get_mcc_interval()
        number_of_mccs_right = mcc_max - mcc_min
        print("Number of mcc samples", number_of_mccs_right)
        m = max_detections_per_mcc_right if max_detections_per_mcc_right > 20 else 20
        array_of_detections_right = np.empty((2, m, number_of_mccs_right))
        array_of_detections_right.fill(np.nan)

        for i in range(0, number_of_mccs_right):
            det = [elem._x for elem in lst_det_right if (elem._mcc == i + mcc_min)]
            n_o_el = np.size(det)
            array_of_detections_right[0, 0:n_o_el, i] = [elem._x for elem in lst_det_right if (elem._mcc == i + mcc_min)]
            array_of_detections_right[1, 0:n_o_el, i] = [elem._y for elem in lst_det_right if (elem._mcc == i + mcc_min)]
            print("n o Elements in a mcc", i, "is", n_o_el)

        R = {"detections": array_of_detections_right}

        print("Matrix size:", array_of_detections_right.shape)
        sio.savemat("RR_detections.mat", R)

if __name__ == "__main__":
    conf_data = dc.parse_CMDLine("./analysis.cnf")
    if conf_data:
        main(conf_data)
