#!/usr/bin/env python
import data_containers as dc
import radar_plots as rplt

def main(conf_data):

    selection = {"beam_tp":conf_data["beams_tp"],
                 "mcc_tp":None, "x_tp":None, "y_tp":None,
                 "rng_tp":None, "vel_tp":(-5,5), "az_tp":None}

    if conf_data["filename_LeftRadar"]:
        l = []
        l.append(conf_data["path_data_folder"])
        l.append(conf_data["filename_LeftRadar"])
        leftradar_path = ''.join(l)

        lst_det_left = dc.DetectionList()
        lst_det_left.append_from_m_file(leftradar_path, True, conf_data["EGO_car_width"])

        LR_data = lst_det_left.get_array_detections_selected(selection=selection)

        print("LR : N of Det: ", len(LR_data["mcc"]), "starting MCC: ", min(LR_data["mcc"]), "ending MCC: ",
              max(LR_data["mcc"]))
        print("LR : min vel: ", min(LR_data["velocity"]), "max vel: ", max(LR_data["velocity"]))
        print("LR : min az: ", min(LR_data["azimuth"]), "max az: ", max(LR_data["azimuth"]))
        print("LR : min rng: ", min(LR_data["range"]), "max rng: ", max(LR_data["range"]))
        print("LR : min x: ", min(LR_data["x"]), "max x: ", max(LR_data["x"]))
        print("LR : min y: ", min(LR_data["y"]), "max y: ", max(LR_data["y"]))

    if conf_data["filename_RightRadar"]:
        l = []
        l.append(conf_data["path_data_folder"])
        l.append(conf_data["filename_RightRadar"])
        rightradar_path = ''.join(l)

        lst_det_right = dc.DetectionList()
        lst_det_right.append_from_m_file(rightradar_path, False, conf_data["EGO_car_width"])

        RR_data = lst_det_right.get_array_detections_selected(selection=selection)

        print("RR : N of Det: ", len(RR_data["mcc"]), "starting MCC: ", min(RR_data["mcc"]), "ending MCC: ",
                  max(RR_data["mcc"]))
        print("RR : min vel: ", min(RR_data["velocity"]), "max vel: ", max(RR_data["velocity"]))
        print("RR : min az: ", min(RR_data["azimuth"]), "max az: ", max(RR_data["azimuth"]))
        print("RR : min rng: ", min(RR_data["range"]), "max rng: ", max(RR_data["range"]))
        print("RR : min x: ", min(RR_data["x"]), "max x: ", max(RR_data["x"]))
        print("RR : min y: ", min(RR_data["y"]), "max y: ", max(RR_data["y"]))

    if conf_data["output_folder"]:
        l = []
        l.append(conf_data["output_folder"])
        fname_det = '_tmp%s.png' % conf_data["scenario"]
        l.append(fname_det)
        output_path = ''.join(l)
    else:
        output_path = None

    rplt.static_plot_selections(lst_det_left, lst_det_right, selection, output_path)

if __name__ == "__main__":
    conf_data = dc.parse_CMDLine("./analysis.cnf")
    if conf_data:
        main(conf_data)
