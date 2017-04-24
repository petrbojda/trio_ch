import scipy.io as sio
import numpy as np
import configparser
import argparse


class DetectionPoint(object):
    def __init__(self, mcc=0, beam=0,
                 nodet_permcc=0, trackID=0, rng=0,
                 vel=0, azimuth=0, left=True, car_width=1.88):
        self._y_correction_dir = -1 if left else 1
        self._mcc = mcc
        self._beam = beam
        self._nodet = nodet_permcc
        self._trackID = trackID
        self._rng = rng
        self._vel = vel
        self._azimuth = azimuth
        self._x = self._rng * np.cos(self._azimuth)
        self._y = self._y_correction_dir * (self._rng * np.sin(self._azimuth) + car_width / 2)


class DetectionList(list):
    def __init__(self):
        super().__init__()
        self._y_interval = (0,0)
        self._x_interval = (0,0)
        self._azimuth_interval = (0,0)
        self._vel_interval = (0,0)
        self._rng_interval = (0,0)
        self._mcc_interval = (0,0)



    def append_from_m_file(self, data_path, left, car_width):
        radar_data = sio.loadmat(data_path)
        detections = radar_data["Detections"]
        no_d = len(detections)
        for itr in range(0, no_d - 1):
            self.append(DetectionPoint(mcc=int(detections[itr, 0]),
                                       beam=int(detections[itr, 2]),
                                       nodet_permcc=int(detections[itr, 3]),
                                       trackID=0,
                                       rng=float(detections[itr, 5]),
                                       vel=float(detections[itr, 6]),
                                       azimuth=float(detections[itr, 7]),
                                       left=bool(left),
                                       car_width=float(car_width)))

        self._y_interval = (min([elem._y for elem in self]),max([elem._y for elem in self]))
        self._x_interval = (min([elem._x for elem in self]),max([elem._x for elem in self]))
        self._azimuth_interval = (min([elem._azimuth for elem in self]),max([elem._azimuth for elem in self]))
        self._vel_interval = (min([elem._vel for elem in self]),max([elem._vel for elem in self]))
        self._rng_interval = (min([elem._rng for elem in self]),max([elem._rng for elem in self]))
        self._mcc_interval = (min([elem._mcc for elem in self]),max([elem._mcc for elem in self]))

    def get_mcc_interval(self):
        return self._mcc_interval

    def get_max_of_detections_per_mcc(self):
        max_detections_at = max([elem._mcc for elem in self], key=[elem._mcc for elem in self].count)
        max_no_detections = [elem._mcc for elem in self].count(max_detections_at)
        return max_no_detections, max_detections_at



    def get_array_detections_selected(self, **kwarg):

        if 'beam' in kwarg:
            beam = kwarg['beam']
        else:
            beam = [0,1,2,3]

        if 'mcc' in kwarg:
            mcc_i = kwarg['mcc'] if (len(kwarg['mcc']) == 2) else (kwarg['mcc'],kwarg['mcc'])
        else:
            mcc_i = self._mcc_interval

        if 'x' in kwarg:
            x_i = kwarg['x'] if (len(kwarg['x']) == 2) else (kwarg['x'],kwarg['x'])
        else:
            x_i = self._x_interval

        if 'y' in kwarg:
            y_i = kwarg['y'] if (len(kwarg['y']) == 2) else (kwarg['y'],kwarg['y'])
        else:
            y_i = self._y_interval

        if 'rng' in kwarg:
            rng_i = kwarg['rng'] if (len(kwarg['rng']) == 2) else (kwarg['rng'],kwarg['rng'])
        else:
            rng_i = self._rng_interval

        if 'vel' in kwarg:
            vel_i = kwarg['vel'] if (len(kwarg['vel']) == 2) else (kwarg['vel'],kwarg['vel'])
        else:
            vel_i = self._vel_interval

        if 'az' in kwarg:
            az_i = kwarg['az'] if (len(kwarg['az']) == 2) else (kwarg['az'],kwarg['az'])
        else:
            az_i = self._azimuth_interval

        if 'selection' in kwarg:
            beam = kwarg['selection']['beam_tp'] if kwarg['selection']['beam_tp'] else [0,1,2,3]
            mcc_i = kwarg['selection']['mcc_tp'] if kwarg['selection']['mcc_tp'] else self._mcc_interval
            x_i = kwarg['selection']['x_tp'] if kwarg['selection']['x_tp'] else self._x_interval
            y_i = kwarg['selection']['y_tp'] if kwarg['selection']['y_tp'] else self._y_interval
            rng_i = kwarg['selection']['rng_tp'] if kwarg['selection']['rng_tp'] else self._rng_interval
            vel_i = kwarg['selection']['vel_tp'] if kwarg['selection']['vel_tp'] else self._vel_interval
            az_i = kwarg['selection']['az_tp'] if kwarg['selection']['az_tp'] else self._azimuth_interval


        r_sel = [elem._rng for elem in self if (elem._beam in beam and
                                                mcc_i[0] <= elem._mcc <= mcc_i[1] and
                                                x_i[0] <= elem._x <= x_i[1] and
                                                y_i[0] <= elem._y <= y_i[1] and
                                                rng_i[0] <= elem._rng <= rng_i[1] and
                                                vel_i[0] <= elem._vel <= vel_i[1] and
                                                az_i[0] <= elem._azimuth <= az_i[1])]
        v_sel = [elem._vel for elem in self if (elem._beam in beam and
                                                mcc_i[0] <= elem._mcc <= mcc_i[1] and
                                                x_i[0] <= elem._x <= x_i[1] and
                                                y_i[0] <= elem._y <= y_i[1] and
                                                rng_i[0] <= elem._rng <= rng_i[1] and
                                                vel_i[0] <= elem._vel <= vel_i[1] and
                                                az_i[0] <= elem._azimuth <= az_i[1])]
        az_sel = [elem._azimuth for elem in self if (elem._beam in beam and
                                                mcc_i[0] <= elem._mcc <= mcc_i[1] and
                                                x_i[0] <= elem._x <= x_i[1] and
                                                y_i[0] <= elem._y <= y_i[1] and
                                                rng_i[0] <= elem._rng <= rng_i[1] and
                                                vel_i[0] <= elem._vel <= vel_i[1] and
                                                az_i[0] <= elem._azimuth <= az_i[1])]
        mcc_sel = [elem._mcc for elem in self if (elem._beam in beam and
                                                mcc_i[0] <= elem._mcc <= mcc_i[1] and
                                                x_i[0] <= elem._x <= x_i[1] and
                                                y_i[0] <= elem._y <= y_i[1] and
                                                rng_i[0] <= elem._rng <= rng_i[1] and
                                                vel_i[0] <= elem._vel <= vel_i[1] and
                                                az_i[0] <= elem._azimuth <= az_i[1])]
        x_sel = [elem._x for elem in self if (  elem._beam in beam and
                                                mcc_i[0] <= elem._mcc <= mcc_i[1] and
                                                x_i[0] <= elem._x <= x_i[1] and
                                                y_i[0] <= elem._y <= y_i[1] and
                                                rng_i[0] <= elem._rng <= rng_i[1] and
                                                vel_i[0] <= elem._vel <= vel_i[1] and
                                                az_i[0] <= elem._azimuth <= az_i[1])]
        y_sel = [elem._y for elem in self if (  elem._beam in beam and
                                                mcc_i[0] <= elem._mcc <= mcc_i[1] and
                                                x_i[0] <= elem._x <= x_i[1] and
                                                y_i[0] <= elem._y <= y_i[1] and
                                                rng_i[0] <= elem._rng <= rng_i[1] and
                                                vel_i[0] <= elem._vel <= vel_i[1] and
                                                az_i[0] <= elem._azimuth <= az_i[1])]
        beam_sel = [elem._beam for elem in self if (  elem._beam in beam and
                                                mcc_i[0] <= elem._mcc <= mcc_i[1] and
                                                x_i[0] <= elem._x <= x_i[1] and
                                                y_i[0] <= elem._y <= y_i[1] and
                                                rng_i[0] <= elem._rng <= rng_i[1] and
                                                vel_i[0] <= elem._vel <= vel_i[1] and
                                                az_i[0] <= elem._azimuth <= az_i[1])]

        radar_data = {"range": np.array(r_sel),
                      "azimuth": np.array(az_sel),
                      "velocity": np.array(v_sel),
                      "x": np.array(x_sel),
                      "y": np.array(y_sel),
                      "beam": np.array(beam_sel),
                      "mcc": np.array(mcc_sel)}

        return radar_data

    def extend_with_selection(self, radar_data_list, **kwarg):

        if 'beam' in kwarg:
            beam = kwarg['beam']
        else:
            beam = [0,1,2,3]

        if 'mcc' in kwarg:
            mcc_i = kwarg['mcc'] if (len(kwarg['mcc']) == 2) else (kwarg['mcc'],kwarg['mcc'])
        else:
            mcc_i = radar_data_list._mcc_interval

        if 'x' in kwarg:
            x_i = kwarg['x'] if (len(kwarg['x']) == 2) else (kwarg['x'],kwarg['x'])
        else:
            x_i = radar_data_list._x_interval

        if 'y' in kwarg:
            y_i = kwarg['y'] if (len(kwarg['y']) == 2) else (kwarg['y'],kwarg['y'])
        else:
            y_i = radar_data_list._y_interval

        if 'rng' in kwarg:
            rng_i = kwarg['rng'] if (len(kwarg['rng']) == 2) else (kwarg['rng'],kwarg['rng'])
        else:
            rng_i = radar_data_list._rng_interval

        if 'vel' in kwarg:
            vel_i = kwarg['vel'] if (len(kwarg['vel']) == 2) else (kwarg['vel'],kwarg['vel'])
        else:
            vel_i = radar_data_list._vel_interval

        if 'az' in kwarg:
            az_i = kwarg['az'] if (len(kwarg['az']) == 2) else (kwarg['az'],kwarg['az'])
        else:
            az_i = radar_data_list._azimuth_interval

        if 'selection' in kwarg:
            beam = kwarg['selection']['beam_tp'] if kwarg['selection']['beam_tp'] else [0,1,2,3]
            mcc_i = kwarg['selection']['mcc_tp'] if kwarg['selection']['mcc_tp'] else radar_data_list._mcc_interval
            x_i = kwarg['selection']['x_tp'] if kwarg['selection']['x_tp'] else radar_data_list._x_interval
            y_i = kwarg['selection']['y_tp'] if kwarg['selection']['y_tp'] else radar_data_list._y_interval
            rng_i = kwarg['selection']['rng_tp'] if kwarg['selection']['rng_tp'] else radar_data_list._rng_interval
            vel_i = kwarg['selection']['vel_tp'] if kwarg['selection']['vel_tp'] else radar_data_list._vel_interval
            az_i = kwarg['selection']['az_tp'] if kwarg['selection']['az_tp'] else radar_data_list._azimuth_interval


        for elem in radar_data_list:
            if (elem._beam in beam and
                            mcc_i[0] <= elem._mcc <= mcc_i[1] and
                            x_i[0] <= elem._x <= x_i[1] and
                            y_i[0] <= elem._y <= y_i[1] and
                            rng_i[0] <= elem._rng <= rng_i[1] and
                            vel_i[0] <= elem._vel <= vel_i[1] and
                            az_i[0] <= elem._azimuth <= az_i[1]):
                self.append(elem)


        self._y_interval = (min([elem._y for elem in self]),max([elem._y for elem in self]))
        self._x_interval = (min([elem._x for elem in self]),max([elem._x for elem in self]))
        self._azimuth_interval = (min([elem._azimuth for elem in self]),max([elem._azimuth for elem in self]))
        self._vel_interval = (min([elem._vel for elem in self]),max([elem._vel for elem in self]))
        self._rng_interval = (min([elem._rng for elem in self]),max([elem._rng for elem in self]))
        self._mcc_interval = (min([elem._mcc for elem in self]),max([elem._mcc for elem in self]))


def cnf_file_read(cnf_file):
    # Reads the configuration file
    config = configparser.ConfigParser()
    config.read(cnf_file)  # "./analysis.cnf"

    # Read list of available datasets
    new_data_folder = config.get('Datasets', 'data_new')
    old_data_folder = config.get('Datasets', 'data_old')

    # Read a path to a folder with python modules
    path_srcpy_folder = config.get('Paths', 'modules_dir')

    # Read a path to a folder with data
    path_data = config.get('Paths', 'data_dir')
    path_new_data = path_data + new_data_folder
    path_old_data = path_data + old_data_folder

    # Determines the list of available scenarios
    n_o_sc = int(config.get('Available_scenarios', 'number'))
    lst_scenarios_names = []
    for n_sc in range(0, n_o_sc):
        scen_n = "sc_{0:d}".format(n_sc)
        lst_scenarios_names.append(config.get('Available_scenarios', scen_n))
    ego_car_width = config.get('Geometry', 'EGO_car_width')

    conf_data = {"path_new_data": path_new_data,
                 "path_old_data": path_old_data,
                 "list_of_scenarios": lst_scenarios_names,
                 "Number_of_scenarios": n_o_sc,
                 "EGO_car_width": ego_car_width}
    return (conf_data)


def cnf_file_scenario_select(cnf_file, scenario):
    config = configparser.ConfigParser()
    config.read(cnf_file)  # "./analysis.cnf"

    filename_LeftRadar = config.get(scenario, 'left_radar')
    filename_RightRadar = config.get(scenario, 'right_radar')
    filename_LeftDGPS = config.get(scenario, 'left_dgps')
    filename_RightDGPS = config.get(scenario, 'right_dgps')
    filename_BothDGPS = config.get(scenario, 'both_dgps')

    data_filenames = {"filename_LeftRadar": filename_LeftRadar,
                      "filename_RightRadar": filename_RightRadar,
                      "filename_LeftDGPS": filename_LeftDGPS,
                      "filename_RightDGPS": filename_RightDGPS,
                      "filename_BothDGPS": filename_BothDGPS}
    return (data_filenames)


def parse_CMDLine(cnf_file):
    global path_data_folder
    conf_data = cnf_file_read(cnf_file)
    # Parses a set of input arguments comming from a command line
    parser = argparse.ArgumentParser(
        description='''
                            Python script analysis_start downloads data
                            prepared in a dedicated folder according to a
                            pre-defined scenario. Parameters are specified
                            in a configuration file. Scenario has to be
                            selected by an argument.''')
    #      Read command line arguments to get a scenario
    parser.add_argument("-s", "--scenario", help='''Sets an analysis to a given
                                                  scenario. The scenario has to
                                                  be one from an existing ones.''')
    #      Select the radar to process
    parser.add_argument("-r", "--radar",
                        help="Selects a radar(s) to process, one or both from L, R. Write L to process left radar, R to process right one or B to process both of them")
    #      Select the beam to process
    parser.add_argument("-b", "--beam",
                        help="Selects a beam(s) to process, one or more from 0,1,2,3")
    #      Select dataset to process
    parser.add_argument("-d", "--dataset",
                        help="Selects a dataset to process, the new one or the old one")
    #      List set of available scenarios
    parser.add_argument("-l", "--list", action="store_true",
                        help="Prints a list of available scenarios")
    #      Output folder
    parser.add_argument("-o", "--output",
                        help="Sets path to the folder where output files will be stored.")
    #      Select a scenario
    argv = parser.parse_args()

    if argv.beam:
        beams_tp = [int(s) for s in argv.beam.split(',')]
        beams_tp.sort()
    else:
        beams_tp = [0, 1, 2, 3]

    if argv.radar:
        radar_tp = argv.radar
    else:
        radar_tp = "B"

    if argv.dataset:
        dataset = argv.dataset
    else:
        dataset = "new"

    if argv.output:
        print("Output folder is:", argv.output)
        output = argv.output
    else:
        output = None

    if argv.list:
        print("Available scenarios are:")
        for n_sc in range(0, conf_data["Number_of_scenarios"]):
            print('\t \t \t', conf_data["list_of_scenarios"][n_sc])
        conf_data_out = False

    elif argv.scenario in conf_data["list_of_scenarios"]:

        if dataset == "new":
            path_data_folder = conf_data["path_new_data"]
        elif dataset == "old":
            path_data_folder = conf_data["path_old_data"]
        else:
            print("Wrong dataset selected.")

        data_filenames = cnf_file_scenario_select(cnf_file, argv.scenario)

        print("Dataset to process:", dataset)
        print("Data files are stored in:", path_data_folder)

        print("Data for the scenario are in:")
        print('\t \t left_radar:', data_filenames["filename_LeftRadar"])
        print('\t \t right_radar:', data_filenames["filename_RightRadar"])
        print('\t \t left_dgps:', data_filenames["filename_LeftDGPS"])
        print('\t \t right_dgps:', data_filenames["filename_RightDGPS"])
        print('\t \t both_dgps:', data_filenames["filename_BothDGPS"])

        print("Radar to process:", radar_tp)
        for n_beams in range(0, 4):
            if beams_tp.count(n_beams):
                print("Beam", n_beams, "will be processed:", beams_tp.count(n_beams), "times.")

        conf_data_out = {"scenario": argv.scenario,
                         "path_data_folder": path_data_folder,
                         "filename_LeftRadar": data_filenames["filename_LeftRadar"],
                         "filename_RightRadar": data_filenames["filename_RightRadar"],
                         "filename_LeftDGPS": data_filenames["filename_LeftDGPS"],
                         "filename_RightDGPS": data_filenames["filename_RightDGPS"],
                         "filename_BothDGPS": data_filenames["filename_BothDGPS"],
                         "EGO_car_width": conf_data["EGO_car_width"],
                         "beams_tp": beams_tp,
                         "radar_tp": radar_tp,
                         "output_folder": output}
        if radar_tp == "L":
            conf_data_out["filename_RightRadar"] = None
        elif radar_tp == "R":
            conf_data_out["filename_LeftRadar"] = None
        elif radar_tp == "B":
            conf_data_out["filename_LeftRadar"] = data_filenames["filename_LeftRadar"]
            conf_data_out["filename_RightRadar"] = data_filenames["filename_RightRadar"]
        else:
            conf_data_out["filename_LeftRadar"] = None
            conf_data_out["filename_RightRadar"] = None
            print("The input argument -r (--radar) is not correct")
            quit()

    else:
        print("No scenario selected.")
        conf_data_out = False

    return (conf_data_out)
