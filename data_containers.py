import scipy.io as sio
import numpy as np
import configparser
import argparse


class DetectionPoint(object):
    def __init__(self, mcc=0, beam=0,
                 nodet_permcc=0, trackID=0, rng=0.0,
                 vel=0.0, azimuth=0.0, left=True, car_width=1.88):

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

class ReferencePoint(object):
    def __init__(self, mccL=0, mccR=0, TAR_dist=0.0, TAR_distX=0.0,TAR_distY=0.0,
                 TAR_velX=0.0, TAR_velY=0.0, TAR_hdg=0.0,
                 EGO_velX=0.0, EGO_velY=0.0, EGO_accX=0.0, EGO_accY=0.0, EGO_hdg=0.0,):

        self._mccL = mccL
        self._mccR = mccR
        self._TAR_dist = TAR_dist
        self._TAR_distX = TAR_distX
        self._TAR_distY = TAR_distY
        self._TAR_velX = TAR_velX
        self._TAR_velY = TAR_velY
        self._TAR_hdg = TAR_hdg
        self._EGO_velX = EGO_velX
        self._EGO_velY = EGO_velY
        self._EGO_accX = EGO_accX
        self._EGO_accY = EGO_accY
        self._EGO_hdg = EGO_hdg

class TrackPoint(object):
    def __init__(self,mcc,x,y,dx,dy,beam):
        self.mcc = mcc
        self.x = x
        self.dx = dx
        self.y = y
        self.dy = dy
        self._beam = beam

    def get_array(self):
        x = np.array([self.x,self.dx,self.y,self.dy])
        return x.reshape(4,1)


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



class ReferenceList(list):
    def __init__(self):

        super().__init__()
        self._mccL_interval = (0,0)
        self._mccR_interval = (0,0)

    def append_from_m_file(self, data_path):

        DGPS_data = sio.loadmat(data_path)

        no_dL = len(DGPS_data["MCC_LeftRadar"])
        no_dR = len(DGPS_data["MCC_RightRadar"])
        no_d = max(no_dL,no_dR)

        print("DGPSdata Left:",len(DGPS_data["MCC_LeftRadar"]))
        print("DGPSdata Left list:",int(DGPS_data["MCC_LeftRadar"][20]))
        for itr in range(0, no_d - 1):
            self.append(ReferencePoint(mccL=int(DGPS_data["MCC_LeftRadar"][itr]),
                                       mccR=int(DGPS_data["MCC_RightRadar"][itr]),
                                       TAR_dist=float(DGPS_data["TARGET_dist"][itr]),
                                       TAR_distX=float(DGPS_data["TARGET_distX"][itr]),
                                       TAR_distY=float(DGPS_data["TARGET_distY"][itr]),
                                       TAR_velX=float(DGPS_data["TARGET_AbsVel_x"][itr]),
                                       TAR_velY=float(DGPS_data["TARGET_AbsVel_y"][itr]),
                                       TAR_hdg=float(DGPS_data["TARGET_Heading"][itr]),
                                       EGO_velX=float(DGPS_data["EGO_AbsVel_x"][itr]),
                                       EGO_velY=float(DGPS_data["EGO_AbsVel_y"][itr]),
                                       EGO_accX=float(DGPS_data["EGO_Acc_x"][itr]),
                                       EGO_accY=float(DGPS_data["EGO_Acc_y"][itr]),
                                       EGO_hdg=float(DGPS_data["EGO_Heading"][itr])
                                       ))

        self._mccL_interval = (min([elem._mccL for elem in self]),max([elem._mccL for elem in self]))
        self._mccR_interval = (min([elem._mccR for elem in self]),max([elem._mccR for elem in self]))

    def get_mccL_interval(self):

        return self._mccL_interval

    def get_mccR_interval(self):

        return self._mccR_interval

    def get_mccB_interval(self):

        mcc_min = min(self._mccL_interval[0],self._mccR_interval[0])
        mcc_max = max(self._mccL_interval[1],self._mccR_interval[1])
        mccB = (mcc_min,mcc_max)
        return mccB


    def get_array_references_selected(self, **kwarg):

        if 'mccL' in kwarg:
            if kwarg['mccL']:
                mccL_i = kwarg['mccL'] if (len(kwarg['mccL']) == 2) else (kwarg['mccL'],kwarg['mccL'])
            else:
                mccL_i = self._mccL_interval
        else:
            mccL_i = self._mccL_interval

        if 'mccR' in kwarg:
            if kwarg['mccR']:
                mccR_i = kwarg['mccR'] if (len(kwarg['mccR']) == 2) else (kwarg['mccR'],kwarg['mccR'])
            else:
                mccR_i = self._mccR_interval
        else:
            mccR_i = self._mccR_interval

        mccL_sel = [elem._mccL for elem in self if ( mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                     mccR_i[0] <= elem._mccR <= mccR_i[1] )]
        mccR_sel = [elem._mccR for elem in self if ( mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                     mccR_i[0] <= elem._mccR <= mccR_i[1] )]
        TAR_dist_sel = [elem._TAR_dist for elem in self if ( mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                             mccR_i[0] <= elem._mccR <= mccR_i[1] )]
        TAR_distX_sel = [elem._TAR_distX for elem in self if ( mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                               mccR_i[0] <= elem._mccR <= mccR_i[1] )]
        TAR_distY_sel = [elem._TAR_distY for elem in self if ( mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                               mccR_i[0] <= elem._mccR <= mccR_i[1] )]
        TAR_velX_sel = [elem._TAR_velX for elem in self if ( mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                             mccR_i[0] <= elem._mccR <= mccR_i[1] )]
        TAR_velY_sel = [elem._TAR_velY for elem in self if ( mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                             mccR_i[0] <= elem._mccR <= mccR_i[1] )]
        TAR_hdg_sel = [elem._TAR_hdg for elem in self if ( mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                           mccR_i[0] <= elem._mccR <= mccR_i[1] )]
        EGO_velX_sel = [elem._EGO_velX for elem in self if ( mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                             mccR_i[0] <= elem._mccR <= mccR_i[1] )]
        EGO_velY_sel = [elem._EGO_velY for elem in self if ( mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                               mccR_i[0] <= elem._mccR <= mccR_i[1] )]
        EGO_accX_sel = [elem._EGO_accX for elem in self if ( mccL_i[0] <= elem._mccL <= mccL_i[1] and mccR_i[0] <= elem._mccR <= mccR_i[1] )]
        EGO_accY_sel = [elem._EGO_accY for elem in self if ( mccL_i[0] <= elem._mccL <= mccL_i[1] and mccR_i[0] <= elem._mccR <= mccR_i[1] )]
        EGO_hdg_sel = [elem._EGO_hdg for elem in self if ( mccL_i[0] <= elem._mccL <= mccL_i[1] and mccR_i[0] <= elem._mccR <= mccR_i[1] )]

        DGPS_data = { "mccL": np.array(mccL_sel),
                      "mccR": np.array(mccR_sel),
                      "TAR_dist": np.array(TAR_dist_sel),
                      "TAR_distX": np.array(TAR_distX_sel),
                      "TAR_distY": np.array(TAR_distY_sel),
                      "TAR_velX": np.array(TAR_velX_sel),
                      "TAR_velY": np.array(TAR_velY_sel),
                      "TAR_hdg": np.array(TAR_hdg_sel),
                      "EGO_velX": np.array(EGO_velX_sel),
                      "EGO_velY": np.array(EGO_velY_sel),
                      "EGO_accX": np.array(EGO_accX_sel),
                      "EGO_accY": np.array(EGO_accY_sel),
                      "EGO_hdg": np.array(EGO_hdg_sel)}

        return DGPS_data


class Track(list):
    def __init__(self):
        super().__init__()
        self._y_predicted = (0,0)
        self._x_predicted = (0,0)
        self._trackID = (0,0)
        self._velx_interval = (0,0)
        self._x_interval = (0,0)
        self._vely_interval = (0,0)
        self._y_interval = (0,0)
        self._mcc_interval = (0,0)

    def append_point(self, mcc,x,y,dx,dy,beam):
        self.append(TrackPoint(mcc,x,y,dx,dy,beam))
        self._y_interval = (min([elem._y for elem in self]),max([elem._y for elem in self]))
        self._x_interval = (min([elem._x for elem in self]),max([elem._x for elem in self]))
        self._vely_interval = (min([elem._vely for elem in self]),max([elem._vely for elem in self]))
        self._velx_interval = (min([elem._velx for elem in self]),max([elem._velx for elem in self]))
        self._mcc_interval = (min([elem._mcc for elem in self]),max([elem._mcc for elem in self]))

    def get_mcc_interval(self):

        return self._mcc_interval




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
    DGPS_xcompensation = config.get(scenario, 'DGPS_xcompensation')

    data_filenames = {"filename_LeftRadar": filename_LeftRadar,
                      "filename_RightRadar": filename_RightRadar,
                      "filename_LeftDGPS": filename_LeftDGPS,
                      "filename_RightDGPS": filename_RightDGPS,
                      "filename_BothDGPS": filename_BothDGPS,
                      "DGPS_xcompensation": DGPS_xcompensation}
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
                         "DGPS_xcompensation": data_filenames["DGPS_xcompensation"],
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

    return conf_data_out
