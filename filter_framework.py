#!/usr/bin/env python

import data_containers as dc
import track_management as tm
import radar_plots as rp
import numpy as np
import logging
import logging.config

class NoLoggerConfiguration(Exception): pass

def main(config_data):
    if config_data["filename_LOGcfg"]:
        cfg_logfile_path = config_data["filename_LOGcfg"]
        logging.config.fileConfig(cfg_logfile_path)
        # create logger
        logger = logging.getLogger(__name__)
        # logfile_level = config_data["log_level"]
        # level_of_logging = getattr(logging,logfile_level.upper())
        # logging.basicConfig(filename=logfile_path,level=level_of_logging)
        logger.info('Started the test_implementing')
        logger.info('Configuration file: %s',cfg_logfile_path)
        logger.info(76 * '=')
    else:
        raise(NoLoggerConfiguration())

    selection = {"beam_tp": config_data["beams_tp"],
                 "mcc_tp": None, "x_tp": None, "y_tp": None,
                 "rng_tp": None, "vel_tp": None, "az_tp": None,
                 "trackID_tp": None, }
    # Structure 'selection' constrains  data to use as input to the tracker.
    # An exact value or interval of 'mcc', 'azimuth', 'range' ... etc can be
    # specified here to block unwanted data to enter.

    # In this example only linear KF is being used, classical constant velocity model,
    # measurement contains only 2D vector (rho, theta)
    used_tracker_type = {'filter_type': 'kalman_filter', 'dim_x': 4, 'dim_z': 2}

    track_mgmt_LR = tm.TrackManager(tracker_type=used_tracker_type)
    track_mgmt_RR = tm.TrackManager(tracker_type=used_tracker_type)

    logger.info("Dataset to process: %s", config_data["scenario"])
    logger.info("Data files are stored in: %s", config_data["path_data_folder"])
    logger.info("Data for the scenario are in:")
    logger.info('\t \t left_radar: %s', config_data["filename_LeftRadar"])
    logger.info('\t \t right_radar: %s', config_data["filename_RightRadar"])
    logger.info('\t \t left_dgps: %s', config_data["filename_LeftDGPS"])
    logger.info('\t \t right_dgps %s:', config_data["filename_RightDGPS"])
    logger.info('\t \t both_dgps: %s', config_data["filename_BothDGPS"])
    logger.info('\t \t logger_cfg_file: %s', config_data["filename_LOGcfg"])
    logger.info("Radar to process: %s", config_data["radar_tp"])

    for n_beams in range(0, 4):
        if config_data["beams_tp"].count(n_beams):
            logger.info("Beams %d will be processed: %d times", n_beams, config_data["beams_tp"].count(n_beams))

    # Load Data from .mat files
    if config_data["filename_LeftRadar"]:
        logger.info(76 * '=')
        logger.info("Left Radar Data:")
        l = []
        l.append(config_data["path_data_folder"])
        l.append(config_data["filename_LeftRadar"])
        leftradar_path = ''.join(l)     

        lst_det_LR = dc.DetectionList()
        lst_det_LR.append_data_from_m_file(leftradar_path, True, config_data["EGO_car_width"])
        mcc_interval_LR = lst_det_LR.get_mcc_interval()
        logger.info('MCC Left start: %s, end: %s, dMCC=%d, number of detections: %d.',
                                                                            mcc_interval_LR[0],
                                                                            mcc_interval_LR[1],
                                                                            mcc_interval_LR[1] - mcc_interval_LR[0],
                                                                            len(lst_det_LR))

    if config_data["filename_RightRadar"]:
        logger.info(76 * '=')
        logger.info("Right Radar Data:")
        l = []
        l.append(config_data["path_data_folder"])
        l.append(config_data["filename_RightRadar"])
        rightradar_path = ''.join(l)

        lst_det_RR = dc.DetectionList()
        lst_det_RR.append_data_from_m_file(rightradar_path, False, config_data["EGO_car_width"])
        mcc_interval_RR = lst_det_RR.get_mcc_interval()
        logger.info('MCC Right start: %s, end: %s, dMCC=%d, number of detections: %d.',
                                                                             mcc_interval_RR[0],
                                                                             mcc_interval_RR[1],
                                                                             mcc_interval_RR[1] -  mcc_interval_RR[0],
                                                                             len(lst_det_RR))

    # Calculate valid mcc interval for detections to be presented
    logger.info(76 * '=')
    logger.info("Data Preprocessor Settings:")
    if config_data["filename_LeftRadar"] and config_data["filename_RightRadar"]:
        mcc_start = min(mcc_interval_LR[0], mcc_interval_RR[0])
        logger.debug('Both radars will be processed ')
        mcc_end = max(mcc_interval_LR[1], mcc_interval_RR[1])
    elif config_data["filename_LeftRadar"]:
        mcc_start = mcc_interval_LR[0]
        mcc_end = mcc_interval_LR[1]
        logger.debug('Just left radar will be processed ')
    else:
        mcc_start = mcc_interval_RR[0]
        mcc_end = mcc_interval_RR[1]
        logger.debug('Just right radar will be processed ')
    if config_data["number_of_mcc_to_process"]:
        mcc_end = min(mcc_start + int(config_data["number_of_mcc_to_process"]),mcc_end)
        logger.debug('Number of processed MCCs from cnf file: %s.', config_data["number_of_mcc_to_process"])
    logger.info('Processing will start at: %d, end: %d, dMCC=%d.',
                    mcc_start,
                    mcc_end,
                    mcc_end - mcc_start)


    logger.debug('Inside of the MCC interval from %s to %s: ', mcc_start, mcc_end)
    mcc_step = 1

    # Prepare options to plot results
    if config_data["plot_tp"]:
        l = []

    logger.debug(75 * '=')
    logger.debug("Filtering loop:")

    #----------------- Filtering loop
    i_prev = mcc_start
    for i in range(mcc_start, mcc_end, mcc_step):  # number of frames

        selection["mcc_tp"] = (i_prev, i)
        #-------------- Left radar filter
        if lst_det_LR:
            lst_det_per_loop_cycle_LR = lst_det_LR.get_lst_detections_selected(selection=selection)
            lst_det_per_loop_cycle_LR.calculate_intervals()
            # Is it correct to assign this for every iteration? Potential to write
            # more effective code.
            if lst_det_per_loop_cycle_LR:
                logger.debug('Number of detections for a LR mcc %d is %d', i, len(lst_det_per_loop_cycle_LR))
                track_mgmt_LR.new_detections(lst_det_per_loop_cycle_LR)
                # Let's see what data is in a list of not assigned detections and
                # how the new track is formed / if any

            else:
                logger.debug('There is no detection for current LR mcc %d.', i)
        logger.debug('Predict cycle for each track in a list of %d started for LR at mcc: %d.',len(track_mgmt_LR), i)
        track_mgmt_LR.predict(i)

        lst_not_assigned_LR, new_track_LR = track_mgmt_LR.port_data("track_init")
        list_of_tracks = track_mgmt_LR.port_data("tracks_array")
        if new_track_LR:
            logger.debug("Type of ported new_track list %s", type(new_track_LR))
            logger.debug("Type of ported new_track element %s", type(new_track_LR[-1]))
        else:
            logger.debug("New_track not ported/created")
        rp.static_plotTrackMan_initialization(lst_det_per_loop_cycle_LR,
                                              lst_not_assigned_LR,
                                              new_track_LR,
                                              list_of_tracks)
        # This line is redundant if only one mcc is being processed per loop cycle.
        # However if mcc_step is different than 1, it might be good to keep it here:
        i_prev = i + 1



#     TODO: graphical representation of the results

if __name__ == "__main__":
    config_data = dc.parse_CMDLine("./analysis.cnf")
    if config_data:
        try:
            main(config_data)
        except NoLoggerConfiguration:
            print("The log file cannot be created. Specify it's filename in a main config file.")
