#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid.axes_size as Size
from mpl_toolkits.axes_grid import Divider
import matplotlib
import copy


def static_plot_grid_hist_selections(lst_det_left, lst_det_right, selection, fname_det):
    """
	Plots data in an analytic way. Only one MCC's set of detections is depicted here.
	Input data dictionaries:
		Both left and right radars of the same structure
		radar_data = { 	"range": ,
						"azimuth": , 
						"velocity": ,
						"x": ,
						"y": ,
						"beam": ,
						"mcc":  	}

	"""
    ###### Plot starts here:
    cms = matplotlib.cm
    color_map_left = cms.Blues
    color_map_right = cms.RdPu

    if fname_det:
        f1 = plt.figure(1, (23, 13), dpi=300)
    else:
        f1 = plt.figure(1, (15, 8))

    vel_range = range(-90, 70)

    f1.clf()
    # the rect parameter will be ignore as we will set axes_locator
    rect = (0.05, 0.07, 0.9, 0.87)
    f1ax = [f1.add_axes(rect, label="%d" % i) for i in range(7)]
    horiz = [Size.Scaled(5.), Size.Fixed(1.0), Size.Scaled(1.5), Size.Fixed(1.0), Size.Scaled(2.)]
    vert = [Size.Scaled(1.), Size.Fixed(0.5), Size.Scaled(1.),
            Size.Fixed(0.5), Size.Scaled(1.), Size.Fixed(0.5), Size.Scaled(1.)]

    # divide the axes rectangle into grid whose size is specified by horiz * vert
    divider = Divider(f1, rect, horiz, vert, aspect=False)
    f1ax[0].set_axes_locator(divider.new_locator(nx=0, ny=0, ny1=7))
    f1ax[1].set_axes_locator(divider.new_locator(nx=2, ny=0))
    f1ax[2].set_axes_locator(divider.new_locator(nx=2, ny=2))
    f1ax[3].set_axes_locator(divider.new_locator(nx=2, ny=4))
    f1ax[4].set_axes_locator(divider.new_locator(nx=2, ny=6))
    f1ax[5].set_axes_locator(divider.new_locator(nx=4, ny=0, ny1=3))
    f1ax[6].set_axes_locator(divider.new_locator(nx=4, ny=4, ny1=7))

    f1ax[0].axis([-40, 100, -80, 80])
    f1ax[5].axis([-140, 140, 0, 100])
    f1ax[6].axis([-140, 140, -100, 80])

    f1ax[0].grid(True)
    f1ax[1].grid(True)
    f1ax[2].grid(True)
    f1ax[3].grid(True)
    f1ax[4].grid(True)
    f1ax[5].grid(True)
    f1ax[6].grid(True)

    number_of_dets_left = 0
    number_of_dets_left_processed = 0
    number_of_dets_right = 0
    number_of_dets_right_processed = 0

    #################### Left radar plot
    if lst_det_left:
        LR_data = lst_det_left.get_array_detections_selected(selection=selection)
        if LR_data["mcc"].any():
            LR_data_exists = True
            f1ax[2].hist(LR_data["velocity"], vel_range, color=color_map_left(0.4), normed=1)
            number_of_dets_left = np.size(LR_data["mcc"])

            if selection["beam_tp"].count(0):
                selection_tp = copy.deepcopy(selection)
                selection_tp["beam_tp"] = [0]
                LR0_data = lst_det_left.get_array_detections_selected(selection=selection_tp)
                f1ax[1].hist(LR0_data["velocity"], vel_range,
                             color=color_map_left(0.2), normed=1, label='beam 0')
                f1ax[0].plot(LR0_data["x"], LR0_data["y"],
                             color=color_map_left(0.2), marker='o', ls='None', label='Left RDR, beam 0')
                f1ax[6].plot(-180 * LR0_data["azimuth"] / np.pi, LR0_data["velocity"],
                             color=color_map_left(0.2), marker='o', ls='None', label='Left RDR, beam 0')
                f1ax[5].plot(-180 * LR0_data["azimuth"] / np.pi, LR0_data["range"],
                             color=color_map_left(0.2), marker='o', ls='None', label='Left RDR, beam 0')
                number_of_dets_left_processed += np.size(LR0_data["mcc"])

            if selection["beam_tp"].count(1):
                selection_tp = copy.deepcopy(selection)
                selection_tp["beam_tp"] = [1]
                LR1_data = lst_det_left.get_array_detections_selected(selection=selection_tp)
                f1ax[1].hist(LR1_data["velocity"], vel_range,
                             color=color_map_left(0.4), normed=1, label='beam 1')
                f1ax[0].plot(LR1_data["x"], LR1_data["y"],
                             color=color_map_left(0.4), marker='o', ls='None', label='Left RDR, beam 1')
                f1ax[6].plot(-180 * LR1_data["azimuth"] / np.pi, LR1_data["velocity"],
                             color=color_map_left(0.4), marker='o', ls='None', label='Left RDR, beam 1')
                f1ax[5].plot(-180 * LR1_data["azimuth"] / np.pi, LR1_data["range"],
                             color=color_map_left(0.4), marker='o', ls='None', label='Left RDR, beam 1')
                number_of_dets_left_processed += np.size(LR1_data["mcc"])

            if selection["beam_tp"].count(2):
                selection_tp = copy.deepcopy(selection)
                selection_tp["beam_tp"] = [2]
                LR2_data = lst_det_left.get_array_detections_selected(selection=selection_tp)
                f1ax[1].hist(LR2_data["velocity"], vel_range,
                             color=color_map_left(0.6), normed=1, label='beam 2')
                f1ax[0].plot(LR2_data["x"], LR2_data["y"],
                             color=color_map_left(0.6), marker='o', ls='None', label='Left RDR, beam 2')
                f1ax[6].plot(-180 * LR2_data["azimuth"] / np.pi, LR2_data["velocity"],
                             color=color_map_left(0.6), marker='o', ls='None', label='Left RDR, beam 2')
                f1ax[5].plot(-180 * LR2_data["azimuth"] / np.pi, LR2_data["range"],
                             color=color_map_left(0.6), marker='o', ls='None', label='Left RDR, beam 2')
                number_of_dets_left_processed += np.size(LR2_data["mcc"])

            if selection["beam_tp"].count(3):
                selection_tp = copy.deepcopy(selection)
                selection_tp["beam_tp"] = [3]
                LR3_data = lst_det_left.get_array_detections_selected(selection=selection_tp)
                f1ax[1].hist(LR3_data["velocity"], vel_range,
                             color=color_map_left(0.8), normed=1, label='beam 3')
                f1ax[0].plot(LR3_data["x"], LR3_data["y"],
                             color=color_map_left(0.8), marker='o', ls='None', label='Left RDR, beam 3')
                f1ax[6].plot(-180 * LR3_data["azimuth"] / np.pi, LR3_data["velocity"],
                             color=color_map_left(0.8), marker='o', ls='None', label='Left RDR, beam 3')
                f1ax[5].plot(-180 * LR3_data["azimuth"] / np.pi, LR3_data["range"],
                             color=color_map_left(0.8), marker='o', ls='None', label='Left RDR, beam 3')
                number_of_dets_left_processed += np.size(LR3_data["mcc"])

            plt.draw()
        else:
            LR_data_exists = False

    #################### Right radar plot
    if lst_det_right:
        RR_data = lst_det_right.get_array_detections_selected(mcc=selection['mcc_tp'])
        if RR_data["mcc"].any():
            RR_data_exists = True
            f1ax[4].hist(RR_data["velocity"], vel_range, color=color_map_left(0.4), normed=1)
            number_of_dets_right = np.size(RR_data["mcc"])

            if selection["beam_tp"].count(0):
                selection_tp = copy.deepcopy(selection)
                selection_tp["beam_tp"] = [0]
                RR0_data = lst_det_right.get_array_detections_selected(selection=selection_tp)
                f1ax[3].hist(RR0_data["velocity"], vel_range,
                             color=color_map_right(0.2), normed=1, label='beam 0')
                f1ax[0].plot(RR0_data["x"], RR0_data["y"],
                             color=color_map_right(0.2), marker='o', ls='None', label='Right RDR, beam 0')
                f1ax[6].plot(-180 * RR0_data["azimuth"] / np.pi, RR0_data["velocity"],
                             color=color_map_right(0.2), marker='o', ls='None', label='Right RDR, beam 0')
                f1ax[5].plot(-180 * RR0_data["azimuth"] / np.pi, RR0_data["range"],
                             color=color_map_right(0.2), marker='o', ls='None', label='Right RDR, beam 0')
                number_of_dets_right_processed += np.size(RR0_data["mcc"])

            if selection["beam_tp"].count(1):
                selection_tp = copy.deepcopy(selection)
                selection_tp["beam_tp"] = [1]
                RR1_data = lst_det_right.get_array_detections_selected(selection=selection_tp)
                f1ax[3].hist(RR1_data["velocity"], vel_range,
                             color=color_map_right(0.4), normed=1, label='beam 1')
                f1ax[0].plot(RR1_data["x"], RR1_data["y"],
                             color=color_map_right(0.4), marker='o', ls='None', label='Right RDR, beam 1')
                f1ax[6].plot(-180 * RR1_data["azimuth"] / np.pi, RR1_data["velocity"],
                             color=color_map_right(0.4), marker='o', ls='None', label='Right RDR, beam 1')
                f1ax[5].plot(-180 * RR1_data["azimuth"] / np.pi, RR1_data["range"],
                             color=color_map_right(0.4), marker='o', ls='None', label='Right RDR, beam 1')
                number_of_dets_right_processed += np.size(RR1_data["mcc"])

            if selection["beam_tp"].count(2):
                selection_tp = copy.deepcopy(selection)
                selection_tp["beam_tp"] = [2]
                RR2_data = lst_det_right.get_array_detections_selected(selection=selection_tp)
                f1ax[3].hist(RR2_data["velocity"], vel_range,
                             color=color_map_right(0.6), normed=1, label='beam 2')
                f1ax[0].plot(RR2_data["x"], RR2_data["y"],
                             color=color_map_right(0.6), marker='o', ls='None', label='Right RDR, beam 2')
                f1ax[6].plot(-180 * RR2_data["azimuth"] / np.pi, RR2_data["velocity"],
                             color=color_map_right(0.6), marker='o', ls='None', label='Right RDR, beam 2')
                f1ax[5].plot(-180 * RR2_data["azimuth"] / np.pi, RR2_data["range"],
                             color=color_map_right(0.6), marker='o', ls='None', label='Right RDR, beam 2')
                number_of_dets_right_processed += np.size(RR2_data["mcc"])

            if selection["beam_tp"].count(3):
                selection_tp = copy.deepcopy(selection)
                selection_tp["beam_tp"] = [3]
                RR3_data = lst_det_right.get_array_detections_selected(selection=selection_tp)
                f1ax[3].hist(RR3_data["velocity"], vel_range,
                             color=color_map_right(0.8), normed=1, label='beam 3')
                f1ax[0].plot(RR3_data["x"], RR3_data["y"],
                             color=color_map_right(0.8), marker='o', ls='None', label='Right RDR, beam 3')
                f1ax[6].plot(-180 * RR3_data["azimuth"] / np.pi, RR3_data["velocity"],
                             color=color_map_right(0.8), marker='o', ls='None', label='Right RDR, beam 3')
                f1ax[5].plot(-180 * RR3_data["azimuth"] / np.pi, RR3_data["range"],
                             color=color_map_right(0.8), marker='o', ls='None', label='Right RDR, beam 3')
                number_of_dets_right_processed += np.size(RR3_data["mcc"])

            plt.draw()
        else:
            RR_data_exists = False
    if LR_data_exists and RR_data_exists:
        lgd2 = f1ax[0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

        f1ax[0].set_xlabel('x [meters]')
        f1ax[0].set_ylabel('y [meters]')
        f1ax[1].set_ylabel('Left RDR, separate')
        f1ax[1].set_xlabel('Velocity $v(mcc)$ [m/s]')
        f1ax[2].set_ylabel('Left RDR, all')
        f1ax[3].set_ylabel('Right RDR, separate')
        f1ax[4].set_ylabel('Right RDR, all')
        f1ax[5].set_xlabel('Azimuth [deg]')
        f1ax[5].set_ylabel('Range [meters]')
        f1ax[6].set_ylabel('Velocity [km h$^{-1}$]')

        tit = "MCC: %08d Num of Det: L=%d R=%d In Selected Beams: L=%d R=%d" % (selection["mcc_tp"][0],
                                                                                number_of_dets_left,
                                                                                number_of_dets_right,
                                                                                number_of_dets_left_processed,
                                                                                number_of_dets_right_processed)
        f1.suptitle(tit, fontsize=14, fontweight='bold')

        if fname_det:
            f1.savefig(fname_det)
        else:
            plt.show()
    else:
        print("Nothing to plot for MCC:", selection["mcc_tp"])


def static_plot_selections(lst_det_left, lst_det_right, selection, fname_det):
    """
	Plots data in an analytic way. Complete set of detections for all MCCs is depicted here.
	Input data dictionaries:
		Both left and right radars of the same structure
		radar_data = { 	"range": ,
						"azimuth": , 
						"velocity": ,
						"x": ,
						"y": ,
						"beam": ,
						"mcc":  	}

	"""

    ###### Plot starts here:
    cms = matplotlib.cm
    color_map_left = cms.Blues
    color_map_right = cms.RdPu

    if fname_det:
        f1 = plt.figure(1, (23, 13), dpi=300)
    else:
        f1 = plt.figure(1, (15, 8))

    f1ax1 = f1.add_subplot(111)
    f1ax1.grid(True)
    plt.title('Detections', loc='left')

    number_of_dets_left_processed = 0
    number_of_dets_right_processed = 0
    #################### Left radar plot
    if lst_det_left:
        if selection["beam_tp"].count(0):
            selection_tp = copy.deepcopy(selection)
            selection_tp["beam_tp"] = [0]
            LR0_data = lst_det_left.get_array_detections_selected(selection=selection_tp)

            f1ax1.plot(LR0_data["x"], LR0_data["y"],
                       color=color_map_left(0.2), marker='o', ls='None', label='Left RDR, beam 0')
            number_of_dets_left_processed += np.size(LR0_data["mcc"])

        if selection["beam_tp"].count(1):
            selection_tp = copy.deepcopy(selection)
            selection_tp["beam_tp"] = [1]
            LR1_data = lst_det_left.get_array_detections_selected(selection=selection_tp)
            f1ax1.plot(LR1_data["x"], LR1_data["y"],
                       color=color_map_left(0.4), marker='o', ls='None', label='Left RDR, beam 1')
            number_of_dets_left_processed += np.size(LR1_data["mcc"])

        if selection["beam_tp"].count(2):
            selection_tp = copy.deepcopy(selection)
            selection_tp["beam_tp"] = [2]
            LR2_data = lst_det_left.get_array_detections_selected(selection=selection_tp)
            f1ax1.plot(LR2_data["x"], LR2_data["y"],
                       color=color_map_left(0.6), marker='o', ls='None', label='Left RDR, beam 2')
            number_of_dets_left_processed += np.size(LR2_data["mcc"])

        if selection["beam_tp"].count(3):
            selection_tp = copy.deepcopy(selection)
            selection_tp["beam_tp"] = [3]
            LR3_data = lst_det_left.get_array_detections_selected(selection=selection_tp)
            f1ax1.plot(LR3_data["x"], LR3_data["y"],
                       color=color_map_left(0.8), marker='o', ls='None', label='Left RDR, beam 3')
            number_of_dets_left_processed += np.size(LR3_data["mcc"])

        plt.draw()

    #################### Right radar plot
    if lst_det_right:

        if selection["beam_tp"].count(0):
            selection_tp = copy.deepcopy(selection)
            selection_tp["beam_tp"] = [0]
            RR0_data = lst_det_right.get_array_detections_selected(selection=selection_tp)
            f1ax1.plot(RR0_data["x"], RR0_data["y"],
                       color=color_map_right(0.2), marker='o', ls='None', label='Right RDR, beam 0')
            number_of_dets_right_processed += np.size(RR0_data["mcc"])

        if selection["beam_tp"].count(1):
            selection_tp = copy.deepcopy(selection)
            selection_tp["beam_tp"] = [1]
            RR1_data = lst_det_right.get_array_detections_selected(selection=selection_tp)
            f1ax1.plot(RR1_data["x"], RR1_data["y"],
                       color=color_map_right(0.4), marker='o', ls='None', label='Right RDR, beam 1')
            number_of_dets_right_processed += np.size(RR1_data["mcc"])

        if selection["beam_tp"].count(2):
            selection_tp = copy.deepcopy(selection)
            selection_tp["beam_tp"] = [2]
            RR2_data = lst_det_right.get_array_detections_selected(selection=selection_tp)
            f1ax1.plot(RR2_data["x"], RR2_data["y"],
                       color=color_map_right(0.6), marker='o', ls='None', label='Right RDR, beam 2')
            number_of_dets_right_processed += np.size(RR2_data["mcc"])

        if selection["beam_tp"].count(3):
            selection_tp = copy.deepcopy(selection)
            selection_tp["beam_tp"] = [3]
            RR3_data = lst_det_right.get_array_detections_selected(selection=selection_tp)
            f1ax1.plot(RR3_data["x"], RR3_data["y"],
                       color=color_map_right(0.8), marker='o', ls='None', label='Right RDR, beam 3')
            number_of_dets_right_processed += np.size(RR3_data["mcc"])

        plt.draw()

    lgd2 = f1ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

    f1ax1.set_xlabel('x [meters]')
    f1ax1.set_ylabel('y [meters]')

    tit = "In Selected Beams: L=%d R=%d" % (number_of_dets_left_processed, number_of_dets_right_processed)
    f1.suptitle(tit, fontsize=14, fontweight='bold')

    if fname_det:
        f1.savefig(fname_det)
    else:
        plt.show()


