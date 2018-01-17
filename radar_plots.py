#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid.axes_size as Size
from mpl_toolkits.axes_grid import Divider
import matplotlib
import copy

def static_plot_grid_hist_selections(lst_det_left, lst_det_right, selection, fname_det):
    # Plot starts here:
    cms = matplotlib.cm
    color_map_left = cms.Blues
    color_map_right = cms.RdPu

    if fname_det:
        f1 = plt.figure(1, (23, 13), dpi=1200)
    else:
        f1 = plt.figure(1, (15, 8))

    vel_range = range(-90, 70)

    f1.clf()
    # the rect parameter is ignored as we set the axes_locator
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

    # Left radar plot
    if lst_det_left:
        LR_data = lst_det_left.get_array_detections_selected(selection=selection)
        if LR_data["mcc"].any():
            LR_data_exists = True
            f1ax[2].hist(LR_data["rvelocity"], vel_range, color=color_map_left(0.4), normed=1)
            number_of_dets_left = np.size(LR_data["mcc"])

            if selection["beam_tp"].count(0):
                selection_tp = copy.deepcopy(selection)
                selection_tp["beam_tp"] = [0]
                LR0_data = lst_det_left.get_array_detections_selected(selection=selection_tp)
                f1ax[1].hist(LR0_data["rvelocity"], vel_range,
                             color=color_map_left(0.2), normed=1, label='beam 0')
                f1ax[0].plot(LR0_data["x"], LR0_data["y"],
                             color=color_map_left(0.2), marker='o', ls='None', label='Left RDR, beam 0')
                f1ax[6].plot(-180 * LR0_data["razimuth"] / np.pi, LR0_data["rvelocity"],
                             color=color_map_left(0.2), marker='o', ls='None', label='Left RDR, beam 0')
                f1ax[5].plot(-180 * LR0_data["razimuth"] / np.pi, LR0_data["range"],
                             color=color_map_left(0.2), marker='o', ls='None', label='Left RDR, beam 0')
                number_of_dets_left_processed += np.size(LR0_data["mcc"])

            if selection["beam_tp"].count(1):
                selection_tp = copy.deepcopy(selection)
                selection_tp["beam_tp"] = [1]
                LR1_data = lst_det_left.get_array_detections_selected(selection=selection_tp)
                f1ax[1].hist(LR1_data["rvelocity"], vel_range,
                             color=color_map_left(0.4), normed=1, label='beam 1')
                f1ax[0].plot(LR1_data["x"], LR1_data["y"],
                             color=color_map_left(0.4), marker='o', ls='None', label='Left RDR, beam 1')
                f1ax[6].plot(-180 * LR1_data["razimuth"] / np.pi, LR1_data["rvelocity"],
                             color=color_map_left(0.4), marker='o', ls='None', label='Left RDR, beam 1')
                f1ax[5].plot(-180 * LR1_data["razimuth"] / np.pi, LR1_data["range"],
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
                f1ax[1].hist(LR3_data["rvelocity"], vel_range,
                             color=color_map_left(0.8), normed=1, label='beam 3')
                f1ax[0].plot(LR3_data["x"], LR3_data["y"],
                             color=color_map_left(0.8), marker='o', ls='None', label='Left RDR, beam 3')
                f1ax[6].plot(-180 * LR3_data["razimuth"] / np.pi, LR3_data["rvelocity"],
                             color=color_map_left(0.8), marker='o', ls='None', label='Left RDR, beam 3')
                f1ax[5].plot(-180 * LR3_data["razimuth"] / np.pi, LR3_data["range"],
                             color=color_map_left(0.8), marker='o', ls='None', label='Left RDR, beam 3')
                number_of_dets_left_processed += np.size(LR3_data["mcc"])

            plt.draw()
        else:
            LR_data_exists = False

    # Right radar plot
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
            f1.savefig(fname_det, format='eps', dpi=1200)
        else:
            plt.show()
    else:
        print("Nothing to plot for MCC:", selection["mcc_tp"])


def static_plot_selections(lst_det_left, lst_det_right, selection, fname_det):
    # Plot starts here:
    cms = matplotlib.cm
    color_map_left = cms.Blues
    color_map_right = cms.RdPu

    if fname_det:
        f1 = plt.figure(1, (23, 13), dpi=1200)
    else:
        f1 = plt.figure(1, (15, 8))

    f1ax1 = f1.add_subplot(111)
    f1ax1.grid(True)
    f1ax1.axis([-40, 100, -80, 80])
    plt.title('Detections', loc='left')

    number_of_dets_left_processed = 0
    number_of_dets_right_processed = 0
    # Left radar plot
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

    # Right radar plot
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
        f1.savefig(fname_det, format='eps', dpi=1200)
    else:
        plt.show()


def static_plotREF_selections(lst_det_left, lst_det_right,
                              lst_ref_left, lst_ref_right, lst_ref_both,
                              selection, fname_det, DGPS_xcompensation):
    # Plot starts here:
    cms = matplotlib.cm
    color_map_left = cms.Blues
    color_map_right = cms.RdPu
    color_map_ref = cms.Greens

    if fname_det:
        f1 = plt.figure(1, (23, 13), dpi=1200)
    else:
        f1 = plt.figure(1, (15, 8))

    f1ax1 = f1.add_subplot(111)
    f1ax1.grid(True)
    f1ax1.axis([-40, 100, -80, 80])
    plt.title('Detections', loc='left')

    number_of_dets_left_processed = 0
    number_of_dets_right_processed = 0
    # Left radar plot
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

    # Right radar plot
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

    # Reference plot
    if lst_ref_left:
        mcc_tp = selection["mcc_tp"]

        DGPSLeft_data = lst_ref_left.get_array_references_selected(mccL=mcc_tp)
        f1ax1.plot(abs(DGPSLeft_data["TAR_distX"] + DGPS_xcompensation), DGPSLeft_data["TAR_distY"],
                   color=color_map_ref(0.7), marker='+', ls='None', label='Left DGPS')

        plt.draw()

    if lst_ref_right:
        mcc_tp = selection["mcc_tp"]

        DGPSRight_data = lst_ref_right.get_array_references_selected(mccL=mcc_tp)
        f1ax1.plot(abs(DGPSRight_data["TAR_distX"] + DGPS_xcompensation), DGPSRight_data["TAR_distY"],
                   color=color_map_ref(0.3), marker='+', ls='None', label='Right DGPS')

        plt.draw()

    if lst_ref_both:
        mcc_tp = selection["mcc_tp"]

        DGPSBoth_data = lst_ref_both.get_array_references_selected(mccL=mcc_tp)
        f1ax1.plot(abs(DGPSBoth_data["TAR_distX"] + DGPS_xcompensation), DGPSBoth_data["TAR_distY"],
                   color=color_map_ref(1.0), marker='+', ls='None', label='Both DGPS')

        plt.draw()

    # Legend and Title of the plot
    lgd2 = f1ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

    f1ax1.set_xlabel('x [meters]')
    f1ax1.set_ylabel('y [meters]')

    tit = "In Selected Beams: L=%d R=%d" % (number_of_dets_left_processed, number_of_dets_right_processed)
    f1.suptitle(tit, fontsize=14, fontweight='bold')

    if fname_det:
        f1.savefig(fname_det, format='eps', dpi=1200)
    else:
        plt.show()



def static_plotTrackMan_initialization(lst_detections, lst_not_assigned, new_track, list_of_tracks):
    # Plot starts here:
    cms = matplotlib.cm
    color_map_new_det = cms.Blues
    color_map_not_assig = cms.Greys
    color_map_newly_formed = cms.Oranges
    color_map_tracks = cms.Greens

    f1 = plt.figure(1, (15, 8))

    f1ax1 = f1.add_subplot(111)
    f1ax1.grid(True)
    f1ax1.axis([-40, 100, -80, 80])
    plt.title('Detections', loc='left')

    number_of_new_dets_processed = 0
    number_of_unassig_dets_processed = 0
    number_of_track_points = 0
    number_of_active_tracks = 0
    number_of_deactivated_tracks = 0
    # Newly Incoming Detection Plot
    if lst_detections:
        print("radar_plt: Plotting newly incoming detection list of the length:",len(lst_detections))
        new_data = lst_detections.get_array_detections()

        f1ax1.plot(new_data["x"], new_data["y"],
                   color=color_map_new_det(0.8), marker='^', ls='None', label='New Detection')
        number_of_new_dets_processed += np.size(new_data["mcc"])
        print("radar_plt: New detections: ", number_of_new_dets_processed, "detections. new_data has ", len(new_data),
              "detections", new_data)
        print("radar_plt: new_data[x]:", new_data["x"], "new_data[y]:", new_data["y"])

        plt.draw()

    # Unassigned Detections Plot
    if lst_not_assigned:
        print("radar_plt: Plotting unassigned detection list of the length:",len(lst_detections))
        unassig_data = lst_not_assigned.get_array_detections()

        f1ax1.plot(unassig_data["x"], unassig_data["y"],
                   color=color_map_not_assig(0.8), marker='+', ls='None', label='Unassigned Detection')
        number_of_unassig_dets_processed += np.size(unassig_data["mcc"])
        print("radar_plt: Unassigned detections: ", number_of_unassig_dets_processed, "detections. unassig_data has ", len(unassig_data),
              "detections", unassig_data)
        print("radar_plt: unassig_data[x]:", unassig_data["x"], "unassig_data[y]:", unassig_data["y"])

        plt.draw()

    if list_of_tracks:
        print("radar_plt: Plotting existing tracks:",len(list_of_tracks))

        for elem in list_of_tracks:
            if elem["active"]:
                brightness = 0.8
                number_of_active_tracks += 1
            else:
                brightness = 0.3
                number_of_deactivated_tracks += 1

            f1ax1.plot(elem["x"], elem["y"],
                   color=color_map_tracks(brightness), ls='-')
        plt.draw()
        print("radar_plt: active tracks:", number_of_active_tracks,
               "deactivated tracks:",number_of_deactivated_tracks)
    # # Newly Formed Track Plot
    if new_track:
        print("radar_plt: Plotting newly formed track of the length:",len(new_track))
        new_track_data = new_track.get_array_trackpoints()

        f1ax1.plot(new_track_data["x"], new_track_data["y"],
                   color=color_map_newly_formed(0.8), ls='-', label='New Track')
        number_of_track_points += np.size(new_track_data["mcc"])
        print("radar_plt: Processed ", number_of_unassig_dets_processed, "track points. new_track_data has ", len(new_track_data),
              "points", new_track_data)
        print("radar_plt: new_track_data[x]:", new_track_data["x"], "new_track_data[y]:", new_track_data["y"])

        plt.draw()





    # Legend and Title of the plot
    lgd2 = f1ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

    f1ax1.set_xlabel('x [meters]')
    f1ax1.set_ylabel('y [meters]')

    tit = "New Track Started: incoming dets =%d unassigned dets=%d" % (number_of_new_dets_processed, number_of_unassig_dets_processed)
    f1.suptitle(tit, fontsize=14, fontweight='bold')

    plt.show()
