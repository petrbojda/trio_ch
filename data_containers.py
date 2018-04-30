import scipy.io as sio
import numpy as np
import numpy.linalg as npla
from scipy.linalg import block_diag
import logging
import configparser
import argparse
import itertools
import copy
import heapq
import tracking_filters as tf
from utils import Q_discrete_white_noise


class DetectionPoint(object):
    def __init__(self, mcc=0, beam=0,
                 nodet_permcc=0, trackID=0, rng=0.0,
                 vel=0.0, azimuth=0.0, left=True, car_width=1.88):
        """ Creates a point as it was detected by the RADAR.

        :rtype: object

        :param mcc: The state of a Master-Clock-Cycle counter of the RADAR.
        :type mcc: int
        :param beam: Identifies to which beam the detection has been assigned by the RADAR.
        :type beam: int
        :param nodet_permcc: Number of detections for this particular MCC.
        :type nodet_permcc: int
        :param trackID: ID of the track to which the detection has been assigned if so
        :type trackID: int
        :param rng: Range of the detection measured by RADAR
        :type rng: float
        :param vel: Velocity of the detection measured by RADAR
        :type vel: float
        :param azimuth: Azimuth of the detection measured by RADAR
        :type azimuth: float
        :param left: TRUE if the detection was measured by the left RADAR, FALSE if by right one
        :type left: bool
        :param car_width: The width of an EGO car. (is being used to align left and right RADAR measurement)
        :type car_width: float
        """
        self._y_correction_dir = 1 if left else -1
        self._mcc = mcc
        self._beam = beam
        self._nodet = nodet_permcc
        self._trackID = trackID

        self._x = rng * np.cos(azimuth)
        self._y = self._y_correction_dir * (rng * np.sin(azimuth) + car_width / 2)
        self._azimuth = np.arctan(self._y/self._x)
        self._rng = np.sqrt(self._x**2 + self._y**2)
        self._vel = vel



    def set_XY (self,x,y):
        """ For an existing detection sets its _x and _y attributes independently

        :param x: value to assign to _x
        :param y: value to assign to _y
        :type x: float
        :type y: float
        """
        self._x = x
        self._y = y
        self._azimuth = np.arctan(self._y / self._x)
        self._rng = np.sqrt(self._x ** 2 + self._y ** 2)


    def set_XYvel(self, x, y, vel):
        self._x = x
        self._y = y
        self._azimuth = np.arctan(self._y / self._x)
        self._rng = np.sqrt(self._x ** 2 + self._y ** 2)
        self._vel = vel


    def equalsXY(self,detection_point):
        test = (self._x == detection_point._x) and (self._y == detection_point._y)
        # TODO: add calculations for range and azimuth
        return test

    def get_mcc(self):
        return self._mcc

    def get_xy_array(self):
        dx = self._vel * np.cos(self._azimuth)
        dy = self._vel * np.sin(self._azimuth)
        x = np.array([self._x, dx, self._y, dy])
        return x.reshape(4, 1)

    def test_in_range_of(self, detection, **kwargs):
        if 'dist' in kwargs:
            dx = detection._x - self._x
            dy = detection._y - self._y
            test_dist = kwargs['dist'] > npla.norm([dx,dy])
        else:
            logging.getLogger(__name__).critical(
                "DetectionPoint.test_in_range_of: distance from the det1 not defined, a criteria dist is always True")
            test_dist = True

        if 'vel' in kwargs:
            test_vel = detection._vel - kwargs['vel'] < self._vel < detection._vel + kwargs['vel']
        else:
            logging.getLogger(__name__).critical(
                "DetectionPoint.test_in_range_of: radar velocity of the det1 not defined, a criteria vel is always True")
            test_vel = True

        if 'az' in kwargs:
            test_az = detection._azimuth - kwargs['az'] < self._azimuth < detection._azimuth + kwargs['az']
        else:
            logging.getLogger(__name__).debug(
                "DetectionPoint.test_in_range_of: extent of azimuths is not defined, a criteria az is always True")
            test_az = True

        if 'beam' in kwargs:
            test_beam = (self._beam in kwargs['beam']) & (detection._beam in kwargs['beam'])
        else:
            logging.getLogger(__name__).debug(
                "DetectionPoint.test_in_range_of: extent of beams is not defined, a criteria beam is always True")
            test_beam = True

        return test_dist & test_vel &  test_az & test_beam




class ReferencePoint(object):
    def __init__(self, mccL=0, mccR=0, TAR_dist=0.0, TAR_distX=0.0, TAR_distY=0.0,
                 TAR_velX=0.0, TAR_velY=0.0, TAR_hdg=0.0,
                 EGO_velX=0.0, EGO_velY=0.0, EGO_accX=0.0, EGO_accY=0.0, EGO_hdg=0.0, ):
        """ Creates a point as delivered by a referential DGPS system.

        :rtype: object

        :param mccL: The state of a Master-Clock-Cycle counter of the left RADAR
        :type mccL: int
        :param mccR: The state of a Master-Clock-Cycle counter of the right RADAR
        :type mccR: int
        :param TAR_dist: Target vehicle's distance from the EGO car
        :type TAR_dist: float
        :param TAR_distX: Target vehicle's distance from the EGO car projected to the X axis
        :type TAR_distX: float
        :param TAR_distY: Target vehicle's distance from the EGO car projected to the Y axis
        :type TAR_distY: float
        :param TAR_velX: Target vehicle absolute velocity along the X axis
        :type TAR_velX: float
        :param TAR_velY: Target vehicle absolute velocity along the Y axis
        :type TAR_velY: float
        :param TAR_hdg: Target vehicle's heading
        :type TAR_hdg: float
        :param EGO_velX: EGO car's absolute velocity along the X axis
        :type EGO_velX: float
        :param EGO_velY: EGO car's absolute velocity along the Y axis
        :type EGO_velY: float
        :param EGO_accX: EGO car's absolute acceleration along the X axis
        :type EGO_accX: float
        :param EGO_accY: EGO car's absolute acceleration along the Y axis
        :type EGO_accY: float
        :param EGO_hdg: EGO car's heading
        :type EGO_hdg: float
        """

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
    def __init__(self, mcc=0, beam=[], x=0, y=0, dx=0, dy=0,
                 rvelocity=0, razimuth=0, rrange=0):
        """ Creates a point which is a part of the track.

        :rtype: object

        :param mcc: The state of a Master-Clock-Cycle counter of the RADAR.
        :type mcc: int
        :param beam: Identifies to which beam the detection has been assigned by the RADAR.
        :type beam: int
        :param x: X coordinate
        :type x: float
        :param y: Y coordinate
        :type  y: float
        :param dx: the time derivative of x
        :type dx: float
        :param dy: the time derivative of y
        :type dy: float
        :param rvelocity: an absolute velocity as measured by RADAR
        :type rvelocity: float
        :param razimuth: an azimuth as measured by RADAR
        :type razimuth: float
        """
        self.mcc = mcc
        self.x = x
        self.dx = dx
        self.y = y
        self.dy = dy
        self.beam = beam
        self.razimuth = razimuth
        self.rrange = rrange
        self.rvelocity = rvelocity

    def get_array(self):
        """ Returns x and y coordinates and their time derivatives in a 4-element numpy array

        :return: a numpy of the shape (4,1); np.array([x, dx, y, dy])
        """
        x = np.array([self.x, self.dx, self.y, self.dy])
        return x.reshape(4, 1)

    def get_xy_array(self):
        dx = self.rvelocity * np.cos(self.razimuth)
        dy = self.rvelocity * np.sin(self.razimuth)
        x = np.array([self.x,dx, self.y, dy])
        return x.reshape(4, 1)

    def get_z_array(self):
        z = np.array([self.x, self.y])
        return z.reshape(2, 1)

class Gate(object):
    def __init__(self, beam=[], x=0, y=0, diffx=0, diffy=0, dx=0, dy=0, diffdx=0, diffdy=0,
                 rvelocity=0, d_rvelocity = 0, razimuth=0, d_razimuth=0, rrange=0, d_rrange=0):
        self._x = x
        self._diff_x = diffx
        self._y = y
        self._diff_y = diffy
        self._dx = dx
        self._diff_dx = diffdx
        self._dy = dy
        self._diff_dy = diffdy
        self._beam = beam
        self._raz = razimuth
        self._diff_raz = d_razimuth
        self._rvel = rvelocity
        self._diff_rvel = d_rvelocity
        self._rrng = rrange
        self._diff_rrng = d_rrange
        logging.getLogger(__name__).debug("Gate.__init__: diff_x = %s, diff_y = %s",self._diff_x, self._diff_y)


    def set_center_point_from_det(self, detection):
        self._x = detection._x
        self._y = detection._y
        self._beam = detection._beam
        self._raz = detection._azimuth
        self._rrng = detection._rng
        self._rvel = detection._vel

    def test_detection_in_gate(self, detection, **kwargs):
        if kwargs:
            test_x = self._x - self._diff_x/2 < detection._x < self._x + self._diff_x/2 if 'x' in kwargs else True
            test_y = self._y - self._diff_y/2 < detection._y < self._y + self._diff_y/2 if 'y' in kwargs else True
            test_vel = self._rvel - self._diff_rvel/2 < detection._vel < self._x + self._diff_rvel/2 \
                if 'rvel' in kwargs else True
            test_rng = self._rrng - self._diff_rrng/2 < detection._rng < self._rrng + self._diff_rrng/2 \
                if 'rrng' in kwargs else True
            test_az = self._raz - self._diff_raz/2 < detection._azimuth < self._raz + self._diff_raz/2 \
                if 'rvel' in kwargs else True
            if 'beam' in kwargs:
                test_beam = detection._beam in self._beam
            else:
                test_beam = True

            return test_x & test_y & test_vel & test_rng & test_az & test_beam
        else:
            gate_x_min = self._x - self._diff_x/2
            gate_x_max = self._x + self._diff_x/2
            gate_y_min = self._y - self._diff_y/2
            gate_y_max = self._y + self._diff_y/2
            gate_rvel_min = self._rvel - self._diff_rvel / 2
            gate_rvel_max = self._rvel + self._diff_rvel / 2
            return  (gate_x_min < detection._x < gate_x_max) & \
                    (gate_y_min < detection._y < gate_y_max) & \
                    (gate_rvel_min < detection._vel < gate_rvel_max)

    def test_trackpoint_in_gate(self, tp, **kwargs):
        if kwargs:
            test_x = self._x - self._diff_x/2 < tp.x < self._x + self._diff_x/2 if 'x' in kwargs else True
            test_y = self._y - self._diff_y/2 < tp.y < self._y + self._diff_y/2 if 'y' in kwargs else True
            test_dx = self._dx - self._diff_dx/2 < tp.dx < self._dx + self._diff_dx/2 if 'dx' in kwargs else True
            test_dy = self._dy - self._diff_dy/2 < tp.dy < self._dy + self._diff_dy/2 if 'dy' in kwargs else True
            test_vel = self._rvel - self._diff_rvel/2 < tp.rvelocity < self._x + self._diff_rvel/2 \
                if 'rvel' in kwargs else True
            test_rng = self._rrng - self._diff_rrng/2 < tp.rrange < self._rrng + self._diff_rrng/2 \
                if 'rrng' in kwargs else True
            test_az = self._raz - self._diff_raz/2 < tp.razimuth < self._raz + self._diff_raz/2 \
                if 'rvel' in kwargs else True
            if 'beam' in kwargs:
                test_beam = tp.beam in self._beam
            else:
                test_beam = True

            return test_x & test_y & test_vel & test_rng & test_az & test_beam & test_dx & test_dy
        else:
            gate_x_min = self._x - self._diff_x/2
            gate_x_max = self._x + self._diff_x/2
            gate_y_min = self._y - self._diff_y/2
            gate_y_max = self._y + self._diff_y/2
            return  (gate_x_min < tp.x < gate_x_max) & \
                    (gate_y_min < tp.y < gate_y_max)

    def get_detection_dist_from_center(self, detection):
        c = np.array([self._x,self._y])
        d = np.array([detection._x, detection._y])
        diff = np.array([self._diff_x, self._diff_y])
        aim = (npla.norm(diff) - npla.norm(c-d)) / npla.norm(diff)
        return aim

    def get_trackpoint_dist_from_center(self, tp):
        c = np.array([self._x,self._y])
        d = np.array([tp.x, tp.y])
        diff = np.array([self._diff_x, self._diff_y])
        aim = (npla.norm(diff) - npla.norm(c-d)) / npla.norm(diff)
        return aim

    def get_center_array(self):
        xy = np.array([self._x, self._y])
        return xy.reshape(2, 1)



class DetectionList(list):
    def __init__(self):
        super().__init__()
        self._y_interval = (0, 0)
        self._x_interval = (0, 0)
        self._azimuth_interval = (0, 0)
        self._vel_interval = (0, 0)
        self._rng_interval = (0, 0)
        self._mcc_interval = (0, 0)
        self._trackID_interval = (0, 0)
        logging.getLogger(__name__).debug("DetectionList.__init__: list initialized")

    def append_detection(self, detection_point):
        self.append(detection_point)
        self.calculate_intervals()

    def append_data_from_m_file(self, data_path, left, car_width):
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
        self.calculate_intervals()
        logging.getLogger(__name__).debug("DetectionList.append_data_from_m_file: points appended = %s", no_d)

    def calculate_intervals(self):
        self._y_interval = (min([elem._y for elem in self]), max([elem._y for elem in self]))
        self._x_interval = (min([elem._x for elem in self]), max([elem._x for elem in self]))
        self._azimuth_interval = (min([elem._azimuth for elem in self]), max([elem._azimuth for elem in self]))
        self._vel_interval = (min([elem._vel for elem in self]), max([elem._vel for elem in self]))
        self._rng_interval = (min([elem._rng for elem in self]), max([elem._rng for elem in self]))
        self._mcc_interval = (min([elem._mcc for elem in self]), max([elem._mcc for elem in self]))
        return self._mcc_interval


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
            beam = [0, 1, 2, 3]

        if 'mcc' in kwarg:
            mcc_i = kwarg['mcc'] if (len(kwarg['mcc']) == 2) else (kwarg['mcc'], kwarg['mcc'])
        else:
            mcc_i = self._mcc_interval

        if 'x' in kwarg:
            x_i = kwarg['x'] if (len(kwarg['x']) == 2) else (kwarg['x'], kwarg['x'])
        else:
            x_i = self._x_interval

        if 'y' in kwarg:
            y_i = kwarg['y'] if (len(kwarg['y']) == 2) else (kwarg['y'], kwarg['y'])
        else:
            y_i = self._y_interval

        if 'rng' in kwarg:
            rng_i = kwarg['rng'] if (len(kwarg['rng']) == 2) else (kwarg['rng'], kwarg['rng'])
        else:
            rng_i = self._rng_interval

        if 'vel' in kwarg:
            vel_i = kwarg['vel'] if (len(kwarg['vel']) == 2) else (kwarg['vel'], kwarg['vel'])
        else:
            vel_i = self._vel_interval

        if 'az' in kwarg:
            az_i = kwarg['az'] if (len(kwarg['az']) == 2) else (kwarg['az'], kwarg['az'])
        else:
            az_i = self._azimuth_interval

        if 'trackID' in kwarg:
            trackID_i = kwarg['trackID'] if (len(kwarg['trackID']) == 2) else (kwarg['trackID'], kwarg['trackID'])
        else:
            trackID_i = self._trackID_interval

        if 'selection' in kwarg:
            beam = kwarg['selection']['beam_tp'] if kwarg['selection']['beam_tp'] else [0, 1, 2, 3]
            mcc_i = kwarg['selection']['mcc_tp'] if kwarg['selection']['mcc_tp'] else self._mcc_interval
            x_i = kwarg['selection']['x_tp'] if kwarg['selection']['x_tp'] else self._x_interval
            y_i = kwarg['selection']['y_tp'] if kwarg['selection']['y_tp'] else self._y_interval
            rng_i = kwarg['selection']['rng_tp'] if kwarg['selection']['rng_tp'] else self._rng_interval
            vel_i = kwarg['selection']['vel_tp'] if kwarg['selection']['vel_tp'] else self._vel_interval
            az_i = kwarg['selection']['az_tp'] if kwarg['selection']['az_tp'] else self._azimuth_interval
            trackID_i = kwarg['selection']['trackID_tp'] if kwarg['selection']['trackID_tp'] else self._trackID_interval

        r_sel = [elem._rng for elem in self if (elem._beam in beam and
                                                mcc_i[0] <= elem._mcc <= mcc_i[1] and
                                                trackID_i[0] <= elem._trackID <= trackID_i[1] and
                                                x_i[0] <= elem._x <= x_i[1] and
                                                y_i[0] <= elem._y <= y_i[1] and
                                                rng_i[0] <= elem._rng <= rng_i[1] and
                                                vel_i[0] <= elem._vel <= vel_i[1] and
                                                az_i[0] <= elem._azimuth <= az_i[1])]
        v_sel = [elem._vel for elem in self if (elem._beam in beam and
                                                mcc_i[0] <= elem._mcc <= mcc_i[1] and
                                                trackID_i[0] <= elem._trackID <= trackID_i[1] and
                                                x_i[0] <= elem._x <= x_i[1] and
                                                y_i[0] <= elem._y <= y_i[1] and
                                                rng_i[0] <= elem._rng <= rng_i[1] and
                                                vel_i[0] <= elem._vel <= vel_i[1] and
                                                az_i[0] <= elem._azimuth <= az_i[1])]
        az_sel = [elem._azimuth for elem in self if (elem._beam in beam and
                                                     mcc_i[0] <= elem._mcc <= mcc_i[1] and
                                                     trackID_i[0] <= elem._trackID <= trackID_i[1] and
                                                     x_i[0] <= elem._x <= x_i[1] and
                                                     y_i[0] <= elem._y <= y_i[1] and
                                                     rng_i[0] <= elem._rng <= rng_i[1] and
                                                     vel_i[0] <= elem._vel <= vel_i[1] and
                                                     az_i[0] <= elem._azimuth <= az_i[1])]
        mcc_sel = [elem._mcc for elem in self if (elem._beam in beam and
                                                  mcc_i[0] <= elem._mcc <= mcc_i[1] and
                                                  trackID_i[0] <= elem._trackID <= trackID_i[1] and
                                                  x_i[0] <= elem._x <= x_i[1] and
                                                  y_i[0] <= elem._y <= y_i[1] and
                                                  rng_i[0] <= elem._rng <= rng_i[1] and
                                                  vel_i[0] <= elem._vel <= vel_i[1] and
                                                  az_i[0] <= elem._azimuth <= az_i[1])]
        x_sel = [elem._x for elem in self if (elem._beam in beam and
                                              mcc_i[0] <= elem._mcc <= mcc_i[1] and
                                              trackID_i[0] <= elem._trackID <= trackID_i[1] and
                                              x_i[0] <= elem._x <= x_i[1] and
                                              y_i[0] <= elem._y <= y_i[1] and
                                              rng_i[0] <= elem._rng <= rng_i[1] and
                                              vel_i[0] <= elem._vel <= vel_i[1] and
                                              az_i[0] <= elem._azimuth <= az_i[1])]
        y_sel = [elem._y for elem in self if (elem._beam in beam and
                                              mcc_i[0] <= elem._mcc <= mcc_i[1] and
                                              trackID_i[0] <= elem._trackID <= trackID_i[1] and
                                              x_i[0] <= elem._x <= x_i[1] and
                                              y_i[0] <= elem._y <= y_i[1] and
                                              rng_i[0] <= elem._rng <= rng_i[1] and
                                              vel_i[0] <= elem._vel <= vel_i[1] and
                                              az_i[0] <= elem._azimuth <= az_i[1])]
        beam_sel = [elem._beam for elem in self if (elem._beam in beam and
                                                    mcc_i[0] <= elem._mcc <= mcc_i[1] and
                                                    trackID_i[0] <= elem._trackID <= trackID_i[1] and
                                                    x_i[0] <= elem._x <= x_i[1] and
                                                    y_i[0] <= elem._y <= y_i[1] and
                                                    rng_i[0] <= elem._rng <= rng_i[1] and
                                                    vel_i[0] <= elem._vel <= vel_i[1] and
                                                    az_i[0] <= elem._azimuth <= az_i[1])]
        trackID_sel = [elem._trackID for elem in self if (elem._beam in beam and
                                                          mcc_i[0] <= elem._mcc <= mcc_i[1] and
                                                          trackID_i[0] <= elem._trackID <= trackID_i[1] and
                                                          x_i[0] <= elem._x <= x_i[1] and
                                                          y_i[0] <= elem._y <= y_i[1] and
                                                          rng_i[0] <= elem._rng <= rng_i[1] and
                                                          vel_i[0] <= elem._vel <= vel_i[1] and
                                                          az_i[0] <= elem._azimuth <= az_i[1])]

        radar_data = {"range": np.array(r_sel),
                      "razimuth": np.array(az_sel),
                      "rvelocity": np.array(v_sel),
                      "x": np.array(x_sel),
                      "y": np.array(y_sel),
                      "trackID": np.array(trackID_sel),
                      "beam": np.array(beam_sel),
                      "mcc": np.array(mcc_sel)}
        logging.getLogger(__name__).debug("DetectionList.get_array_detections_selected: number of detections selected is %s MCCs from %s to %s",
                                          len(mcc_sel), min(mcc_sel), max(mcc_sel))
        return radar_data

    def get_array_detections(self):
        r_sel = [elem._rng for elem in self]
        v_sel = [elem._vel for elem in self]
        az_sel = [elem._azimuth for elem in self]
        mcc_sel = [elem._mcc for elem in self]
        x_sel = [elem._x for elem in self]
        y_sel = [elem._y for elem in self]
        beam_sel = [elem._beam for elem in self]
        trackID_sel = [elem._trackID for elem in self]

        radar_data = {"range": np.array(r_sel),
                      "razimuth": np.array(az_sel),
                      "rvelocity": np.array(v_sel),
                      "x": np.array(x_sel),
                      "y": np.array(y_sel),
                      "trackID": np.array(trackID_sel),
                      "beam": np.array(beam_sel),
                      "mcc": np.array(mcc_sel)}
        return radar_data

    def get_lst_detections_selected(self, **kwarg):
        if 'beam' in kwarg:
            beam = kwarg['beam']
        else:
            beam = [0, 1, 2, 3]

        if 'mcc' in kwarg:
            mcc_i = kwarg['mcc'] if (len(kwarg['mcc']) == 2) else (kwarg['mcc'], kwarg['mcc'])
        else:
            mcc_i = self._mcc_interval

        if 'x' in kwarg:
            x_i = kwarg['x'] if (len(kwarg['x']) == 2) else (kwarg['x'], kwarg['x'])
        else:
            x_i = self._x_interval

        if 'y' in kwarg:
            y_i = kwarg['y'] if (len(kwarg['y']) == 2) else (kwarg['y'], kwarg['y'])
        else:
            y_i = self._y_interval

        if 'rng' in kwarg:
            rng_i = kwarg['rng'] if (len(kwarg['rng']) == 2) else (kwarg['rng'], kwarg['rng'])
        else:
            rng_i = self._rng_interval

        if 'vel' in kwarg:
            vel_i = kwarg['vel'] if (len(kwarg['vel']) == 2) else (kwarg['vel'], kwarg['vel'])
        else:
            vel_i = self._vel_interval

        if 'az' in kwarg:
            az_i = kwarg['az'] if (len(kwarg['az']) == 2) else (kwarg['az'], kwarg['az'])
        else:
            az_i = self._azimuth_interval

        if 'trackID' in kwarg:
            trackID_i = kwarg['trackID'] if (len(kwarg['trackID']) == 2) else (kwarg['trackID'], kwarg['trackID'])
        else:
            trackID_i = self._trackID_interval

        if 'selection' in kwarg:
            beam = kwarg['selection']['beam_tp'] if kwarg['selection']['beam_tp'] else [0, 1, 2, 3]
            mcc_i = kwarg['selection']['mcc_tp'] if kwarg['selection']['mcc_tp'] else self._mcc_interval
            x_i = kwarg['selection']['x_tp'] if kwarg['selection']['x_tp'] else self._x_interval
            y_i = kwarg['selection']['y_tp'] if kwarg['selection']['y_tp'] else self._y_interval
            rng_i = kwarg['selection']['rng_tp'] if kwarg['selection']['rng_tp'] else self._rng_interval
            vel_i = kwarg['selection']['vel_tp'] if kwarg['selection']['vel_tp'] else self._vel_interval
            az_i = kwarg['selection']['az_tp'] if kwarg['selection']['az_tp'] else self._azimuth_interval
            trackID_i = kwarg['selection']['trackID_tp'] if kwarg['selection']['trackID_tp'] else self._trackID_interval

        lst_selected_detection = DetectionList()

        for elem in self:
            if (elem._beam in beam and
                            mcc_i[0] <= elem._mcc <= mcc_i[1] and
                            trackID_i[0] <= elem._trackID <= trackID_i[1] and
                            x_i[0] <= elem._x <= x_i[1] and
                            y_i[0] <= elem._y <= y_i[1] and
                            rng_i[0] <= elem._rng <= rng_i[1] and
                            vel_i[0] <= elem._vel <= vel_i[1] and
                            az_i[0] <= elem._azimuth <= az_i[1]):
                lst_selected_detection.append(elem)

        logging.getLogger(__name__).debug("DetectionList.get_lst_detections_selected: number of detections selected is %s MCCs from %s to %s",
                                          len(self), lst_selected_detection[0]._mcc, lst_selected_detection[-1]._mcc)
        return lst_selected_detection

    def extend_with_selection(self, radar_data_list, **kwarg):

        if 'beam' in kwarg:
            beam = kwarg['beam']
        else:
            beam = [0, 1, 2, 3]

        if 'mcc' in kwarg:
            mcc_i = kwarg['mcc'] if (len(kwarg['mcc']) == 2) else (kwarg['mcc'], kwarg['mcc'])
        else:
            mcc_i = radar_data_list._mcc_interval

        if 'x' in kwarg:
            x_i = kwarg['x'] if (len(kwarg['x']) == 2) else (kwarg['x'], kwarg['x'])
        else:
            x_i = radar_data_list._x_interval

        if 'y' in kwarg:
            y_i = kwarg['y'] if (len(kwarg['y']) == 2) else (kwarg['y'], kwarg['y'])
        else:
            y_i = radar_data_list._y_interval

        if 'rng' in kwarg:
            rng_i = kwarg['rng'] if (len(kwarg['rng']) == 2) else (kwarg['rng'], kwarg['rng'])
        else:
            rng_i = radar_data_list._rng_interval

        if 'vel' in kwarg:
            vel_i = kwarg['vel'] if (len(kwarg['vel']) == 2) else (kwarg['vel'], kwarg['vel'])
        else:
            vel_i = radar_data_list._vel_interval

        if 'az' in kwarg:
            az_i = kwarg['az'] if (len(kwarg['az']) == 2) else (kwarg['az'], kwarg['az'])
        else:
            az_i = radar_data_list._azimuth_interval

        if 'selection' in kwarg:
            beam = kwarg['selection']['beam_tp'] if kwarg['selection']['beam_tp'] else [0, 1, 2, 3]
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

        self._y_interval = (min([elem._y for elem in self]), max([elem._y for elem in self]))
        self._x_interval = (min([elem._x for elem in self]), max([elem._x for elem in self]))
        self._azimuth_interval = (min([elem._azimuth for elem in self]), max([elem._azimuth for elem in self]))
        self._vel_interval = (min([elem._vel for elem in self]), max([elem._vel for elem in self]))
        self._rng_interval = (min([elem._rng for elem in self]), max([elem._rng for elem in self]))
        self._mcc_interval = (min([elem._mcc for elem in self]), max([elem._mcc for elem in self]))


class UnAssignedDetectionList(DetectionList):
    def __init__(self, Tsampling, gate):
        """

        :param Tsampling:
        :param gate:
        """
        super().__init__()
        self._Tsampling = Tsampling
        self._lst_tracks_possible = []
        self._gate_pattern = Gate(beam=[], x=0, y=0, diffx=gate._diff_x, diffy=gate._diff_y, dx=0, dy=0,
                                  diffdx=0, diffdy=0, rvelocity=0, d_rvelocity = gate._diff_rvel, razimuth=0,
                                  d_razimuth=gate._diff_raz, rrange=0, d_rrange=gate._diff_rrng)
        logging.getLogger(__name__).debug("UnAssignedDetectionList.__init__: list initialized, gate: dim_x=%s, dim_y=%s",
                                          self._gate_pattern._diff_x, self._gate_pattern._diff_y)

    def two_point_projection(self, det1, det2):
        """ Extrapolates two detections in terms of the first order polynomial.
        The extrapolation is computed from x,y coordinates.

        :param start_detection: The first point of the extrapolation. X and Y coordinates are being used only.
        :param end_detection: The second point of the extrapolation. X and Y coordinates are being used only.
        :return: Projected point, X and Y coordinates are set only.

        :type start_detection: DetectionPoint
        :type end_detection: DetectionPoint
        :rtype: DetectionPoint
        """

        if det2.test_in_range_of (det1,dist=2,vel=.2):
            projected_point = DetectionPoint()
            x = 2 * det2._x - det1._x
            y = 2 * det2._y - det1._y
            vel = np.average([det1._vel, det2._vel])
            projected_point.set_XYvel(x,y,vel)
            logging.getLogger(__name__).debug(
                "UnAssignedDetectionList.two_point_projection: projected point exists at: x=%s, y=%s", x, y)
            return projected_point
        else:
            logging.getLogger(__name__).debug(
                "UnAssignedDetectionList.two_point_projection: projected point does not exist, det2 not in range of det1.")
            return False

    def test_det_in_gate_3points(self,detection, detection_1, detection_2):
        expected_point = self.two_point_projection(detection_1, detection_2)
        if expected_point:
            self._gate_pattern.set_center_point_from_det(expected_point)
            distance = self._gate_pattern.test_detection_in_gate(detection)
            return  {'det1':detection_1,'det2':detection_2,'det3':detection,'dist':distance}
        else:
            return False

    def new_detection(self, detection):
        """Tests whether or not the list of unassigned detections can form a new track.

        :param detection: The detection which is going to be tested.
        :type detection: DetectionPoint
        """
        logger = logging.getLogger(__name__)
        logger.info("UnAssignedDetectionList.new_detection: tested new detection with MCC: %s", detection._mcc)
        logger.debug("\t at x: %s, y: %s", detection._x, detection._y)
        aim = []
        if len(self)>1:
            for det1, det2 in itertools.combinations(self,2):
                logger.debug("UnAssignedDetectionList.new_detection: number of unassigned detections in a list: %s ",
                             len(self))
                logger.debug("\t\t\t\tcombining det1 %s, det2 %s", det1._mcc, det2._mcc)
                logger.debug("\t\t\t\t\t det1 x: %s, y: %s", det1._x, det1._y)
                logger.debug("\t\t\t\t\t det2 x: %s, y: %s", det2._x, det2._y)
                tri_combined = self.test_det_in_gate_3points(detection, det1, det2)
                if tri_combined:
                    aim.append(tri_combined)
                    logger.debug("\t Detection is in a gate with distance %s.", aim[-1]['dist'])
                else:
                    logger.debug("\t Detection doesn't fit within the gate.")
            if aim:
                logger.debug("UnAssignedDetectionList.new_detection: searching for the best fit.")
                logger.debug("\t\t\tThere is %s combinations where detection fit in a gate.",len(aim))
                max_aim = heapq.nlargest(1, aim, key=lambda s: s['dist'])
                logger.debug("\t maximum distance is %s.", max_aim['dist'])
                new_track = Track()
                new_track.append(max_aim['det1'])
                new_track.append(max_aim['det2'])
                new_track.append(max_aim['det3'])
                logger.debug("\t\t\ttherefore a new track will be created.")
                aim.clear()
                self.remove(max_aim('det1'))
                logger.debug("\t\t\tremoved det 1: %s.",max_aim('det1'))
                self.remove(max_aim('det2'))
                logger.debug("\t\t\tremoved det 2: %s.", max_aim('det2'))
                logger.info("UnAssignedDetectionList.new_detection: "
                            "Detection triggers a new track. %s detections remain in a list of unassigned.", len(self))
                return new_track
            else:
                self.append(detection)
                logger.info("UnAssignedDetectionList.new_detection: "
                            "Detection stored in an unassigned list. Now it contains: %s dets", len(self))
                return False
        else:
            self.append(detection)
            logger.info("UnAssignedDetectionList.new_detection:"
                        "Detection stored in an unassigned list. Now it contains: %s dets", len(self))
            return False


    def remove_detections_by_mcc(self, mcc_interval):
        mcc_i = mcc_interval if (len(mcc_interval) == 2) else (mcc_interval, mcc_interval)
        for elem in self:
            if mcc_i[0] <= elem._mcc <= mcc_i[1]:
                self.remove(elem)

    def remove_detection(self, detection):
        self.remove(detection)

class ReferenceList(list):
    def __init__(self):
        super().__init__()
        self._mccL_interval = (0, 0)
        self._mccR_interval = (0, 0)

    def append_from_m_file(self, data_path):
        DGPS_data = sio.loadmat(data_path)
        no_dL = len(DGPS_data["MCC_LeftRadar"])
        no_dR = len(DGPS_data["MCC_RightRadar"])
        no_d = max(no_dL, no_dR)
        logging.getLogger(__name__).debug("ReferenceList.append_from_m_file:  DGPSdata Left: %s",
                                          len(DGPS_data["MCC_LeftRadar"]))
        logging.getLogger(__name__).debug("ReferenceList.append_from_m_file:  DGPSdata Left list: %s",
                                          int(DGPS_data["MCC_LeftRadar"][20]))

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
        self._mccL_interval = (min([elem._mccL for elem in self]), max([elem._mccL for elem in self]))
        self._mccR_interval = (min([elem._mccR for elem in self]), max([elem._mccR for elem in self]))

    def get_mccL_interval(self):
        return self._mccL_interval

    def get_mccR_interval(self):
        return self._mccR_interval

    def get_mccB_interval(self):
        mcc_min = min(self._mccL_interval[0], self._mccR_interval[0])
        mcc_max = max(self._mccL_interval[1], self._mccR_interval[1])
        mccB = (mcc_min, mcc_max)
        return mccB

    def get_array_references_selected(self, **kwarg):
        if 'mccL' in kwarg:
            if kwarg['mccL']:
                mccL_i = kwarg['mccL'] if (len(kwarg['mccL']) == 2) else (kwarg['mccL'], kwarg['mccL'])
            else:
                mccL_i = self._mccL_interval
        else:
            mccL_i = self._mccL_interval

        if 'mccR' in kwarg:
            if kwarg['mccR']:
                mccR_i = kwarg['mccR'] if (len(kwarg['mccR']) == 2) else (kwarg['mccR'], kwarg['mccR'])
            else:
                mccR_i = self._mccR_interval
        else:
            mccR_i = self._mccR_interval

        mccL_sel = [elem._mccL for elem in self if (mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                    mccR_i[0] <= elem._mccR <= mccR_i[1])]
        mccR_sel = [elem._mccR for elem in self if (mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                    mccR_i[0] <= elem._mccR <= mccR_i[1])]
        TAR_dist_sel = [elem._TAR_dist for elem in self if (mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                            mccR_i[0] <= elem._mccR <= mccR_i[1])]
        TAR_distX_sel = [elem._TAR_distX for elem in self if (mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                              mccR_i[0] <= elem._mccR <= mccR_i[1])]
        TAR_distY_sel = [elem._TAR_distY for elem in self if (mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                              mccR_i[0] <= elem._mccR <= mccR_i[1])]
        TAR_velX_sel = [elem._TAR_velX for elem in self if (mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                            mccR_i[0] <= elem._mccR <= mccR_i[1])]
        TAR_velY_sel = [elem._TAR_velY for elem in self if (mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                            mccR_i[0] <= elem._mccR <= mccR_i[1])]
        TAR_hdg_sel = [elem._TAR_hdg for elem in self if (mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                          mccR_i[0] <= elem._mccR <= mccR_i[1])]
        EGO_velX_sel = [elem._EGO_velX for elem in self if (mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                            mccR_i[0] <= elem._mccR <= mccR_i[1])]
        EGO_velY_sel = [elem._EGO_velY for elem in self if (mccL_i[0] <= elem._mccL <= mccL_i[1] and
                                                            mccR_i[0] <= elem._mccR <= mccR_i[1])]
        EGO_accX_sel = [elem._EGO_accX for elem in self if
                        (mccL_i[0] <= elem._mccL <= mccL_i[1] and mccR_i[0] <= elem._mccR <= mccR_i[1])]
        EGO_accY_sel = [elem._EGO_accY for elem in self if
                        (mccL_i[0] <= elem._mccL <= mccL_i[1] and mccR_i[0] <= elem._mccR <= mccR_i[1])]
        EGO_hdg_sel = [elem._EGO_hdg for elem in self if
                       (mccL_i[0] <= elem._mccL <= mccL_i[1] and mccR_i[0] <= elem._mccR <= mccR_i[1])]
        DGPS_data = {"mccL": np.array(mccL_sel),
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
    def __init__(self, trackID):
        super().__init__()
        self._tracker = None
        self._predicted_gate = Gate(beam=[], x=0, y=0, diffx=2, diffy=2, dx=0, dy=0, diffdx=0, diffdy=0,
                 rvelocity=0, d_rvelocity = 0, razimuth=0, d_razimuth=0, rrange=0, d_rrange=0)
        self._trackID = trackID
        self._velx_interval = (0, 0)
        self._x_interval = (0, 0)
        self._vely_interval = (0, 0)
        self._y_interval = (0, 0)
        self._rvelocity_interval = (0, 0)
        self._razimuth_interval = (0, 0)
        self._rrange_interval = (0, 0)
        self._mcc_interval = (0, 0)
        self._last_update = None
        self._active = True

    def activate(self):
        self._active = True

    def deactivate(self):
        self._active = False

    def append_point(self, mcc, x, y, dx, dy, beam):
        self.append(TrackPoint(mcc, x, y, dx, dy, beam))
        self._y_interval = (min([elem.y for elem in self]), max([elem.y for elem in self]))
        self._x_interval = (min([elem.x for elem in self]), max([elem.x for elem in self]))
        self._vely_interval = (min([elem.vely for elem in self]), max([elem.vely for elem in self]))
        self._velx_interval = (min([elem.velx for elem in self]), max([elem.velx for elem in self]))
        self._rvelocity_interval = (min([elem.rvelocity for elem in self]), max([elem.rvelocity for elem in self]))
        self._razimuth_interval = (min([elem.razimuth for elem in self]), max([elem.razimuth for elem in self]))
        self._rrange_interval = (min([elem.rrange for elem in self]), max([elem.rrange for elem in self]))
        self._mcc_interval = (min([elem.mcc for elem in self]), max([elem.mcc for elem in self]))
        return self._trackID

    def append_detection(self, detection):
        self.append(TrackPoint(mcc=detection._mcc,
                               x=detection._x,
                               y=detection._y,
                               razimuth=detection._azimuth,
                               rvelocity=detection._vel,
                               beam=detection._beam))
        self._y_interval = (min([elem.y for elem in self]), max([elem.y for elem in self]))
        self._x_interval = (min([elem.x for elem in self]), max([elem.x for elem in self]))
        self._rvelocity_interval = (min([elem.rvelocity for elem in self]), max([elem.rvelocity for elem in self]))
        self._razimuth_interval = (min([elem.razimuth for elem in self]), max([elem.razimuth for elem in self]))
        self._rrange_interval = (min([elem.rrange for elem in self]), max([elem.rrange for elem in self]))
        self._mcc_interval = (min([elem.mcc for elem in self]), max([elem.mcc for elem in self]))
        return self._trackID

    def append_point_from_radardata_str(self, radardata):
        self.append(TrackPoint(mcc=radardata['mcc'],
                               x=radardata['x'],
                               y=radardata['y'],
                               razimuth=radardata['razimuth'],
                               rvelocity=radardata['rvelocity'],
                               beam=radardata['beam']))
        self._y_interval = (min([elem.y for elem in self]), max([elem.y for elem in self]))
        self._x_interval = (min([elem.x for elem in self]), max([elem.x for elem in self]))
        # self._vely_interval = (min([elem._vely for elem in self]),max([elem._vely for elem in self]))
        # self._velx_interval = (min([elem._velx for elem in self]),max([elem._velx for elem in self]))
        self._rvelocity_interval = (min([elem.rvelocity for elem in self]), max([elem.rvelocity for elem in self]))
        self._razimuth_interval = (min([elem.razimuth for elem in self]), max([elem.razimuth for elem in self]))
        self._rrange_interval = (min([elem.rrange for elem in self]), max([elem.rrange for elem in self]))
        self._mcc_interval = (min([elem.mcc for elem in self]), max([elem.mcc for elem in self]))
        return self._trackID

    def test_trackpoint_in_gate(self,tp):
        if self._predicted_gate.test_trackpoint_in_gate(tp):
            aim = self._predicted_gate.get_trackpoint_dist_from_center(tp)
        else:
            aim = 0
        return aim

    def test_detection_in_gate(self,detection):
        if self._predicted_gate.test_detection_in_gate(detection):
            aim = self._predicted_gate.get_detection_dist_from_center(detection)
        else:
            aim = 0
        return aim

    def init_tracker(self,type='kalman_filter', dim_x=4, dim_z=2, dt=50.0e-3, init_x=np.array([[0, 0, 0, 0]]).T):
        if not(self._tracker):
            self._tracker = tf.KalmanFilter(dim_x=dim_x, dim_z=dim_z)

            self._tracker.F = np.array([[1, dt, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, dt],
                                        [0, 0, 0, 1]])
            q = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
            self._tracker.Q = block_diag(q, q)
            self._tracker.x = init_x
            self._tracker.P = np.eye(4) * 5.0
            self._tracker.H = np.array([[1, 0, 0, 0],
                                        [0, 0, 1, 0]])
            self._tracker.R = np.array([[0.2, 0],[0, 0.1]])
            self._predicted_gate._x = self._tracker.x[0]
            self._predicted_gate._y = self._tracker.x[2]
            logging.getLogger(__name__).debug("Track.init_tracker: Tracker initialized with: ")
            logging.getLogger(__name__).debug("\t state vector x:\t %02.5f", self._tracker.x[0])
            logging.getLogger(__name__).debug("\t \t \t \t \t \t %02.5f", self._tracker.x[1])
            logging.getLogger(__name__).debug("\t \t \t \t \t \t %02.5f", self._tracker.x[2])
            logging.getLogger(__name__).debug("\t \t \t \t \t \t %02.5f", self._tracker.x[3])
            return True
        else:
            return False

    def start_tracker(self):
        self._tracker.update(self[1].get_z_array())
        self._tracker.predict()
        self._tracker.update(self[2].get_z_array())
        self._last_update = self[2].mcc
        self._predicted_gate._x = self._tracker.x[0]
        self._predicted_gate._y = self._tracker.x[2]
        logging.getLogger(__name__).debug("Track.start_tracker: Tracker started, current posteriori")
        logging.getLogger(__name__).debug(" \t\t x = %02.5f",
                                          self._predicted_gate.get_center_array()[0])
        logging.getLogger(__name__).debug(" \t\t y = %02.5f",
                                          self._predicted_gate.get_center_array()[1])

    def update_tracker(self):
        self._tracker.update(self[-1].get_z_array())
        self._last_update = self[-1].mcc
        self._predicted_gate._x = self._tracker.x[0]
        self._predicted_gate._y = self._tracker.x[2]
        logging.getLogger(__name__).debug("Track.update_tracker: Tracker's update cycle called, current posteriori")
        logging.getLogger(__name__).debug(" \t\t x = %02.5f",
                                          self._predicted_gate.get_center_array()[0])
        logging.getLogger(__name__).debug(" \t\t y = %02.5f",
                                          self._predicted_gate.get_center_array()[1])

    def predict(self):
        self._tracker.predict()
        self._predicted_gate._x = self._tracker.x[0]
        self._predicted_gate._y = self._tracker.x[2]
        logging.getLogger(__name__).debug("Track.predict: Tracker's predict cycle called, current apriori")
        logging.getLogger(__name__).debug(" \t\t x = %02.5f",
                                          self._predicted_gate.get_center_array()[0])
        logging.getLogger(__name__).debug(" \t\t y = %02.5f",
                                          self._predicted_gate.get_center_array()[1])

    def get_mcc_interval(self):
        return self._mcc_interval

    def get_ID(self):
        return self._trackID

    def get_predicted_gate(self):
        return self._predicted_gate

    def get_array_trackpoints(self):
        mcc_sel = [elem.mcc for elem in self]
        x_sel = [elem.x for elem in self]
        y_sel = [elem.y for elem in self]
        razimuth_sel = [elem.razimuth for elem in self]
        rvelocity_sel = [elem.rvelocity for elem in self]
        beam_sel = [elem.beam for elem in self]

        track_data = {"active":self._active,
                      "mcc": np.array(mcc_sel),
                      "razimuth": np.array(razimuth_sel),
                      "rvelocity": np.array(rvelocity_sel),
                      "x": np.array(x_sel),
                      "y": np.array(y_sel),
                      "beam": np.array(beam_sel)}
        return track_data

    def set_predicted_gate(self, predicted_gate):
        self._predicted_gate = copy.copy(predicted_gate)



def cnf_file_parser(cnf_file):
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

    # Read data-preprocessor settings
    radar_select = config.get('DataProcessSettings', 'radar')
    number_of_mcc = config.get('DataProcessSettings', 'number_of_mcc')

    data_preprocessor_settings = {
        "radar_select": radar_select,
        "number_of_mcc": number_of_mcc}

    return conf_data, data_preprocessor_settings


def cnf_datapaths_parser(cnf_file, scenario):
    config = configparser.ConfigParser()
    config.read(cnf_file)  # "./analysis.cnf"

    filename_LeftRadar = config.get(scenario, 'left_radar')
    filename_RightRadar = config.get(scenario, 'right_radar')
    filename_LeftDGPS = config.get(scenario, 'left_dgps')
    filename_RightDGPS = config.get(scenario, 'right_dgps')
    filename_BothDGPS = config.get(scenario, 'both_dgps')
    DGPS_xcompensation = config.get(scenario, 'DGPS_xcompensation')
    filename_logger_configuration = config.get('LOG_file', 'cfg_filename')

    data_filenames = {"filename_LeftRadar": filename_LeftRadar,
                      "filename_RightRadar": filename_RightRadar,
                      "filename_LeftDGPS": filename_LeftDGPS,
                      "filename_RightDGPS": filename_RightDGPS,
                      "filename_BothDGPS": filename_BothDGPS,
                      "filename_LOGcfg": filename_logger_configuration,
                      "DGPS_xcompensation": DGPS_xcompensation}
    return data_filenames


def parse_CMDLine(cnf_file):
    global path_data_folder
    conf_data, data_preprocessor_settings = cnf_file_parser(cnf_file)
    number_of_mcc_to_process = data_preprocessor_settings["number_of_mcc"]

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
    #      Ploting option
    parser.add_argument("-p", "--plot",
                        help="Set the plot options.")
    #      Select a scenario
    argv = parser.parse_args()

    if argv.beam:
        beams_tp = [int(s) for s in argv.beam.split(',')]
        beams_tp.sort()
    else:
        beams_tp = [0, 1, 2, 3]

    if argv.radar:
        radar_tp = argv.radar
    elif data_preprocessor_settings["radar_select"]:
        radar_tp = data_preprocessor_settings["radar_select"]
    else:
        radar_tp = "B"

    if argv.plot:
        plot_tp = argv.plot
    else:
        plot_tp = "all"

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

        data_filenames = cnf_datapaths_parser(cnf_file, argv.scenario)

        conf_data_out = {"scenario": argv.scenario,
                         "path_data_folder": path_data_folder,
                         "filename_LeftRadar": data_filenames["filename_LeftRadar"],
                         "filename_RightRadar": data_filenames["filename_RightRadar"],
                         "filename_LeftDGPS": data_filenames["filename_LeftDGPS"],
                         "filename_RightDGPS": data_filenames["filename_RightDGPS"],
                         "filename_BothDGPS": data_filenames["filename_BothDGPS"],
                         "filename_LOGcfg": data_filenames["filename_LOGcfg"],
                         "DGPS_xcompensation": data_filenames["DGPS_xcompensation"],
                         "EGO_car_width": conf_data["EGO_car_width"],
                         "beams_tp": beams_tp,
                         "radar_tp": radar_tp,
                         "plot_tp": plot_tp,
                         "output_folder": output,
                         "number_of_mcc_to_process": number_of_mcc_to_process}

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
