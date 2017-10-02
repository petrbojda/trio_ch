import numpy as np
import data_containers as dc
import scipy.linalg as linalg
from numpy import dot, zeros, eye, asarray
from utils import setter, setter_scalar, dot3, setter_1d


class TrackingFilter(object):

    def __init__(self, dim_x, dim_z, dim_u=0):
        """

        :param dim_x:
        :param dim_z:
        :param dim_u:
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self._x = zeros((dim_x,1)) # state
        self._P = eye(dim_x)       # uncertainty covariance
        self._B = 0                # control transition matrix
        self._F = 0                # state transition matrix
        self._R = eye(dim_z)       # state uncertainty
        self._Q = eye(dim_x)       # process uncertainty
        self._y = zeros((dim_z, 1))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

class ExtendedKalmanFilter(object):

    def __init__(self, dim_x, dim_z, dim_u=0):
        """

        :param dim_x:
        :param dim_z:
        :param dim_u:
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self._x = zeros((dim_x,1)) # state
        self._P = eye(dim_x)       # uncertainty covariance
        self._B = 0                # control transition matrix
        self._F = 0                # state transition matrix
        self._R = eye(dim_z)       # state uncertainty
        self._Q = eye(dim_x)       # process uncertainty
        self._y = zeros((dim_z, 1))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)


    def predict_update(self, z, HJacobian, Hx, args=(), hx_args=(), u=0):


        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        F = self._F
        B = self._B
        P = self._P
        Q = self._Q
        R = self._R
        x = self._x

        H = HJacobian(x, *args)

        # predict step
        x = dot(F, x) + dot(B, u)
        P = dot3(F, P, F.T) + Q

        # update step
        S = dot3(H, P, H.T) + R
        K = dot3(P, H.T, linalg.inv (S))

        self._x = x + dot(K, (z - Hx(x, *hx_args)))

        I_KH = self._I - dot(K, H)
        self._P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)


    def update(self, z, HJacobian, Hx, R=None, args=(), hx_args=(),
               residual=np.subtract):


        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        P = self._P
        if R is None:
            R = self._R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        x = self._x

        H = HJacobian(x, *args)

        S = dot3(H, P, H.T) + R
        K = dot3(P, H.T, linalg.inv (S))

        hx =  Hx(x, *hx_args)
        y = residual(z, hx)
        self._x = x + dot(K, y)

        I_KH = self._I - dot(K, H)
        self._P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)


    def predict_x(self, u=0):


        self._x = dot(self._F, self._x) + dot(self._B, u)


    def predict(self, u=0):


        self.predict_x(u)
        self._P = dot3(self._F, self._P, self._F.T) + self._Q


    @property
    def Q(self):
        """ Process uncertainty matrix"""
        return self._Q


    @Q.setter
    def Q(self, value):
        """ Process uncertainty matrix"""
        self._Q = setter_scalar(value, self.dim_x)


    @property
    def P(self):
        """ state covariance matrix"""
        return self._P


    @P.setter
    def P(self, value):
        """ state covariance matrix"""
        self._P = setter_scalar(value, self.dim_x)


    @property
    def R(self):
        """ measurement uncertainty"""
        return self._R


    @R.setter
    def R(self, value):
        """ measurement uncertainty"""
        self._R = setter_scalar(value, self.dim_z)


    @property
    def F(self):
        """State Transition matrix"""
        return self._F


    @F.setter
    def F(self, value):
        """State Transition matrix"""
        self._F = setter(value, self.dim_x, self.dim_x)


    @property
    def B(self):
        """ control transition matrix"""
        return self._B


    @B.setter
    def B(self, value):
        """ control transition matrix"""
        self._B = setter(value, self.dim_x, self.dim_u)


    @property
    def x(self):
        """ state estimate vector """
        return self._x

    @x.setter
    def x(self, value):
        """ state estimate vector """
        self._x = setter_1d(value, self.dim_x)

    @property
    def K(self):
        """ Kalman gain """
        return self._K

    @property
    def y(self):
        """ measurement residual (innovation) """
        return self._y

    @property
    def S(self):
        """ system uncertainty in measurement space """
        return self._S


class FadingKalmanFilter(object):

    def __init__(self, alpha, dim_x, dim_z, dim_u=0):


        assert alpha >= 1
        assert dim_x > 0
        assert dim_z > 0
        assert dim_u >= 0


        self.alpha_sq = alpha**2
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x,1)) # state
        self.P = eye(dim_x)       # uncertainty covariance
        self.Q = eye(dim_x)       # process uncertainty
        self.B = 0                # control transition matrix
        self.F = 0                # state transition matrix
        self.H = 0                 # Measurement function
        self.R = eye(dim_z)       # state uncertainty

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = 0 # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = 0 # system uncertainty in measurement space

        # identity matrix. Do not alter this.
        self.I = np.eye(dim_x)


    def update(self, z, R=None):


        if z is None:
            return

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        H = self.H
        P = self.P
        x = self.x

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, x)

        # S = HPH' + R
        # project system uncertainty into measurement space
        S = dot3(H, P, H.T) + R

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        K = dot3(P, H.T, linalg.inv(S))

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = x + dot(K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self.I - dot(K, H)
        self.P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)

        self.S = S
        self.K = K


    def predict(self, u=0):
       # x = Fx + Bu
        self.x = dot(self.F, self.x) + dot(self.B, u)

        # P = FPF' + Q
        self.P = self.alpha_sq * dot3(self.F, self.P, self.F.T) + self.Q


    def batch_filter(self, zs, Rs=None, update_first=False):
        n = np.size(zs,0)
        if Rs is None:
            Rs = [None]*n

        # mean estimates from Kalman Filter
        means   = zeros((n,self.dim_x,1))
        means_p = zeros((n,self.dim_x,1))

        # state covariances from Kalman Filter
        covariances   = zeros((n,self.dim_x,self.dim_x))
        covariances_p = zeros((n,self.dim_x,self.dim_x))

        if update_first:
            for i,(z,r) in enumerate(zip(zs,Rs)):
                self.update(z,r)
                means[i,:]         = self.x
                covariances[i,:,:] = self.P

                self.predict()
                means_p[i,:]         = self.x
                covariances_p[i,:,:] = self.P
        else:
            for i,(z,r) in enumerate(zip(zs,Rs)):
                self.predict()
                means_p[i,:]         = self.x
                covariances_p[i,:,:] = self.P

                self.update(z,r)
                means[i,:]         = self.x
                covariances[i,:,:] = self.P

        return (means, covariances, means_p, covariances_p)


    def get_prediction(self, u=0):


        x = dot(self.F, self.x) + dot(self.B, u)
        P = self.alpha_sq * dot3(self.F, self.P, self.F.T) + self.Q
        return (x, P)


    def residual_of(self, z):
        return z - dot(self.H, self.x)


    def measurement_of_state(self, x):

        return dot(self.H, x)
