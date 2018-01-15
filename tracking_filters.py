import math
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import data_containers as dc
import scipy.linalg as linalg
from numpy import dot, zeros, eye, asarray
from utils import setter, setter_scalar, dot3, setter_1d
from stats import logpdf


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

class KalmanFilter(object):

    def __init__(self, dim_x, dim_z, dim_u=0):
        assert dim_x > 0
        assert dim_z > 0
        assert dim_u >= 0

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x,1)) # state
        self.P = eye(dim_x)       # uncertainty covariance
        self.Q = eye(dim_x)       # process uncertainty
        self.B = 0                # control transition matrix
        self.F = 0                # state transition matrix
        self.H = 0                 # Measurement function
        self.R = eye(dim_z)        # state uncertainty
        self._alpha_sq = 1.        # fading memory control
        self.M = 0                 # process-measurement cross correlation

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = 0 # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z)) # system uncertainty

        # identity matrix. Do not alter this.
        self.I = np.eye(dim_x)


    def update(self, z, R=None, H=None):
        if z is None:
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        if H is None:
            H = self.H
        P = self.P
        x = self.x

        # handle special case: if z is in form [[z]] but x is not a column
        # vector dimensions will not match
        if x.ndim==1 and shape(z) == (1,1):
            z = z[0]

        if shape(z) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            z = np.asarray([z])

        # y = z - Hx
        # error (residual) between measurement and prediction
        Hx = dot(H, x)

        assert shape(Hx) == shape(z) or (shape(Hx) == (1,1) and shape(z) == (1,)), \
               'shape of z should be {}, but it is {}'.format(
               shape(Hx), shape(z))
        self.y = z - Hx

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot3(H, P, H.T) + R

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot3(P, H.T, linalg.inv(self.S))

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self.I - dot(self.K, H)
        self.P = dot3(I_KH, P, I_KH.T) + dot3(self.K, R, self.K.T)

        self.log_likelihood = logpdf(z, dot(H, x), self.S)


    def update_correlated(self, z, R=None, H=None):
        if z is None:
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        if H is None:
            H = self.H
        x = self.x
        P = self.P
        M = self.M

        # handle special case: if z is in form [[z]] but x is not a column
        # vector dimensions will not match
        if x.ndim==1 and shape(z) == (1,1):
            z = z[0]

        if shape(z) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            z = np.asarray([z])

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, x)

        # project system uncertainty into measurement space
        self.S = dot3(H, P, H.T) + dot(H, M) + dot(M.T, H.T) + R

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(dot(P, H.T) + M, linalg.inv(self.S))

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = x + dot(self.K, self.y)
        self.P = P - dot(self.K, dot(H, P) + M.T)

        # compute log likelihood
        self.log_likelihood = logpdf(z, dot(H, x), self.S)


    def test_matrix_dimensions(self, z=None, H=None, R=None, F=None, Q=None):
        if H is None: H = self.H
        if R is None: R = self.R
        if F is None: F = self.F
        if Q is None: Q = self.Q
        x = self.x
        P = self.P

        assert x.ndim == 1 or x.ndim == 2, \
                "x must have one or two dimensions, but has {}".format(
                x.ndim)

        if x.ndim == 1:
            assert x.shape[0] == self.dim_x, \
                   "Shape of x must be ({},{}), but is {}".format(
                   self.dim_x, 1, x.shape)
        else:
            assert x.shape == (self.dim_x, 1), \
                   "Shape of x must be ({},{}), but is {}".format(
                   self.dim_x, 1, x.shape)

        assert P.shape == (self.dim_x, self.dim_x), \
               "Shape of P must be ({},{}), but is {}".format(
               self.dim_x, self.dim_x, P.shape)

        assert Q.shape == (self.dim_x, self.dim_x), \
               "Shape of P must be ({},{}), but is {}".format(
               self.dim_x, self.dim_x, P.shape)

        assert F.shape == (self.dim_x, self.dim_x), \
               "Shape of F must be ({},{}), but is {}".format(
               self.dim_x, self.dim_x, F.shape)


        assert np.ndim(H) == 2, \
               "Shape of H must be (dim_z, {}), but is {}".format(
               P.shape[0], shape(H))

        assert H.shape[1] == P.shape[0], \
               "Shape of H must be (dim_z, {}), but is {}".format(
               P.shape[0], H.shape)

        # shape of R must be the same as HPH'
        hph_shape = (H.shape[0], H.shape[0])
        r_shape = shape(R)

        if H.shape[0] == 1:
            # r can be scalar, 1D, or 2D in this case
            assert r_shape == () or r_shape == (1,) or r_shape == (1,1), \
            "R must be scalar or one element array, but is shaped {}".format(
            r_shape)
        else:
            assert r_shape == hph_shape, \
            "shape of R should be {} but it is {}".format(hph_shape, r_shape)


        if z is not None:
            z_shape = shape(z)
        else:
            z_shape = (self.dim_z, 1)

        # H@x must have shape of z
        Hx = dot(H, x)

        if z_shape == (): # scalar or np.array(scalar)
            assert Hx.ndim == 1 or shape(Hx) == (1,1), \
            "shape of z should be {}, not {} for the given H".format(
                   shape(Hx), z_shape)

        elif shape(Hx) == (1,):
            assert z_shape[0] == 1, 'Shape of z must be {} for the given H'.format(shape(Hx))

        else:
            assert (z_shape == shape(Hx) or
                    (len(z_shape) == 1 and shape(Hx) == (z_shape[0], 1))), \
                    "shape of z should be {}, not {} for the given H".format(
                    shape(Hx), z_shape)

        if np.ndim(Hx) > 1 and shape(Hx) != (1,1):
            assert shape(Hx) == z_shape, \
               'shape of z should be {} for the given H, but it is {}'.format(
               shape(Hx), z_shape)


    def predict(self, u=0, B=None, F=None, Q=None):
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        # x = Fx + Bu
        self.x = dot(F, self.x) + dot(B, u)

        # P = FPF' + Q
        self.P = self._alpha_sq * dot3(F, self.P, F.T) + Q


    def batch_filter(self, zs, Fs=None, Qs=None, Hs=None, Rs=None, Bs=None, us=None, update_first=False):
        n = np.size(zs,0)
        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n
        if Hs is None:
            Hs = [self.H] * n
        if Rs is None:
            Rs = [self.R] * n
        if Bs is None:
            Bs = [self.B] * n
        if us is None:
            us = [0] * n

        if len(Fs) < n: Fs = [Fs]*n
        if len(Qs) < n: Qs = [Qs]*n
        if len(Hs) < n: Hs = [Hs]*n
        if len(Rs) < n: Rs = [Rs]*n
        if len(Bs) < n: Bs = [Bs]*n
        if len(us) < n: us = [us]*n


        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means   = zeros((n, self.dim_x))
            means_p = zeros((n, self.dim_x))
        else:
            means   = zeros((n, self.dim_x, 1))
            means_p = zeros((n, self.dim_x, 1))

        # state covariances from Kalman Filter
        covariances   = zeros((n, self.dim_x, self.dim_x))
        covariances_p = zeros((n, self.dim_x, self.dim_x))

        if update_first:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.update(z, R=R, H=H)
                means[i,:]         = self.x
                covariances[i,:,:] = self.P

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i,:]         = self.x
                covariances_p[i,:,:] = self.P
        else:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i,:]         = self.x
                covariances_p[i,:,:] = self.P

                self.update(z, R=R, H=H)
                means[i,:]         = self.x
                covariances[i,:,:] = self.P

        return (means, covariances, means_p, covariances_p)



    def rts_smoother(self, Xs, Ps, Fs=None, Qs=None):
        assert len(Xs) == len(Ps)
        shape = Xs.shape
        n = shape[0]
        dim_x = shape[1]

        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n

        # smoother gain
        K = zeros((n,dim_x,dim_x))

        x, P = Xs.copy(), Ps.copy()

        for k in range(n-2,-1,-1):
            P_pred = dot3(Fs[k+1], P[k], Fs[k+1].T) + Qs[k+1]

            K[k]  = dot3(P[k], Fs[k+1].T, linalg.inv(P_pred))
            x[k] += dot(K[k], x[k+1] - dot(Fs[k+1], x[k]))
            P[k] += dot3(K[k], P[k+1] - P_pred, K[k].T)

        return (x, P, K)


    def get_prediction(self, u=0):
        x = dot(self.F, self.x) + dot(self.B, u)
        P = self._alpha_sq * dot3(self.F, self.P, self.F.T) + self.Q
        return (x, P)


    def residual_of(self, z):
        return z - dot(self.H, self.x)


    def measurement_of_state(self, x):
        return dot(self.H, x)


    @property
    def alpha(self):
        return self._alpha_sq**.5


    @property
    def likelihood(self):
        return math.exp(self.log_likelihood)


    @alpha.setter
    def alpha(self, value):
        assert np.isscalar(value)
        assert value > 0

        self._alpha_sq = value**2
