from lime.fft import fft2
from lime.phys import sinc
from lime.style import set_style

import numpy as np
from numpy import sqrt, exp, pi
import matplotlib.pyplot as plt

def heaviside(x):
    """
    Heaviside function defined in a grid.
      returns 0 if x<=0, and 1 if x>0
    """
    x = np.asarray(x)
    y = np.zeros(x.shape)
    y[x > 0] = 1.0
    return y

class Biphoton:
    def __init__(self, omegap, bw, Te, phase_matching='sinc'):
        """
        Class for entangled photon pair.
        Parameters
        ----------
        omegap: float
            pump carrier frequency
        bw: float
            pump bandwidth
        phase_matching: str
            type of phase matching. Default is 'sinc'.
        """
        self.omegap = omegap
        self.pump_bandwidth = bw
        self.phase_matching = phase_matching
        self.signal_center_frequency = omegap/2.
        self.idler_center_frequency = omegap/2.
        self.entanglement_time = Te
        self.__jsa = None
        self.p = None
        self.q = None

    def pump(self, bandwidth):
        """
        pump pulse envelope
        Parameters
        ----------
        bandwidth

        Returns
        -------

        """
        alpha = np.sqrt(1. / (np.sqrt(2. * np.pi) * bandwidth)) * \
                np.exp(-(p + q) ** 2 / 4. / bandwidth ** 2)
        return alpha

    def set_grid(self, p, q):
        self.p = p
        self.q = q
        return

    def jsa(self):
        p = self.p
        q = self.q
        bw = self.pump_bandwidth

        self.__jsa = _jsa(p, q, bw, model=self.phase_matching,
                          Te=self.entanglement_time)
        return self.__jsa

    def detect(self):
        """
        two-photon detection amplitude in a temporal grid defined by
        the spectral grid.

        Returns
        -------
        t1: 1d array
        t2: 1d array
        d: detection amplitude in the temporal grid (t1, t2)

        """

        if self.__jsa is None:
            raise ValueError('Please call jsa() to compute the __jsa first.')

        bw = self.pump_bandwidth
        omega_s = self.signal_center_frequency
        omega_i = self.idler_center_frequency
        p = self.p
        q = self.q
        dp = p[1] - p[0]
        dq = q[1] - q[0]
        return _detection_amplitude(self.__jsa, omega_s, omega_i, dp, dq)

    def detect_si(self):
        pass

    def detect_is(self):
        pass

    def g2(self):
        pass

    def plt_jsa(self, fname=None):
        fig, ax = plt.subplots(figsize=(4,3))
        set_style(14)

        ax.imshow(self.__jsa, origin='lower')

        plt.show()

        if fname is not None:
            fig.savefig('__jsa.pdf')

        return ax

def _jsa(p, q, pump_bw, model='sinc', Te=None):
    '''
    Construct the joint spectral amplitude

    Parameters
    ----------
    p : 1d array
        signal frequency
    q : 1d array
        idler frequency
    pump_bw : float
        pump bandwidth
    sm : float
        1/entanglement time
    Te : float
        Entanglement time.

    Returns
    -------
    jsa : TYPE
        DESCRIPTION.

    '''
    P, Q = np.meshgrid(p, q)
    sigma_plus = pump_bw
    sigma_minus = 1./Te

    # pump envelope
    alpha = np.sqrt(1. / (np.sqrt(2. * np.pi) * sigma_plus)) * \
            np.exp(-(P + Q) ** 2 / 4. / sigma_plus ** 2)

    # phase-matching function

    if model == 'Gaussian':
        beta = np.sqrt(1. / np.sqrt(2. * np.pi) / sigma_minus) * \
          np.exp(-(P - Q) ** 2 / 4. / sigma_minus ** 2)

        jsa = sqrt(2) * alpha * beta

    elif model == 'sinc':

        beta =  sqrt(0.5 * Te / np.pi) * sinc(Te * (P - Q)/4.)

    # const =  np.trace(dag(f).dot(f))*dq*dp

        jsa = alpha * beta

    return jsa


def _detection_amplitude(jsa, omega1, omega2, dp, dq):
    '''
    Detection amplitude <0|E(t)E(t')|Phi>, t, t' are defined on a 2D grid used
    in the FFT, E(t) = Es(t) + Ei(t) is the total electric field operator.
    This contains two amplitudes corresponding to two different
    ordering of photon interactions
        <0|T Ei(t)Es(t')|Phi> + <0|T Es(t)Ei(t')|Phi>

    The t, t' are defined relative to t0, i.e, they are temporal durations from t0.

    Parameters
    ----------
    jsa : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    omega1 : float
        central frequency of signal beam
    omega2 : float
        central frequency of idler beam

    Returns
    -------
    d : TYPE
        DESCRIPTION.

    '''

    t1, t2, jta = fft2(jsa, dp, dq)

    dt2 = t2[1] - t2[0]

    T1, T2 = np.meshgrid(t1, t2)

    # detection amplitude d(t1, t2) ~ JTA(t2, t1)
    d = np.exp(-1j * omega2 * T1 - 1j * omega1 * T2) * \
        np.sqrt(omega1 * omega2) * jta.T +\
        np.exp(-1j * omega1 * T1 - 1j * omega2 * T2) * \
        np.sqrt(omega1 * omega2) * jta

 #   amp = np.einsum('ij, ij -> i', d, heaviside(T1 - T2) * \
 #                   np.exp(-1j * gap20 * (T1-T2))) * dt2

    return t1, t2, d


if __name__ == '__main__':
    from lime.units import au2ev, au2fs
    p = np.linspace(-2, 2, 128)/au2ev
    q = p
    epp = Biphoton(omegap=3/au2ev, bw=0.2/au2ev, Te=10/au2fs)
    epp.set_grid(p, q)
    JSA = epp.jsa()

    #epp.plt_jsa()
    #t1, t2, d = epp.detect()
    #print(d.shape)