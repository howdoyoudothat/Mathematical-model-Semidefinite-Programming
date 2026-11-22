"""Eight-element uniform linear array and real I/Q covariance SDP.

The complex baseband vector x in C^N is represented by z=[Re(x); Im(x)]
in R^(2N).  A real symmetric PSD covariance Q=E[zz.T] describes the
transmit signal.  Directional power satisfies
P(theta)=<B(theta),Q>, B=u u.T+v v.T,
u=[Re(a);Im(a)], v=[-Im(a);Re(a)].
"""
from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class Experiment:
    n_antennas: int = 8
    carrier_hz: float = 28e9
    speed_of_light: float = 3e8
    spacing_in_wavelengths: float = 0.5
    desired_angle_deg: float = 0.0
    desired_gain: float = 1.0
    sidelobe_limit: float = 0.025

    @property
    def wavelength_m(self):
        return self.speed_of_light / self.carrier_hz

    @property
    def spacing_m(self):
        return self.spacing_in_wavelengths * self.wavelength_m

    @property
    def signal_dim(self):
        return 2 * self.n_antennas

    @property
    def sidelobe_angles_deg(self):
        positive = [20.0 + 0.5 * k for k in range(11)]
        positive += [30., 35., 40., 45., 50., 60., 70., 80., 90.]
        return np.array(sorted([-x for x in positive] + positive), dtype=float)

    def steering(self, angle_deg: float) -> np.ndarray:
        n = np.arange(self.n_antennas)
        theta = np.deg2rad(angle_deg)
        return np.exp(2j * np.pi * self.spacing_in_wavelengths * n * np.sin(theta))

    def directional_matrix(self, angle_deg: float) -> np.ndarray:
        a = self.steering(angle_deg)
        u = np.r_[a.real, a.imag]
        v = np.r_[-a.imag, a.real]
        return np.outer(u, u) + np.outer(v, v)

    def directional_matrices(self) -> np.ndarray:
        return np.stack([self.directional_matrix(float(x)) for x in self.sidelobe_angles_deg])

    def dual_data(self):
        B0 = self.directional_matrix(self.desired_angle_deg)
        Bs = self.directional_matrices()
        D = np.concatenate([B0[None], -Bs], axis=0)
        c = np.r_[self.desired_gain, -self.sidelobe_limit * np.ones(Bs.shape[0])]
        return B0, Bs, D, c

    def power(self, Q: np.ndarray, angle_deg: float) -> float:
        return float(np.sum(self.directional_matrix(angle_deg) * Q))

    def powers(self, Q: np.ndarray, angles_deg: Sequence[float]) -> np.ndarray:
        return np.array([self.power(Q, float(a)) for a in angles_deg])

    def uniform_power(self, angles_deg: Sequence[float]) -> np.ndarray:
        a0 = self.steering(self.desired_angle_deg)
        w = a0 / self.n_antennas
        return np.array([abs(np.vdot(self.steering(float(a)), w))**2 for a in angles_deg])
