"""Simple pulse simulator (NumPy).

Provides a tiny QubitSim with a Gaussian pulse generator and a simple
Hadamard-like state maker that includes exponential T2 decay.
"""
# typing
from typing import Tuple, Optional, Union
import numpy as np
# optional plotting
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False


class QubitSim:
    def __init__(self, T2: float = 20e-6) -> None:
        """Create simulator.

        Args:
            T2: decoherence time in seconds.
        """
        self.T2 = float(T2)

    def gaussian_pulse(self, duration: float = 50e-9, samples: int = 101, sigma: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a Gaussian pulse with many configurable options.

        Parameters (choice): either supply `sample_rate` or `samples`.

        Args:
            duration: pulse duration in seconds.
            sample_rate: samples per second (if provided, overrides `samples`).
            samples: number of samples (used if sample_rate is None).
            amplitude: peak amplitude of envelope.
            sigma: Gaussian sigma in seconds. If None uses duration/6.
            center: center time (seconds). If None uses duration/2.
            truncation: truncate tails at ±n_sigma.
            carrier_freq: carrier frequency in Hz (0 => no carrier).
            phase: carrier phase in radians.
            normalize: if True, normalize envelope before applying amplitude.
            dc_offset: add DC offset to final waveform.
            drag_alpha: if non-zero, returns complex waveform with DRAG quadrature (alpha * -dA/dt).
            return_time: if True returns (t, wf) else returns wf only.

        Returns:
            (t, wf) or wf depending on `return_time`.
        """
        # NOTE: keep signature backward-compatible by inferring samples when called with old args
        raise NotImplementedError

    # backward-compatibility wrapper (keeps old signature behavior)
    def gaussian_pulse_simple(self, duration: float = 50e-9, samples: int = 101, sigma: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        return self.gaussian_pulse(duration=duration, samples=samples, sigma=sigma)

    def gaussian_pulse(self,
                       duration: float = 50e-9,
                       *,
                       sample_rate: Optional[float] = None,
                       samples: Optional[int] = None,
                       amplitude: float = 1.0,
                       sigma: Optional[float] = None,
                       center: Optional[float] = None,
                       truncation: float = 4.0,
                       carrier_freq: float = 0.0,
                       phase: float = 0.0,
                       normalize: bool = True,
                       dc_offset: float = 0.0,
                       drag_alpha: float = 0.0,
                       return_time: bool = True,
                       dtype: Union[type, str] = float,
                       ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Flexible Gaussian pulse generator.

        Either provide `sample_rate` (preferred) or `samples` to control sampling.
        """
        duration = float(duration)
        if center is None:
            center = duration / 2.0

        # determine samples
        if sample_rate is not None:
            samples = int(max(2, round(duration * float(sample_rate))))
        if samples is None:
            samples = 101
        samples = int(samples)

        # sigma default
        if sigma is None:
            sigma = duration / 6.0

        # time array (uniform sampling)
        t = np.linspace(0.0, duration, samples, endpoint=False)

        # envelope (Gaussian, centered at `center`)
        env = np.exp(-((t - float(center)) ** 2) / (2.0 * float(sigma) ** 2))

        # optional truncation window: zero out outside center +/- truncation*sigma
        if truncation is not None and truncation > 0:
            mask = np.abs(t - float(center)) <= (float(truncation) * float(sigma))
            env = env * mask.astype(float)

        # normalize envelope peak to 1 if requested
        if normalize:
            peak = np.max(np.abs(env))
            if peak > 0:
                env = env / peak

        # apply amplitude and dc offset
        if amplitude != 1.0:
            env = env * float(amplitude)
        if dc_offset != 0.0:
            env = env + float(dc_offset)

        # DRAG: create complex waveform if requested
        if abs(drag_alpha) > 1e-12:
            # derivative of envelope (finite difference)
            dt = duration / samples
            denv = np.gradient(env, dt)
            # DRAG quadrature (commonly -i * alpha * dA/dt)
            wf = env + 1j * ( - float(drag_alpha) * denv )
        else:
            wf = env

        # carrier modulation (if carrier_freq provided)
        if carrier_freq and abs(carrier_freq) > 0.0:
            carrier = np.cos(2.0 * np.pi * float(carrier_freq) * t + float(phase))
            # if waveform is complex, modulate both real and imag appropriately
            if np.iscomplexobj(wf):
                wf = wf * np.exp(1j * (2.0 * np.pi * float(carrier_freq) * t + float(phase)))
            else:
                wf = wf * carrier

        # cast dtype
        if dtype is not None and dtype is not float:
            try:
                wf = wf.astype(dtype)
            except Exception:
                pass

        if return_time:
            return t, wf
        return wf




def _demo() -> None:
    sim = QubitSim()
    # example: 1 GS/s sampling, 120 ns pulse, amplitude 0.8, carrier 30 MHz, DRAG alpha=0.2
    #t, pulse = sim.gaussian_pulse(duration=120e-9, sample_rate=1e9, amplitude=0.8,
    #                                sigma=18e-8, carrier_freq=30e6, phase=0.0,
    #                                drag_alpha=0.2, return_time=True)

    #t, pulse = sim.gaussian_pulse(duration=120e-9, sample_rate=1e9, amplitude=0.8, sigma=18e-9, drag_alpha=0.2)

    #t, pulse = sim.gaussian_pulse(duration=120e-9, sample_rate=1e9, amplitude=0.8, sigma=18e-9)

    #t, pulse = sim.gaussian_pulse(duration=50e-9, samples=100, return_time=True)
    t, pulse = sim.gaussian_pulse(duration=40e-9, sample_rate=2.4e9, amplitude=1.0, sigma=10e-9, carrier_freq=5e6, phase=0.0, drag_alpha=0.0)


    #print("t[0..4]:", np.round(t[:5], 12))
    print("pulse[0..4]:", np.round(pulse[:5], 6))

    # show a simple plot


    try:
        plt.figure(figsize=(8, 3))
        # plot real part if complex
        if np.iscomplexobj(pulse):
            plt.plot(t * 1e9, np.real(pulse), '-k', label='I (real)')
            plt.plot(t * 1e9, np.imag(pulse), '--r', label='Q (imag)')
            plt.legend()
        else:
            plt.plot(t * 1e9, pulse, '-k')
        plt.xlabel('Time (ns)')
        plt.ylabel('Amplitude')
        plt.title(f'Gaussian pulse (duration {t[-1]*1e9:.1f} ns)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print('Plot failed:', exc)

  


if __name__ == "__main__":
    _demo()

