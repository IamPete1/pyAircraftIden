# This program is public domain
# Authors: Paul Kienzle, Nadav Horesh
"""
Chirp z-transform.
We provide two interfaces to the chirp z transform, an object interface
which precalculates part of the transform and can be applied efficiently
to many different data sets and a functional interface which is applied
only to the given data set.
Transforms
----------
CZT: callable (x,axis=-1)->array
   define a chirp-z transform that can be applied to different signals
ZoomFFT: callable (x,axis=-1)->array
   define a Fourier transform on a range of frequencies
ScaledFFT: callable (x,axis=-1)->array
   define a limited frequency FFT
Functions
---------
czt: array
   compute the chirp-z transform for a signal
zoomfft: array
   compute the Fourier transform on a range of frequencies
scaledfft: array
   compute a limited frequency FFT for a signal
"""
__all__ = ['czt', 'zoomfft', 'scaledfft', 'CZT', 'ZoomFFT', 'ScaledFFT']

import math, cmath

import numpy as np
from numpy import pi, arange
from scipy.fftpack import fft, ifft, fftshift


class CZT:
    """
    Chirp-Z Transform.
    Transform to compute the frequency response around a spiral.
    Objects of this class are callables which can compute the
    chirp-z transform on their inputs.  This object precalculates
    constants for the given transform.
    If w does not lie on the unit circle, then the transform will be
    around a spiral with exponentially increasing radius.  Regardless,
    angle will increase linearly.
    The chirp-z transform can be faster than an equivalent fft with 
    zero padding.  Try it with your own array sizes to see.  It is 
    theoretically faster for large prime fourier transforms, but not 
    in practice.
    The chirp-z transform is considerably less precise than the
    equivalent zero-padded FFT, with differences on the order of 1e-7
    from the direct transform rather than the on the order of 1e-15 as 
    seen with zero-padding.
    See zoomfft for a friendlier interface to partial fft calculations.
    """

    def __init__(self, n, m=None, w=1, a=1):
        """
        Chirp-Z transform definition.
        Parameters:
        ----------
        n: int
          The size of the signal
        m: int
          The number of points desired.  The default is the length of
          the input data.
        a: complex
          The starting point in the complex plane.  The default is 1.
        w: complex or float
          If w is complex, it is the ratio between points in each step.
          If w is float, it serves as a frequency scaling factor. for instance 
          when assigning w=0.5, the result FT will span half of frequncy range 
          (that fft would result) at half of the frequncy step size.
        Returns:
        --------
        CZT:
          callable object f(x,axis=-1) for computing the chirp-z transform on x
        """
        if m is None:
            m = n
        if w is None:
            w = cmath.exp(-1j * pi / m)
        elif type(w) in (float, int):
            w = cmath.exp(-2j * pi / m * w)
        self.w, self.a = w, a
        self.m, self.n = m, n

        k = arange(max(m, n))
        wk2 = w ** (k ** 2 / 2.)
        nfft = 2 ** nextpow2(n + m - 1)
        self._Awk2 = (a ** -k * wk2)[:n]
        self._nfft = nfft
        self._Fwk2 = fft(1 / np.hstack((wk2[n - 1:0:-1], wk2[:m])), nfft)
        self._wk2 = wk2[:m]
        self._yidx = slice(n - 1, n + m - 1)

    def __call__(self, x, axis=-1):
        """
        Parameters:
        ----------
        x: array
          The signal to transform.
        axis: int
          Array dimension to operate over.  The default is the final 
          dimension.
        Returns:
        -------
          An array of the same dimensions as x, but with the length of the
          transformed axis set to m.  Note that this is a view on a much
          larger array.
        """
        x = np.asarray(x)
        if x.shape[axis] != self.n:
            raise ValueError("CZT defined for length %d, not %d" %
                             (self.n, x.shape[axis]))
        # Calculate transpose coordinates, to allow operation on any given axis
        trnsp = np.arange(x.ndim)
        trnsp[[axis, -1]] = [-1, axis]
        x = x.transpose(*trnsp)
        y = ifft(self._Fwk2 * fft(x * self._Awk2, self._nfft))
        y = y[..., self._yidx] * self._wk2
        return y.transpose(*trnsp)


def nextpow2(n):
    """
    Return the smallest power of two greater than or equal to n.
    """
    return int(math.ceil(math.log(n) / math.log(2)))


class ZoomFFT(CZT):
    """
    Zoom FFT transform.
    This is a specialization of the chirp Z transform, CZT for a set of
    equally spaced frequencies.
    """

    def __init__(self, n, f1, f2=None, m=None, Fs=2):
        """
        Zoom FFT transform.
        Defines a Fourier transform for a set of equally spaced frequencies.
        Parameters:
        ----------
        n: int
          size of the signal
        m: int
          size of the output
        f1, f2: float
          start and end frequencies; if f2 is not specified, use 0 to f1
        Fs: float
          sampling frequency (default=2)
        Returns:
        -------
        A ZoomFFT instance
          A callable object f(x,axis=-1) for computing the zoom FFT on x.
        Sampling frequency is 1/dt, the time step between samples in the
        signal x.  The unit circle corresponds to frequencies from 0 up
        to the sampling frequency.  The default sampling frequency of 2
        means that f1,f2 values up to the Nyquist frequency are in the 
        range [0,1). For f1,f2 values expressed in radians, a sampling 
        frequency of 1/pi should be used.
        To plot the transform results use something like the following:
            t = transform(len(x), f1, f2, m)
            f = linspace(f1, f2, m)
            y = t(x)
            plot(f,y)
        """
        if m is None: m = n
        if f2 is None: f1, f2 = 0., f1
        w = cmath.exp(-2j * pi * (f2 - f1) / ((m - 1) * Fs))
        a = cmath.exp(2j * pi * f1 / Fs)
        CZT.__init__(self, n, m=m, w=w, a=a)
        self.f1, self.f2, self.Fs = f1, f2, Fs


class ScaledFFT(CZT):
    def __init__(self, n, m=None, scale=1.0):
        """
        Scaled fft transform.
        Similar to fft, where the frequency range is scaled and divided 
        into m-1 equal steps.  Like the FFT, frequencies are arranged from 
        0 to scale*Fs/2-delta followed by -scale*Fs/2 to -delta, where delta 
        is the step size scale*Fs/m for sampling frequence Fs. The intended 
        use is in a convolution of two signals, each has its own sampling step.
        This is equivalent to:
            fftshift(zoomfft(x, -scale, scale*(m-2.)/m, m=m))
        For example:
            m,n = 10,len(x)
            sf = ScaledFFT(n, m=m, scale=0.25)
            X = fftshift(fft(x))
            W = linspace(-8, 8*(n-2.)/n, n)
            SX = fftshift(sf(x))
            SW = linspace(-2, 2*(m-2.)/m, m)
            plot(X,W,SX,SW)
        Parameters:
        ----------
        n: int
          Size of the signal
        m: int
          The size of the output.
          Default: m=n
        scale: float
          Frequency scaling factor.
          Default: scale=1.0
        Returns:
        -------
        callable f(x,axis=-1)
          function for computing the scaled FFT on x.
        """
        if m is None:
            m = n
        w = np.exp(-2j * pi / m * scale)
        a = w ** ((m + 1) // 2)
        CZT.__init__(self, n=n, m=m, a=a, w=w)
        self.scale = scale

    def __call__(self, x, axis=-1):
        return fftshift(CZT.__call__(self, x, axis), axes=(axis,))

    __call__.__doc__ = CZT.__call__.__doc__


def scaledfft(x, m=None, scale=1.0, axis=-1):
    """
    Limited frequency FFT.
    See ScaledFFT doc for details
    Parameters:
    ----------
    x:   input array
    m:   int
      The length of the output signal
    scale: float
      A frequency scaling factor
    axis: int
      The array dimension to operate over.  The default is the
      final dimension.
    Returns:
    -------
      An array of the same rank of 'x', but with the size if 
      the 'axis' dimension set to 'm'    
    """
    transform = ScaledFFT(x.shape[axis], m, scale)
    return transform(x, axis)


def czt(x, m=None, w=1.0, a=1, axis=-1):
    """
    Compute the frequency response around a spiral.
    Parameters:
    ----------
    x: array
      The set of data to transform.
    m: int
      The number of points desired. Default is the length of the input data.
    a: complex
      The starting point in the complex plane.  Default is 1.
    w: complex or float
      If w is complex, it is the ratio between points in each step.
      If w is float, it is the frequency step scale (relative to the 
      normal dft frquency step).
    axis: int
      Array dimension to operate over.  Default is the final dimension.
    Returns:
    -------
      An array of the same dimensions as x, but with the length of the
      transformed axis set to m.  Note that this is a view on a much
      larger array.  To save space, you may want to call it as
      y = ascontiguousarray(czt(x))
    See zoomfft for a friendlier interface to partial fft calculations.
    If the transform needs to be repeated, use CZT to construct a 
    specialized transform function which can be reused without 
    recomputing constants. 
    """
    x = np.asarray(x)
    transform = CZT(x.shape[axis], m=m, w=w, a=a)
    return transform(x, axis=axis)


def zoomfft(x, f1, f2=None, m=None, Fs=2, axis=-1):
    """
    Compute the Fourier transform of x for frequencies in [f1, f2].
    Parameters:
    ----------
    m: int
      The number of points to evaluate.  The default is the length of x.
    f1, f2: float
      The frequency range. If f2 is not specified, the range 0-f1 is assumed.
    Fs: float
      The sampling frequency.  With a sampling frequency of
      10kHz for example, the range f1 and f2 can be expressed in kHz.
      The default sampling frequency is 2, so f1 and f2 should be 
      in the range 0,1 to keep the transform below the Nyquist
      frequency.
    x : array
      The input signal.
    axis: int
      The array dimension the transform operates over.  The default is the
      final dimension.
    Returns:
    -------
    array
      The transformed signal.  The fourier transform will be calculate
      at the points f1, f1+df, f1+2df, ..., f2, where df=(f2-f1)/m.
    zoomfft(x,0,2-2./len(x)) is equivalent to fft(x).
    To graph the magnitude of the resulting transform, use::
	plot(linspace(f1,f2,m), abs(zoomfft(x,f1,f2,m))).
    If the transform needs to be repeated, use ZoomFFT to construct 
    a specialized transform function which can be reused without 
    recomputing constants.
    """
    x = np.asarray(x)
    transform = ZoomFFT(x.shape[axis], f1, f2=f2, m=m, Fs=Fs)
    return transform(x, axis=axis)


def _test1(x, show=False, plots=[1, 2, 3, 4]):
    norm = np.linalg.norm

    # Normal fft and zero-padded fft equivalent to 10x oversampling
    over = 10
    w = np.linspace(0, 2 - 2. / len(x), len(x))
    y = fft(x)
    wover = np.linspace(0, 2 - 2. / (over * len(x)), over * len(x))
    yover = fft(x, over * len(x))

    # Check that zoomfft is the equivalent of fft
    y1 = zoomfft(x, 0, 2 - 2. / len(y))

    # Check that zoomfft with oversampling is equivalent to zero padding
    y2 = zoomfft(x, 0, 2 - 2. / len(yover), m=len(yover))

    # Check that zoomfft works on a subrange
    f1, f2 = w[3], w[6]
    y3 = zoomfft(x, f1, f2, m=3 * over + 1)
    w3 = np.linspace(f1, f2, len(y3))
    idx3 = slice(3 * over, 6 * over + 1)

    if not show: plots = []
    if plots != []:
        import pylab
    if 0 in plots:
        pylab.figure(0)
        pylab.plot(x)
        pylab.ylabel('Intensity')
    if 1 in plots:
        pylab.figure(1)
        pylab.subplot(311)
        pylab.plot(w, abs(y), 'o', w, abs(y1))
        pylab.legend(['fft', 'zoom'])
        pylab.ylabel('Magnitude')
        pylab.title('FFT equivalent')
        pylab.subplot(312)
        pylab.plot(w, np.angle(y), 'o', w, np.angle(y1))
        pylab.legend(['fft', 'zoom'])
        pylab.ylabel('Phase (radians)')
        pylab.subplot(313)
        pylab.plot(w, abs(y) - abs(y1))  # ,w,np.angle(y)-np.angle(y1))
        # pylab.legend(['magnitude','phase'])
        pylab.ylabel('Residuals')
    if 2 in plots:
        pylab.figure(2)
        pylab.subplot(211)
        pylab.plot(w, abs(y), 'o', wover, abs(y2), wover, abs(yover))
        pylab.ylabel('Magnitude')
        pylab.title('Oversampled FFT')
        pylab.legend(['fft', 'zoom', 'pad'])
        pylab.subplot(212)
        pylab.plot(wover, abs(yover) - abs(y2),
                   w, abs(y) - abs(y2[0::over]), 'o',
                   w, abs(y) - abs(yover[0::over]), 'x')
        pylab.legend(['pad-zoom', 'fft-zoom', 'fft-pad'])
        pylab.ylabel('Residuals')
    if 3 in plots:
        pylab.figure(3)
        ax1 = pylab.subplot(211)
        pylab.plot(w, abs(y), 'o', w3, abs(y3), wover, abs(yover),
                   w[3:7], abs(y3[::over]), 'x')
        pylab.title('Zoomed FFT')
        pylab.ylabel('Magnitude')
        pylab.legend(['fft', 'zoom', 'pad'])
        pylab.plot(w3, abs(y3), 'x')
        ax1.set_xlim(f1, f2)
        ax2 = pylab.subplot(212)
        pylab.plot(wover[idx3], abs(yover[idx3]) - abs(y3),
                   w[3:7], abs(y[3:7]) - abs(y3[::over]), 'o',
                   w[3:7], abs(y[3:7]) - abs(yover[3 * over:6 * over + 1:over]), 'x')
        pylab.legend(['pad-zoom', 'fft-zoom', 'fft-pad'])
        ax2.set_xlim(f1, f2)
        pylab.ylabel('Residuals')
    if plots != []:
        pylab.show()

    err = norm(y - y1) / norm(y)
    # print "direct err %g"%err
    assert err < 1e-10, "error for direct transform is %g" % (err,)
    err = norm(yover - y2) / norm(yover)
    # print "over err %g"%err
    assert err < 1e-10, "error for oversampling is %g" % (err,)
    err = norm(yover[idx3] - y3) / norm(yover[idx3])
    # print "range err %g"%err
    assert err < 1e-10, "error for subrange is %g" % (err,)


def _testscaled(x):
    n = len(x)
    norm = np.linalg.norm
    assert norm(fft(x) - scaledfft(x)) < 1e-10
    assert norm(fftshift(fft(x))[n / 4:3 * n / 4] - fftshift(scaledfft(x, scale=0.5, m=n / 2))) < 1e-10


def test(demo=None, plots=[1, 2, 3]):
    # 0: Gauss
    t = np.linspace(-2, 2, 128)
    x = np.exp(-t ** 2 / 0.01)
    _test1(x, show=(demo == 0), plots=plots)

    # 1: Linear
    x = [1, 2, 3, 4, 5, 6, 7]
    _test1(x, show=(demo == 1), plots=plots)

    # Check near powers of two
    _test1(range(126 - 31), show=False)
    _test1(range(127 - 31), show=False)
    _test1(range(128 - 31), show=False)
    _test1(range(129 - 31), show=False)
    _test1(range(130 - 31), show=False)

    # Check transform on n-D array input
    x = np.reshape(np.arange(3 * 2 * 28), (3, 2, 28))
    y1 = zoomfft(x, 0, 2 - 2. / 28)
    y2 = zoomfft(x[2, 0, :], 0, 2 - 2. / 28)
    err = np.linalg.norm(y2 - y1[2, 0])
    assert err < 1e-15, "error for n-D array is %g" % (err,)

    # 2: Random (not a test condition)
    if demo == 2:
        x = np.random.rand(101)
        _test1(x, show=True, plots=plots)

    # 3: Spikes
    t = np.linspace(0, 1, 128)
    x = np.sin(2 * pi * t * 5) + np.sin(2 * pi * t * 13)
    _test1(x, show=(demo == 3), plots=plots)

    # 4: Sines
    x = np.zeros(100)
    x[[1, 5, 21]] = 1
    _test1(x, show=(demo == 4), plots=plots)

    # 5: Sines plus complex component
    x += 1j * np.linspace(0, 0.5, x.shape[0])
    _test1(x, show=(demo == 5), plots=plots)

    # 6: Scaled FFT on complex sines
    x += 1j * np.linspace(0, 0.5, x.shape[0])
    if demo == 6:
        demo_scaledfft(x, 0.25, 200)
    _testscaled(x)


def demo_scaledfft(v, scale, m):
    import pylab
    shift = pylab.fftshift
    n = len(v)
    x = pylab.linspace(-0.5, 0.5 - 1. / n, n)
    xz = pylab.linspace(-scale * 0.5, scale * 0.5 * (m - 2.) / m, m)
    pylab.figure()
    pylab.plot(x, shift(abs(fft(v))), label='fft')
    pylab.plot(x, shift(abs(scaledfft(v))), 'ro', label='x1 scaled fft')
    pylab.plot(xz, abs(zoomfft(v, -scale, scale * (m - 2.) / m, m=m)),
               'bo', label='zoomfft')
    pylab.plot(xz, shift(abs(scaledfft(v, m=m, scale=scale))),
               'gx', label='x' + str(scale) + ' scaled fft')
    pylab.gca().set_yscale('log')
    pylab.legend()
    pylab.show()


if __name__ == "__main__":
    # Choose demo in [0,4] to show plot, or None for testing only
    test(demo=None)