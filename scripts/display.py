from matplotlib.cm import get_cmap
from matplotlib.axes import Axes
from matplotlib.ticker import Formatter, ScalarFormatter
from matplotlib.ticker import LogLocator, FixedLocator, MaxNLocator
from matplotlib.ticker import SymmetricalLogLocator
import numpy as np
from gui_util import fft_frequencies

class TimeFormatter(Formatter):
    '''A tick formatter for time axes.
    Automatically switches between seconds, minutes:seconds,
    or hours:minutes:seconds.
    Parameters
    ----------
    lag : bool
        If `True`, then the time axis is interpreted in lag coordinates.
        Anything past the midpoint will be converted to negative time.
    unit : str or None
        Abbreviation of the physical unit for axis labels and ticks.
        Either equal to `s` (seconds) or `ms` (milliseconds) or None (default).
        If set to None, the resulting TimeFormatter object adapts its string
        representation to the duration of the underlying time range:
        `hh:mm:ss` above 3600 seconds; `mm:ss` between 60 and 3600 seconds;
        and `ss` below 60 seconds.
    See also
    --------
    matplotlib.ticker.Formatter
    Examples
    --------
    For normal time
    >>> import matplotlib.pyplot as plt
    >>> times = np.arange(30)
    >>> values = np.random.randn(len(times))
    >>> plt.figure()
    >>> ax = plt.gca()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    >>> ax.set_xlabel('Time')
    Manually set the physical time unit of the x-axis to milliseconds
    >>> times = np.arange(100)
    >>> values = np.random.randn(len(times))
    >>> plt.figure()
    >>> ax = plt.gca()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(unit='ms'))
    >>> ax.set_xlabel('Time (ms)')
    For lag plots
    >>> times = np.arange(60)
    >>> values = np.random.randn(len(times))
    >>> plt.figure()
    >>> ax = plt.gca()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(lag=True))
    >>> ax.set_xlabel('Lag')
    '''

    def __init__(self, lag=False, unit=None):

        if unit not in ['s', 'ms', None]:
            raise ParameterError('Unknown time unit: {}'.format(unit))

        self.unit = unit
        self.lag = lag

    def __call__(self, x, pos=None):
        '''Return the time format as pos'''

        _, dmax = self.axis.get_data_interval()
        vmin, vmax = self.axis.get_view_interval()

        # In lag-time axes, anything greater than dmax / 2 is negative time
        if self.lag and x >= dmax * 0.5:
            # In lag mode, don't tick past the limits of the data
            if x > dmax:
                return ''
            value = np.abs(x - dmax)
            # Do we need to tweak vmin/vmax here?
            sign = '-'
        else:
            value = x
            sign = ''

        if self.unit == 's':
            s = '{:.3g}'.format(value)
        elif self.unit == 'ms':
            s = '{:.3g}'.format(value * 1000)
        else:
            if vmax - vmin > 3600:
                s = '{:d}:{:02d}:{:02d}'.format(int(value / 3600.0),
                                                int(np.mod(value / 60.0, 60)),
                                                int(np.mod(value, 60)))
            elif vmax - vmin > 60:
                s = '{:d}:{:02d}'.format(int(value / 60.0),
                                         int(np.mod(value, 60)))
            else:
                s = '{:.2g}'.format(value)

        return '{:s}{:s}'.format(sign, s)

def cmap(data, robust=True, cmap_seq='magma', cmap_bool='gray_r', cmap_div='coolwarm'):
    '''Get a default colormap from the given data.
    If the data is boolean, use a black and white colormap.
    If the data has both positive and negative values,
    use a diverging colormap.
    Otherwise, use a sequential colormap.
    Parameters
    ----------
    data : np.ndarray
        Input data
    robust : bool
        If True, discard the top and bottom 2% of data when calculating
        range.
    cmap_seq : str
        The sequential colormap name
    cmap_bool : str
        The boolean colormap name
    cmap_div : str
        The diverging colormap name
    Returns
    -------
    cmap : matplotlib.colors.Colormap
        The colormap to use for `data`
    See Also
    --------
    matplotlib.pyplot.colormaps
    '''

    data = np.atleast_1d(data)

    if data.dtype == 'bool':
        return get_cmap(cmap_bool)

    data = data[np.isfinite(data)]

    if robust:
        min_p, max_p = 2, 98
    else:
        min_p, max_p = 0, 100

    max_val = np.percentile(data, max_p)
    min_val = np.percentile(data, min_p)

    if min_val >= 0 or max_val <= 0:
        return get_cmap(cmap_seq)

    return get_cmap(cmap_div)

def specshow(data, x_data, y_data,
             x_axis=None, y_axis=None,
             sr=22050, hop_length=512,
             fmin=None, fmax=None,
             bins_per_octave=12,
             ax=None,
             **kwargs):
    
    if np.issubdtype(data.dtype, np.complexfloating):
        warnings.warn('Trying to display complex-valued input. '
                      'Showing magnitude instead.')
        data = np.abs(data)

    kwargs.setdefault('cmap', cmap(data))
    kwargs.setdefault('rasterized', True)
    kwargs.setdefault('edgecolors', 'None')
    kwargs.setdefault('shading', 'flat')

    all_params = dict(kwargs=kwargs,
                      sr=sr,
                      fmin=fmin,
                      fmax=fmax,
                      bins_per_octave=bins_per_octave,
                      hop_length=hop_length)

    # Get the x and y coordinates
#     y_coords = __mesh_coords(y_axis, y_coords, data.shape[0], **all_params)
#     x_coords = __mesh_coords(x_axis, x_coords, data.shape[1], **all_params)
    y_coords = y_data
    x_coords = x_data

    axes = __check_axes(ax)
    out = axes.pcolormesh(x_coords, y_coords, data, **kwargs)
    __set_current_image(ax, out)

    axes.set_xlim(x_coords.min(), x_coords.max())
    axes.set_ylim(y_coords.min(), y_coords.max())

    # Set up axis scaling
    __scale_axes(axes, x_axis, 'x')
    __scale_axes(axes, y_axis, 'y')

    # Construct tickers and locators
    # __decorate_axis(axes.xaxis, x_axis)
    # __decorate_axis(axes.yaxis, y_axis)
    axes.xaxis.set_label_text('Seconds (s)')
    axes.yaxis.set_label_text('Hertz (Hz)')

    axes.yaxis.set_major_formatter(ScalarFormatter())

    return axes

def __check_axes(axes):
    '''Check if "axes" is an instance of an axis object. If not, use `gca`.'''
    if axes is None:
        import matplotlib.pyplot as plt
        axes = plt.gca()
    elif not isinstance(axes, Axes):
        raise ValueError("`axes` must be an instance of matplotlib.axes.Axes. "
                         "Found type(axes)={}".format(type(axes)))
    return axes

def __set_current_image(ax, img):
    '''Helper to set the current image in pyplot mode.
    If the provided `ax` is not `None`, then we assume that the user is using the object API.
    In this case, the pyplot current image is not set.
    '''

    if ax is None:
        import matplotlib.pyplot as plt
        plt.sci(img)
        
def __scale_axes(axes, ax_type, which):
    '''Set the axis scaling'''

    kwargs = dict()
    if which == 'x':
        thresh = 'linthreshx'
        base = 'basex'
        scale = 'linscalex'
        scaler = axes.set_xscale
        limit = axes.set_xlim
    else:
        thresh = 'linthreshy'
        base = 'basey'
        scale = 'linscaley'
        scaler = axes.set_yscale
        limit = axes.set_ylim

    # Map ticker scales
    if ax_type == 'mel':
        mode = 'symlog'
        kwargs[thresh] = 1000.0
        kwargs[base] = 2

    elif ax_type == 'log':
        mode = 'symlog'
        kwargs[base] = 2
        kwargs[thresh] = 65.41
        kwargs[scale] = 0.5

    elif ax_type in ['cqt', 'cqt_hz', 'cqt_note']:
        mode = 'log'
        kwargs[base] = 2

    elif ax_type == 'tempo':
        mode = 'log'
        kwargs[base] = 2
        limit(16, 480)
    else:
        return

    scaler(mode, **kwargs)
    
def __decorate_axis(axis, ax_type):
    '''Configure axis tickers, locators, and labels'''

    if ax_type == 'tonnetz':
        axis.set_major_formatter(TonnetzFormatter())
        axis.set_major_locator(FixedLocator(0.5 + np.arange(6)))
        axis.set_label_text('Tonnetz')

    elif ax_type == 'chroma':
        axis.set_major_formatter(ChromaFormatter())
        axis.set_major_locator(FixedLocator(0.5 +
                                            np.add.outer(12 * np.arange(10),
                                                         [0, 2, 4, 5, 7, 9, 11]).ravel()))
        axis.set_label_text('Pitch class')

    elif ax_type == 'tempo':
        axis.set_major_formatter(ScalarFormatter())
        axis.set_major_locator(LogLocator(base=2.0))
        axis.set_label_text('BPM')

    elif ax_type == 'time':
        axis.set_major_formatter(TimeFormatter(unit=None, lag=False))
        axis.set_major_locator(MaxNLocator(prune=None,
                                           steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Time')

    elif ax_type == 's':
        axis.set_major_formatter(TimeFormatter(unit='s', lag=False))
        axis.set_major_locator(MaxNLocator(prune=None,
                                           steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Time (s)')

    elif ax_type == 'ms':
        axis.set_major_formatter(TimeFormatter(unit='ms', lag=False))
        axis.set_major_locator(MaxNLocator(prune=None,
                                           steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Time (ms)')

    elif ax_type == 'lag':
        axis.set_major_formatter(TimeFormatter(unit=None, lag=True))
        axis.set_major_locator(MaxNLocator(prune=None,
                                           steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Lag')

    elif ax_type == 'lag_s':
        axis.set_major_formatter(TimeFormatter(unit='s', lag=True))
        axis.set_major_locator(MaxNLocator(prune=None,
                                           steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Lag (s)')

    elif ax_type == 'lag_ms':
        axis.set_major_formatter(TimeFormatter(unit='ms', lag=True))
        axis.set_major_locator(MaxNLocator(prune=None,
                                           steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Lag (ms)')

    elif ax_type == 'cqt_note':
        axis.set_major_formatter(NoteFormatter())
        axis.set_major_locator(LogLocator(base=2.0))
        axis.set_minor_formatter(NoteFormatter(major=False))
        axis.set_minor_locator(LogLocator(base=2.0,
                                          subs=2.0**(np.arange(1, 12)/12.0)))
        axis.set_label_text('Note')

    elif ax_type in ['cqt_hz']:
        axis.set_major_formatter(LogHzFormatter())
        axis.set_major_locator(LogLocator(base=2.0))
        axis.set_minor_formatter(LogHzFormatter(major=False))
        axis.set_minor_locator(LogLocator(base=2.0,
                                          subs=2.0**(np.arange(1, 12)/12.0)))
        axis.set_label_text('Hz')

    elif ax_type in ['mel', 'log']:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_major_locator(SymmetricalLogLocator(axis.get_transform()))
        axis.set_label_text('Hz')

    elif ax_type in ['linear', 'hz']:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_label_text('Hz')

    elif ax_type in ['frames']:
        axis.set_label_text('Frames')

    elif ax_type in ['off', 'none', None]:
        axis.set_label_text('')
        axis.set_ticks([])

def __coord_fft_hz(n, sr=22050, **_kwargs):
    '''Get the frequencies for FFT bins'''
    n_fft = 2 * (n - 1)
    # The following code centers the FFT bins at their frequencies
    # and clips to the non-negative frequency range [0, nyquist]
    basis = fft_frequencies(sr=sr, n_fft=n_fft)
    fmax = basis[-1]
    basis -= 0.5 * (basis[1] - basis[0])
    basis = np.append(np.maximum(0, basis), [fmax])
    return basis