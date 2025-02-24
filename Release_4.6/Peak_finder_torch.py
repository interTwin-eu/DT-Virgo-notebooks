import torch

def _peak_prominences_torch(x, peaks, wlen=None):
    """
    Calculate the prominences of peaks.

    Parameters
    ----------
    x : torch.Tensor
        A 1-D tensor representing the input signal.
    peaks : torch.Tensor
        Indices of peaks in `x`.
    wlen : int or None, optional
        Width of the window used in prominence calculations.

    Returns
    -------
    prominences : torch.Tensor
        The prominences of the peaks.
    left_bases : torch.Tensor
        Indices of the left bases of the peaks.
    right_bases : torch.Tensor
        Indices of the right bases of the peaks.
    """
    if wlen is None:
        wlen = 1

    left_bases = torch.zeros_like(peaks)
    right_bases = torch.zeros_like(peaks)
    prominences = torch.zeros_like(peaks, dtype=torch.float)

    for i, peak in enumerate(peaks):
        # Define left and right boundaries for the window
        left = max(0, peak - wlen)
        right = min(len(x) - 1, peak + wlen)
        window = x[left:right + 1]

        # Find the indices of the left and right bases
        left_base_idx = torch.argmax(window[:peak - left])
        right_base_idx = peak - left + torch.argmax(window[peak - left:]) - 1

        # Calculate prominence
        peak_value = x[peak]
        left_base_value = x[left + left_base_idx]
        right_base_value = x[left + right_base_idx]

        prominences[i] = max(peak_value - left_base_value, peak_value - right_base_value)

        left_bases[i] = left + left_base_idx
        right_bases[i] = left + right_base_idx

    return prominences, left_bases, right_bases


def _peak_widths_torch(x, peaks, rel_height=0.5, prominences=None, left_bases=None, right_bases=None):
    """
    Calculate the widths of peaks.

    Parameters
    ----------
    x : torch.Tensor
        A 1-D tensor representing the input signal.
    peaks : torch.Tensor
        Indices of peaks in `x`.
    rel_height : float, optional
        Relative height at which widths are measured.
    prominences : torch.Tensor, optional
        Prominences of the peaks.
    left_bases : torch.Tensor, optional
        Indices of the left bases of the peaks.
    right_bases : torch.Tensor, optional
        Indices of the right bases of the peaks.

    Returns
    -------
    widths : torch.Tensor
        The widths of the peaks.
    width_heights : torch.Tensor
        The heights of the peaks at their widths.
    left_ips : torch.Tensor
        Indices of the left endpoints of the peaks.
    right_ips : torch.Tensor
        Indices of the right endpoints of the peaks.
    """
    if prominences is None or left_bases is None or right_bases is None:
        raise ValueError("Prominences and base indices must be provided.")

    widths = torch.zeros_like(peaks, dtype=torch.float)
    width_heights = torch.zeros_like(peaks, dtype=torch.float)
    left_ips = torch.zeros_like(peaks)
    right_ips = torch.zeros_like(peaks)

    for i, peak in enumerate(peaks):
        # Define the boundaries for the window
        left = int(left_bases[i])
        right = int(right_bases[i])

        # Find the height at which the width is measured
        peak_height = x[peak]
        height = peak_height - prominences[i] * rel_height

        # Find the left and right indices where the peak crosses the height
        left_idx = torch.nonzero(x[left:peak] >= height, as_tuple=False)
        right_idx = torch.nonzero(x[peak:right] >= height, as_tuple=False)

        if len(left_idx) > 0:
            left_ips[i] = left + left_idx[0]
        else:
            left_ips[i] = peak

        if len(right_idx) > 0:
            right_ips[i] = peak + right_idx[-1] + 1
        else:
            right_ips[i] = peak

        widths[i] = right_ips[i] - left_ips[i]
        width_heights[i] = x[peak] - height

    return widths, width_heights, left_ips, right_ips


def _select_by_property(values, vmin, vmax):
    """
    Select indices of values within the given range.

    Parameters
    ----------
    values : torch.Tensor
        The values to select from.
    vmin : float or torch.Tensor or None
        The minimum value (inclusive).
    vmax : float or torch.Tensor or None
        The maximum value (inclusive).

    Returns
    -------
    keep : torch.Tensor
        A boolean tensor indicating the selected indices.
    """
    if vmin is None and vmax is None:
        return torch.ones_like(values, dtype=torch.bool)

    vmin = torch.tensor(vmin, dtype=values.dtype) if vmin is not None else None
    vmax = torch.tensor(vmax, dtype=values.dtype) if vmax is not None else None

    if vmin is not None and vmax is not None:
        if vmin.numel() == 1 and vmax.numel() == 1:
            return (values >= vmin) & (values <= vmax)
        elif vmin.numel() == values.numel() and vmax.numel() == values.numel():
            return (values >= vmin) & (values <= vmax)
        elif vmin.numel() == 1 and vmax.numel() == values.numel():
            return (values >= vmin) & (values <= vmax.unsqueeze(0))
        elif vmin.numel() == values.numel() and vmax.numel() == 1:
            return (values >= vmin) & (values <= vmax.unsqueeze(0))
        else:
            raise ValueError("Invalid shape of vmin and vmax.")
    elif vmin is not None:
        return values >= vmin
    elif vmax is not None:
        return values <= vmax
    else:
        raise ValueError("Invalid vmin and vmax.")



def _unpack_condition_args(condition, x, peaks):
    """
    Unpack condition arguments to minimal and maximal values.

    Parameters
    ----------
    condition : float or torch.Tensor or sequence
        The condition argument.
    x : torch.Tensor
        A 1-D tensor representing the input signal.
    peaks : torch.Tensor
        Indices of peaks in `x`.

    Returns
    -------
    vmin : float or torch.Tensor
        The minimal value.
    vmax : float or torch.Tensor
        The maximal value.
    """
    if condition is None:
        return None, None

    if isinstance(condition, (int, float)):
        return condition, None

    if isinstance(condition, (tuple, list)):
        if len(condition) == 1:
            return condition[0], None
        elif len(condition) == 2:
            return condition

    if isinstance(condition, torch.Tensor):
        if condition.numel() == 1:
            return condition.item(), None
        elif condition.numel() == len(peaks):
            return condition, None
        elif condition.numel() == 2:
            return condition

    raise ValueError("Invalid condition argument.")


def _calculate_plateau_sizes(x, peaks):
    """
    Calculate plateau sizes of peaks.

    Parameters
    ----------
    x : torch.Tensor
        A 1-D tensor representing the input signal.
    peaks : torch.Tensor
        Indices of peaks in `x`.

    Returns
    -------
    plateau_sizes : torch.Tensor
        Sizes of the plateaus.
    left_edges : torch.Tensor
        Indices of the left edges of the plateaus.
    right_edges : torch.Tensor
        Indices of the right edges of the plateaus.
    """
    plateau_sizes = torch.zeros_like(peaks)
    left_edges = torch.zeros_like(peaks)
    right_edges = torch.zeros_like(peaks)

    for i, peak in enumerate(peaks):
        left = peak
        right = peak

        # Move left until the signal decreases or we reach the start
        while left > 0 and x[left - 1] == x[left]:
            left -= 1

        # Move right until the signal decreases or we reach the end
        while right < len(x) - 1 and x[right + 1] == x[right]:
            right += 1

        plateau_sizes[i] = right - left + 1
        left_edges[i] = left
        right_edges[i] = right

    return plateau_sizes, left_edges, right_edges


def _select_by_peak_threshold_torch(x, peaks, tmin, tmax):
    """
    Select peaks based on threshold conditions.

    Parameters
    ----------
    x : torch.Tensor
        A 1-D tensor representing the input signal.
    peaks : torch.Tensor
        Indices of peaks in `x`.
    tmin : float or torch.Tensor
        The minimum threshold.
    tmax : float or torch.Tensor
        The maximum threshold.

    Returns
    -------
    keep : torch.Tensor
        A boolean tensor indicating the selected peaks.
    left_thresholds : torch.Tensor
        The values of the left thresholds for the selected peaks.
    right_thresholds : torch.Tensor
        The values of the right thresholds for the selected peaks.
    """
    left_thresholds = torch.zeros_like(peaks)
    right_thresholds = torch.zeros_like(peaks)

    if tmin is None and tmax is None:
        return torch.ones_like(peaks, dtype=torch.bool), left_thresholds, right_thresholds

    if tmin is None:
        keep = x[peaks] <= tmax
        right_thresholds[keep] = tmax
    elif tmax is None:
        keep = x[peaks] >= tmin
        left_thresholds[keep] = tmin
    else:
        keep = (x[peaks] >= tmin) & (x[peaks] <= tmax)
        left_thresholds[keep] = tmin
        right_thresholds[keep] = tmax

    return keep, left_thresholds, right_thresholds


def _select_by_peak_distance_torch(peaks, distance):
    """
    Select peaks based on distance between peaks.

    Parameters
    ----------
    peaks : torch.Tensor
        Indices of peaks.
    distance : int
        The minimum distance between peaks.

    Returns
    -------
    keep : torch.Tensor
        A boolean tensor indicating the selected peaks.
    """
    if distance is None:
        return torch.ones_like(peaks, dtype=torch.bool)

    keep = torch.ones_like(peaks, dtype=torch.bool)
    idx = 0

    while idx < len(peaks) - 1:
        if peaks[idx + 1] - peaks[idx] < distance:
            # Remove the peak with lower value
            if x[peaks[idx]] > x[peaks[idx + 1]]:
                keep[idx + 1] = False
            else:
                keep[idx] = False
        idx += 1

    return keep




def find_peaks_torch(x, height=None, threshold=None, distance=None,
                     prominence=None, width=None, wlen=None, rel_height=0.5,
                     plateau_size=None):
    """
    Find peaks inside a signal based on peak properties using PyTorch.

    This function takes a 1-D tensor and finds all local maxima by
    simple comparison of neighboring values. Optionally, a subset of these
    peaks can be selected by specifying conditions for a peak's properties.

    Parameters
    ----------
    x : torch.Tensor
        A signal with peaks.
    height : number or torch.Tensor or sequence, optional
        Required height of peaks. Either a number, ``None``, a tensor matching
        `x` or a 2-element sequence of the former. The first element is
        always interpreted as the minimal and the second, if supplied, as the
        maximal required height.
    threshold : number or torch.Tensor or sequence, optional
        Required threshold of peaks, the vertical distance to its neighboring
        samples. Either a number, ``None``, a tensor matching `x` or a
        2-element sequence of the former. The first element is always
        interpreted as the minimal and the second, if supplied, as the maximal
        required threshold.
    distance : number, optional
        Required minimal horizontal distance (>= 1) in samples between
        neighboring peaks. Smaller peaks are removed first until the condition
        is fulfilled for all remaining peaks.
    prominence : number or torch.Tensor or sequence, optional
        Required prominence of peaks. Either a number, ``None``, a tensor
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the minimal and the second, if
        supplied, as the maximal required prominence.
    width : number or torch.Tensor or sequence, optional
        Required width of peaks in samples. Either a number, ``None``, a tensor
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the minimal and the second, if
        supplied, as the maximal required width.
    wlen : int, optional
        Used for calculation of the peaks prominences, thus it is only used if
        one of the arguments `prominence` or `width` is given. See argument
        `wlen` in `peak_prominences` for a full description of its effects.
    rel_height : float, optional
        Used for calculation of the peaks width, thus it is only used if `width`
        is given. See argument  `rel_height` in `peak_widths` for a full
        description of its effects.
    plateau_size : number or torch.Tensor or sequence, optional
        Required size of the flat top of peaks in samples. Either a number,
        ``None``, a tensor matching `x` or a 2-element sequence of the former.
        The first element is always interpreted as the minimal and the second,
        if supplied as the maximal required plateau size.

    Returns
    -------
    peaks : torch.Tensor
        Indices of peaks in `x` that satisfy all given conditions.
    properties : dict
        A dictionary containing properties of the returned peaks which were
        calculated as intermediate results during evaluation of the specified
        conditions:

        * 'peak_heights'
              If `height` is given, the height of each peak in `x`.
        * 'left_thresholds', 'right_thresholds'
              If `threshold` is given, these keys contain a peak's vertical
              distance to its neighbouring samples.
        * 'prominences', 'right_bases', 'left_bases'
              If `prominence` is given, these keys are accessible. See
              `peak_prominences` for a description of their content.
        * 'width_heights', 'left_ips', 'right_ips'
              If `width` is given, these keys are accessible. See `peak_widths`
              for a description of their content.
        * 'plateau_sizes', left_edges', 'right_edges'
              If `plateau_size` is given, these keys are accessible and contain
              the indices of a peak's edges (edges are still part of the
              plateau) and the calculated plateau sizes.

    Notes
    -----
    In the context of this function, a peak or local maximum is defined as any
    sample whose two direct neighbours have a smaller amplitude. For flat peaks
    (more than one sample of equal amplitude wide) the index of the middle
    sample is returned (rounded down in case the number of samples is even).
    For noisy signals the peak locations can be off because the noise might
    change the position of local maxima. In those cases consider smoothing the
    signal before searching for peaks or use other peak finding and fitting
    methods.

    Some additional comments on specifying conditions:

    * Almost all conditions (excluding `distance`) can be given as half-open or
      closed intervals, e.g., ``1`` or ``(1, None)`` defines the half-open
      interval :math:`[1, \\infty]` while ``(None, 1)`` defines the interval
      :math:`[-\\infty, 1]`. The open interval ``(None, None)`` can be specified
      as well, which returns the matching properties without exclusion of peaks.
    * The border is always included in the interval used to select valid peaks.
    * For several conditions the interval borders can be specified with
      tensors matching `x` in shape which enables dynamic constrains based on
      the sample position.
    * The conditions are evaluated in the following order: `plateau_size`,
      `height`, `threshold`, `distance`, `prominence`, `width`. In most cases
      this order is the fastest one because faster operations are applied first
      to reduce the number of peaks that need to be evaluated later.
    * While indices in `peaks` are guaranteed to be at least `distance` samples
      apart, edges of flat peaks may be closer than the allowed `distance`.
    * Use `wlen` to reduce the time it takes to evaluate the conditions for
      `prominence` or `width` if `x` is large or has many local maxima.

    Examples
    --------
    To demonstrate this function's usage we create a random signal `x`:

    >>> import torch
    >>> x = torch.randn(100)

    Let's find all peaks (local maxima) in `x` whose amplitude lies above 0:

    >>> peaks, _ = find_peaks_torch(x, height=0)

    We can select peaks below 0 with ``height=(None, 0)`` or use tensors matching
    `x` in size to reflect a changing condition for different parts of the
    signal.

    Another useful condition for periodic signals can be given with the
    `distance` argument. In this case, we can easily select the positions of
    peaks by demanding a distance of at least 10 samples.

    >>> peaks, _ = find_peaks_torch(x, distance=10)

    Especially for noisy signals peaks can be easily grouped by their
    prominence. E.g., we can select all peaks except for those with prominence
    less than 1.

    >>> peaks, properties = find_peaks_torch(x, prominence=(1, None))

    And, finally, let's examine a different signal `y` which contains
    beat forms of different shape. To select only the atypical peaks, we
    combine two conditions: a minimal prominence of 1 and width of at least 5
    samples.

    >>> y = torch.randn(100)
    >>> peaks, properties = find_peaks_torch(y, prominence=1, width=5)
    """
    if distance is not None and distance < 1:
        raise ValueError('`distance` must be greater or equal to 1')

    peaks = torch.nonzero((x[:-2] < x[1:-1]) & (x[1:-1] > x[2:])).squeeze(1) + 1

    properties = {}

    if plateau_size is not None:
        plateau_sizes, left_edges, right_edges = _calculate_plateau_sizes(x, peaks)
        pmin, pmax = _unpack_condition_args(plateau_size, x, peaks)
        keep = _select_by_property(plateau_sizes, pmin, pmax)
        peaks = peaks[keep]
        properties["plateau_sizes"] = plateau_sizes[keep]
        properties["left_edges"] = left_edges[keep]
        properties["right_edges"] = right_edges[keep]

    if height is not None:
        peak_heights = x[peaks]
        hmin, hmax = _unpack_condition_args(height, x, peaks)
        keep = _select_by_property(peak_heights, hmin, hmax)
        peaks = peaks[keep]
        properties["peak_heights"] = peak_heights[keep]

    if threshold is not None:
        tmin, tmax = _unpack_condition_args(threshold, x, peaks)
        keep, left_thresholds, right_thresholds = _select_by_peak_threshold_torch(
            x, peaks, tmin, tmax)
        peaks = peaks[keep]
        properties["left_thresholds"] = left_thresholds[keep]
        properties["right_thresholds"] = right_thresholds[keep]

    if distance is not None:
        keep = _select_by_peak_distance_torch(peaks, distance)
        peaks = peaks[keep]

    if prominence is not None or width is not None:
        wlen = _arg_wlen_as_expected(wlen)
        properties.update(zip(
            ['prominences', 'left_bases', 'right_bases'],
            _peak_prominences_torch(x, peaks, wlen=wlen)
        ))

    if prominence is not None:
        pmin, pmax = _unpack_condition_args(prominence, x, peaks)
        keep = _select_by_property(properties['prominences'], pmin, pmax)
        peaks = peaks[keep]

    if width is not None:
        properties.update(zip(
            ['widths', 'width_heights', 'left_ips', 'right_ips'],
            _peak_widths_torch(x, peaks, rel_height, properties['prominences'],
                               properties['left_bases'], properties['right_bases'])
        ))
        wmin, wmax = _unpack_condition_args(width, x, peaks)
        keep = _select_by_property(properties['widths'], wmin, wmax)
        peaks = peaks[keep]

    return peaks, properties
