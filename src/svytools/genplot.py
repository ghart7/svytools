# Built-ins
from pathlib import Path
import re
from typing import (
    Optional, Tuple, List, Any, Union, Dict
)
import colorsys
from numbers import Number
import itertools as it

# Standard libs
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import seaborn as sns
from PIL import Image
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# svytools libs
from svytools.genutils import make_logspace, get_config


inverted_rcparams = {
    # Figure
    "figure.facecolor": (0.0, 0.0, 0.0, 0.0),  # transparent figure background
    "figure.edgecolor": "white",

    # Axes
    "axes.facecolor": (0.0, 0.0, 0.0, 0.0),  # transparent axes background
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "axes.titlecolor": "white",

    # Ticks
    "xtick.color": "white",
    "ytick.color": "white",

    # Text
    "text.color": "white",

    # Grid
    "grid.color": "gray",
    "grid.alpha": 0.5,

    # Legend
    "legend.facecolor": (0.0, 0.0, 0.0, 0.0), # transparent legend background
    "legend.edgecolor": "white",
    "legend.labelcolor": "white",
    
    # Saving figures
    "savefig.facecolor": (0.0, 0.0, 0.0, 0.0), # transparent background when saving
    "savefig.edgecolor": (0.0, 0.0, 0.0, 0.0),
    "savefig.transparent": True,
}


def loglog_hist(vals: np.ndarray,
                binminmax: Tuple[Number, Number],
                numbins: int = 100,
                vline: Optional[Number] = None,
                title: Optional[str] = None,
                ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot a log-log histogram of a list-like of values.

    Parameters
    ----------
    vals : np.ndarray
        An array of values to be binned and plotted.
    binminmax : tuple of (Number, Number)
        A tuple specifying the minimum and maximum range for the histogram bins.
        The minimum value must be greater than 0 for a log scale.
    numbins : int, optional
        The number of bins to create for the histogram.
    vline : Number, optional
        If provided, draws a vertical line at this x-coordinate.
    title : str, optional
        The title for the plot's axes.
    ax : plt.Axes, optional
        An existing Matplotlib Axes object to plot on. If None, a new one is created.

    Returns
    -------
    plt.Axes
        The Axes object containing the log-log histogram.
    """ 
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        
    ax.hist(vals, bins=make_logspace(binminmax[0], binminmax[1], numbins))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(which='both', alpha=0.5)
    ax.set_title(title)
    
    if vline is not None:
        ylim = ax.get_ylim()
        ax.vlines(vline, ylim[0], ylim[1], color='k')
        
    return ax


def subplots(nplots: int,
             ncols: Optional[int] = None,
             cols: Optional[int] = None,
             ar: Optional[float] = None,
             fss: Optional[float] = None,
             split_each: Optional[Tuple[int, int]] = None,
             hrs: Optional[List[float]] = None,
             wrs: Optional[List[float]] = None,
             as_seq: bool = False,
             **kwargs: Any) -> Tuple[mpl.figure.Figure, Union[mpl.axes.Axes, np.ndarray]]:
    """Create a matplotlib figure and axes objects.

    Creates a grid of subplots with a specified number of plots,
    maximum columns, and aspect ratio.

    Parameters
    ----------
    nplots : int
        The total number of subplots.
    ncols : int, optional
        The maximum number of columns in the grid, by default 4.
    cols : Optional[int], optional
        Alias for ncols, by default None.
    ar : float, optional
        The aspect ratio (width:height) of each subplot, by default 1.
    fss : Optional[float], optional
        The scaling factor for the figure size, by default 4.
    split_each : Optional[Tuple[int, int]], optional
        If provided, each subplot will be split into a grid of the given shape, by default None.
    hrs : Optional[List[float]], optional
        The height ratios for the split subplots, by default None.
    wrs : Optional[List[float]], optional
        The width ratios for the split subplots, by default None.
    as_seq : bool, optional
        If nplots=1 and split_each is None, return an ndarray of Axes objects instead
        of a single object.
    **kwargs : Any
        Keyword arguments to pass to `matplotlib.pyplot.subplots`.

    Returns
    -------
    Tuple[mpl.figure.Figure, Union[mpl.axes.Axes, np.ndarray]]
        The created Figure object and the array of created Axes objects.
        Extra axes are deleted if `nplots % ncols != 0`.

    Notes
    -----
    If `nplots % ncols != 0`, the extra axes in the last row are deleted.
    """

    if cols is not None:
        ncols = cols

    if ncols is None:
        ncols = 4
    if ar is None:
        ar = 1.0
    if fss is None:
        fss = 4.0

    if nplots == 1:
        fig, axes = plt.subplots(
            1, 1, figsize=(1 * ar * fss, 1 * fss), **kwargs
        )
    else:
        # Calculate the number of rows needed
        nrows = np.ceil(nplots / ncols).astype(int)

        # Create the figure and axes
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * ar * fss, nrows * fss),
            **kwargs,
        )

        if (
            nplots % ncols != 0
        ):  # If nplots % ncols != 0, delete the extra axes
            for ax in axes.flatten()[nplots:]:
                fig.delaxes(ax)

    if split_each is not None:
        if isinstance(axes, mpl.axes.Axes):
            # If only one subplot, convert to a 2D array with one element
            axes = np.array([[axes]])
        # Split each subplot into a grid
        orig_shape = axes.shape
        axes = np.ravel(axes)

        for i in range(len(axes)):
            # Store the sharex, sharey, then remove the original subplot
            # sharex, sharey = axes[i].sharex, axes[i].sharey

            try:  # In the event that it was removed because of nplots % cols != 0
                axes[i].remove()
            except KeyError:
                continue

            # Create a new grid of subplots
            gs = mpl.gridspec.GridSpecFromSubplotSpec(
                *split_each,
                subplot_spec=axes[i].get_subplotspec(),
                height_ratios=hrs,
                width_ratios=wrs,
            )

            # Create the subplots
            axes[i] = np.array(
                [fig.add_subplot(gs[j]) for j in range(np.prod(split_each))]
            ).reshape(split_each)

        axes = axes.reshape(orig_shape)

    if nplots == 1 and split_each is None and as_seq:
        # axes is an Axes object, but user wants a sequence
        axes = np.array([axes])
    return fig, axes


def get_inset_ax_params(pos: str,
                        ax: plt.Axes,
                        width: float,
                        height: float) -> Tuple[List[float], float, Tuple[str, ...], Tuple[bool, ...]]:
    """
    Calculates parameters for creating an inset axes within a parent axes.

    Parameters
    ----------
    pos : str
        The corner position for the inset axes ('lower left', 'upper right', etc.).
    ax : plt.Axes
        The parent axes object.
    width : float
        The width of the inset axes, as a fraction of the parent axes' width.
    height : float
        The height of the inset axes, as a fraction of the parent axes' height.

    Returns
    -------
    tuple
        A tuple containing:
        - `rect`: A list `[left, bottom, width, height]` for the new axes.
        - `yadjust_pos`: A float for adjusting y-tick positions.
        - `yk`: A tuple of y-tick parameter keys.
        - `yv`: A tuple of y-tick parameter boolean values.

    Raises
    ------
    ValueError
        If `pos` is not a valid corner position.
    """
    # yk, yv = y-tick parameters keys, values
    yk = "labelleft", "labelright", "left", "right"

    if pos == 'lower left':
        rect = [ax.get_position().x0, ax.get_position().y0, width, height]
        yadjust_pos = 1
        yv = (False, True, False, True)
    elif pos == 'lower right':
        rect = [ax.get_position().x1 - width, ax.get_position().y0, width, height]
        yadjust_pos = 1
        yv = (True, False, True, False)
    elif pos == 'upper left':
        rect = [ax.get_position().x0, ax.get_position().y1 - height, width, height]
        yadjust_pos = 0.95
        yv = (False, True, False, False)
    elif pos == 'upper right':
        rect = [ax.get_position().x1 - width, ax.get_position().y1 - height, width, height]
        yadjust_pos = 0.95
        yv = (True, False, False, False)
    else:
        raise ValueError(
            "Invalid pos. Choose one of: 'lower left', "
            "'lower right', 'upper left', or 'upper right'."
        )
    return rect, yadjust_pos, yk, yv


def cbar_in_axes(fig: mpl.figure.Figure,
                 ax: mpl.axes.Axes,
                 pos: Optional[str] = None,
                 shape: Optional[Tuple[float, float]] = None,
                 cax: Optional[mpl.axes.Axes] = None) -> Tuple[mpl.axes.Axes, float, Dict[str, bool]]:
    """
    Places a colorbar as an inset inside a Matplotlib axes.

    Parameters
    ----------
    fig : mpl.figure.Figure
        The figure object.
    ax : mpl.axes.Axes
        The axes object to place the colorbar in.
    pos : str, optional
        Position of the colorbar. One of 'lower left', 'lower right',
        'upper left', or 'upper right'. Defaults to 'lower right'.
    shape : tuple of (float, float), optional
        The (width, height) of the colorbar as a fraction of the parent axes'
        dimensions. Defaults to (0.04, 0.12).
    cax : mpl.axes.Axes, optional
        An existing axes to use for the colorbar. If None, a new one is created.

    Returns
    -------
    tuple
        A tuple containing:
        - The colorbar axes object.
        - The y-adjustment position for ticks.
        - A dictionary of tick parameters.

    Raises
    ------
    ValueError
        If `pos` is invalid.
    TypeError
        If `cax` is not a valid Axes object.
    """

    shape = (0.04, 0.12) if shape is None else np.array(shape)

    ax_size = (ax.get_position().x1 - ax.get_position().x0,
               ax.get_position().y1 - ax.get_position().y0)
    
    cbar_w = shape[0] * ax_size[0]
    cbar_h = shape[1] * ax_size[1]
    
    pos = "lower right" if pos is None else pos

    rect, yadjust_pos, yk, yv = get_inset_ax_params(pos, ax, cbar_w, cbar_h)

    if cax is None:
        cax = fig.add_axes(rect)
    elif isinstance(cax, mpl.axes.Axes):
        cax.set_position(rect)
    else:
        raise TypeError("cax must be a matplotlib Axes object or None.")
    
    tick_params = dict(zip(yk, yv))

    return cax, yadjust_pos, tick_params


def adj_light(color: Union[str, Tuple[float, ...]], amount: float = 0.5) -> Tuple[float, float, float]:
    """
    Lightens a given color.

    The function adjusts the luminosity of the color in HLS space.

    Parameters
    ----------
    color : str or tuple
        The input color. Can be a Matplotlib color string, hex string, or RGB tuple.
    amount : float, optional
        The amount to lighten the color. 0.0 gives the original color, 1.0 gives white.

    Returns
    -------
    tuple
        The lightened color as an RGB tuple.
    """
    
    try:
        c = mpl.colors.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def get_contrast_color(hex_color: str, lum_thresh: int = 186) -> str:
    """
    Calculates a high-contrast color (black or white) for a given hex color.

    This is useful for determining a legible text color to overlay on a
    colored background.

    Parameters
    ----------
    hex_color : str
        The background color as a hexadecimal string (e.g., '#FFFFFF').
    lum_thresh : int, optional
        The luminance threshold (0-255) for switching between black and white text.

    Returns
    -------
    str
        '#000000' (black) or '#FFFFFF' (white).
    """
    r, g, b = np.array(mpl.colors.hex2color(hex_color))*255
    rgbsum = (r*0.299 + g*0.587 + b*0.114)
    return '#000000' if rgbsum > lum_thresh else '#FFFFFF'


def natural_sort_key(text: str) -> List[Union[str, int]]:
    """
    Creates a key for natural sorting (e.g., 'item1', 'item2', 'item10').

    Parameters
    ----------
    text : str
        The string to be converted into a sortable key.

    Returns
    -------
    list
        A list of mixed strings and integers for sorting.
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]


def create_gif_from_pngs(png_dir: Union[str, Path],
                         output_gif: Union[str, Path],
                         duration: int = 200,
                         loop: int = 0) -> None:
    """
    Creates an animated GIF from a directory of PNG files.

    The PNG files are sorted naturally before being combined into the GIF.

    Parameters
    ----------
    png_dir : str or Path
        The directory containing the PNG files.
    output_gif : str or Path
        The path for the output GIF file.
    duration : int, optional
        The duration (in milliseconds) for each frame.
    loop : int, optional
        The number of times the GIF should loop (0 means infinite).
    """
    png_files = sorted(Path(png_dir).glob('*.png'), key=natural_sort_key)
    images = [Image.open(f) for f in png_files]
    
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )


def id_axes(ax: mpl.axes.Axes, 
            lim: tuple[float, float] | None = None, 
            tix: list | None = None) -> mpl.axes.Axes:
    """Make x and y axes identical.

    Creates identical x and y axes with the same range, tick locations,
    and tick labels, while ensuring all data remains visible.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify.
    lim : tuple[float, float], optional
        A 2-tuple of (min_limit, max_limit) to use for both axes. If None,
        limits are computed from the existing axes limits to include all data.
    tix : list, optional
        A list of tick locations to apply to both axes. If None, the ticks
        from the axis with more ticks are used.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object.
    """
    if lim is None:
        lim = min(min(ax.get_xlim()), min(ax.get_ylim())), max(max(ax.get_xlim()), max(ax.get_ylim()))

    ax.set_xlim(lim)
    ax.set_ylim(lim)

    if tix is None:
        tix = ax.get_xticks() if len(ax.get_xticks()) > len(ax.get_yticks()) else ax.get_yticks()
    
    tix = np.asarray(tix)
    ax.set_xticks(ticks=tix)
    ax.set_yticks(ticks=tix)

    # Create integer labels for whole numbers, float labels otherwise
    is_float = (tix % 1).astype(bool)
    tix_labels = np.where(is_float, tix.astype(str), tix.astype(int).astype(str))
    
    ax.set_xticklabels(labels=tix_labels)
    ax.set_yticklabels(labels=tix_labels)
    
    return ax


def boxstrip(df, x, y, hue=None, ax=None, box_kws=None, strip_kws=None, common_kws=None):
    """Create a boxplot overlaid with a stripplot.

    This function uses seaborn to create a boxplot and a stripplot on the
    same axes. It provides a convenient way to visualize the distribution
    of a numerical variable across different categories.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the plotting data.
    x : str
        The name of the column in `df` to be used for the x-axis.
    y : str
        The name of the column in `df` to be used for the y-axis.
    hue : str, optional
        The name of the column in `df` for color encoding. Defaults to None.
    ax : mpl.axes.Axes, optional
        The matplotlib axes on which to draw the plot. If None, a new figure
        and axes are created. Defaults to None.
    box_kws : dict, optional
        Keyword arguments to pass to `seaborn.boxplot`.
    strip_kws : dict, optional
        Keyword arguments to pass to `seaborn.stripplot`.
    common_kws : dict, optional
        Keyword arguments common to both plots. Will override
        any conflicting keys in `box_kws` and `strip_kws`.

    Returns
    -------
    mpl.axes.Axes
        The matplotlib axes object containing the plot.
    """

    if ax is None:
        fig, ax = subplots(1, ar=3, fss=6)
    if hue == x:
        dodge = False
    else:
        dodge = True

    if box_kws is None:
        box_kws = {}
    if strip_kws is None:
        strip_kws = {}

    if common_kws:
        box_kws.update(common_kws)
        strip_kws.update(common_kws)
    # print(box_kws)

    default_box_kws = {'dodge': dodge, 'showfliers': False, 'saturation': 1.0}
    default_strip_kws = {'dodge': dodge, 'size' : 5, 'jitter' : 0.1, 'linewidth': 0.5, 'legend': False}

    box_kws = get_config(box_kws, default_box_kws)
    strip_kws = get_config(strip_kws, default_strip_kws)

    ax = sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, **box_kws)
    ax = sns.stripplot(data=df, x=x, y=y, hue=hue, ax=ax, **strip_kws)

    return ax


def set_tick_params(ax: mpl.axes.Axes, 
                    **kwargs: Any) -> mpl.axes.Axes:
    """Set tick parameters with added support for text alignment.

    This function is a wrapper around `matplotlib.axes.Axes.tick_params`
    that adds support for horizontal ('ha') and vertical ('va') alignment
    of tick labels, which is not natively supported by `tick_params`.

    Parameters
    ----------
    ax : mpl.axes.Axes
        The matplotlib axes object to modify.
    **kwargs : Any
        Keyword arguments. These include standard `ax.tick_params` arguments
        plus 'ha'/'horizontalalignment' and 'va'/'verticalalignment'.
        The `axis` kwarg ('x', 'y', or 'both') determines which tick labels
        are affected by alignment settings.

    Returns
    -------
    mpl.axes.Axes
        The modified matplotlib axes object.

    Notes
    -----
    Alignment keywords ('ha', 'horizontalalignment', 'va', 'verticalalignment')
    are extracted and applied separately using `ax.set_xticklabels` and
    `ax.set_yticklabels`. The remaining keywords are passed directly to
    `ax.tick_params`.
    """
    alignment_kws = {}
    for key in ['ha', 'horizontalalignment']:
        if key in kwargs:
            alignment_kws['ha'] = kwargs.pop(key)
    for key in ['va', 'verticalalignment']:
        if key in kwargs:
            alignment_kws['va'] = kwargs.pop(key)

    ax.tick_params(**kwargs)

    if alignment_kws:
        axis = kwargs.get('axis', 'both')
        if axis in ['x', 'both']:
            # To prevent UserWarning about FixedLocator, we get and set ticks
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), **alignment_kws)
        if axis in ['y', 'both']:
            # To prevent UserWarning about FixedLocator, we get and set ticks
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(ax.get_yticklabels(), **alignment_kws)

    return ax


def add_heatmap_frames(ax, mask, color='cyan', linewidth=2):
    """
    Add rectangular frames around True values in a boolean mask on a seaborn heatmap.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object containing the heatmap.
    mask : np.ndarray or pd.DataFrame
        Boolean mask with same shape as heatmap data.
        Frames will be drawn around True values.
    color : str, default='cyan'
        Color of the frame lines.
    linewidth : float, default=2
        Width of the frame lines.
    """
    
    # Convert to numpy array if DataFrame
    if hasattr(mask, 'values'):
        mask = mask.values
    
    # Find contiguous regions
    rows, cols = np.where(mask)
    
    if len(rows) == 0:
        return
    
    # Add rectangles around each True cell
    for i, j in zip(rows, cols):
        rect = mpl.patches.Rectangle((j, i), 1, 1, fill=False, 
                                     edgecolor=color, linewidth=linewidth)
        ax.add_patch(rect)
    
    return ax


def create_alpha_cmap(base_cmap, alpha=None, scale_alpha=False):
    """
    Create a colormap with specified alpha transparency.

    Parameters
    ----------
    base_cmap : str or matplotlib.colors.Colormap
        Either a colormap name or a colormap instance to transform.
    alpha : float, optional
        Fixed alpha value (0 to 1) to apply to the entire colormap.
    scale_alpha : bool, default False
        If True, alpha scales linearly from 0 to 1 across the colormap range.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The new colormap instance with transparency enabled.

    Raises
    ------
    ValueError
        If neither `alpha` is provided nor `scale_alpha` is set to True.
    """
    """Create a colormap with specified alpha transparency.
    
    Args:
        base_cmap: Either a colormap name (str) or a colormap instance
        alpha: Fixed alpha value to apply (float between 0-1)
        scale_alpha: If True, scale alpha from 0 to 1 across the colormap
    """
    if alpha is None and not scale_alpha:
        raise ValueError("Either alpha must be provided or scale_alpha must be True.")
    
    # Handle both string names and colormap instances
    if isinstance(base_cmap, str):
        original_cmap = mpl.colormaps.get_cmap(base_cmap)
        cmap_name = base_cmap
    else:
        original_cmap = base_cmap
        cmap_name = getattr(base_cmap, 'name', 'custom')
    
    colors = original_cmap(np.linspace(0, 1, 256))
    if not scale_alpha:
        colors[:, -1] = alpha  # Set alpha channel
    else:
        colors[:, -1] = np.linspace(0, 1.0, 256)
    
    return mpl.colors.LinearSegmentedColormap.from_list(f'{cmap_name}_alpha', colors)


class BoundingBoxSelector:
    """Interactive bounding box selector for matplotlib axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the bounding box on
    
    Attributes
    ----------
    bbox : dict or None
        The bounding box coordinates as {'x': (xmin, xmax), 'y': (ymin, ymax)} in axes coordinates.
        None if no box has been drawn yet.
    """
    
    def __init__(self, ax, return_square=False):

        def on_select(eclick, erelease):
            self.on_select(eclick, erelease, return_square)
            
        self.ax = ax
        self.lims = [ax.get_xlim(), ax.get_ylim()]
        self.bbox = None
        self.selector = RectangleSelector(
            self.ax,
            on_select, 
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,
            props=dict(facecolor='red', edgecolor='red', alpha=0.3, fill=True, linewidth=2)
        )
        
    def on_select(self, eclick, erelease, return_square=False):
        """Callback for when a region is selected."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Store bounding box coordinates as ranges
        self.bbox = {
            'x': (min(x1, x2), max(x1, x2)),
            'y': (min(y1, y2), max(y1, y2))
        }
        if return_square:
            self.bbox['x'], self.bbox['y'] = self.make_square(self.bbox, self.lims)
        print(f"Bounding box selected: {self.bbox}")
    
    def get_bbox(self):
        """Return the current bounding box coordinates."""
        return self.bbox
    
    def disconnect(self):
        """Disconnect the selector."""
        self.selector.set_active(False)

    @staticmethod
    def make_square(bbox_dict, lims):
        x0, x1 = bbox_dict['x']
        y0, y1 = bbox_dict['y']
        xcen = (x0 + x1) / 2
        ycen = (y0 + y1) / 2
        width = max(x1 - x0, y1 - y0)
        new_x0 = max(xcen - width/2, lims[0][0])
        new_x1 = min(xcen + width/2, lims[0][1])
        new_y0 = max(ycen - width/2, lims[1][0])
        new_y1 = min(ycen + width/2, lims[1][1])
        return [new_x0, new_x1], [new_y0, new_y1]


def color_viz(palette, ncols=10, show_numbers=True, marker_size=500, text_size=10, figsize=None):
    """
    Visualize a color palette as a grid of scatter points.
    
    Parameters
    ----------
    palette : list-like
        List of colors to visualize (can be hex strings, RGB tuples, named colors, etc.)
    ncols : int, optional
        Number of colors per row before wrapping to next line (default: 10)
    show_numbers : bool, optional
        Whether to display the index number on each color point (default: True)
    marker_size : float, optional
        Size of scatter markers (default: 500)
    text_size : float, optional
        Font size for index labels (default: 10)
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated based on ncols and nrows
    
    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    n_colors = len(palette)
    nrows = int(np.ceil(n_colors / ncols))
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (ncols * 0.8, nrows * 0.8)
    
    # Create coordinates for each color
    x_coords = []
    y_coords = []
    colors = []
    
    for i, color in enumerate(palette):
        row = i // ncols
        col = i % ncols
        x_coords.append(col)
        y_coords.append(nrows - 1 - row)  # Top to bottom
        colors.append(color)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x_coords, y_coords, c=colors, s=marker_size, marker='o')
    
    # Add text labels if requested
    if show_numbers:
        for i, (x, y, color) in enumerate(zip(x_coords, y_coords, colors)):
            # Add a check to make sure the color is a hex string for contrast calculation
            if isinstance(color, str) and color.startswith('#'):
                pass
            else:
                # If not a hex string, convert to hex using matplotlib's color conversion
                try:
                    color = mpl.colors.to_hex(color)
                except ValueError:
                    color = '#000000'  # Default to black if conversion fails
            text_color = get_contrast_color(color)
            ax.text(x, y, str(i), ha='center', va='center', 
                   color=text_color, fontsize=text_size, weight='bold')
    
    # Clean up the plot
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(-0.5, nrows - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    return fig, ax


class Ridge:
    """
    A class to create, manage, and plot ridge plots from a DataFrame.

    This class encapsulates the data and methods for generating ridge plots,
    including histogram and KDE computation, peak detection, and normalization.
    """
    def __init__(self, df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None):
        """
        Initializes the Ridge object with data.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the data to plot.
        x : str
            The column name for the x-axis values.
        y : str
            The column name to group by for the ridges on the y-axis.
        hue : str, optional
            The column name for color encoding within each group, by default None.
        """
        self.df = df
        self.x = x
        self.y = y
        self.hue = hue
        self.data = None
        self.breaks = None
        self.normalized_df = None
        self.target_anchors = None
        self.order = None


    def _get_kde(self, n, bins, n_points=1000, log_scale=False, **kwargs):
        """
        Compute the Kernel Density Estimate (KDE) from histogram data.

        Parameters
        ----------
        n : np.ndarray
            The counts for each bin of the histogram.
        bins : np.ndarray
            The bin edges of the histogram.
        n_points : int, optional
            The number of points to evaluate the KDE on, by default 1000.
        log_scale : bool, optional
            Whether the data is on a log scale, by default False.
        **kwargs
            Additional keyword arguments passed to `scipy.stats.gaussian_kde`.

        Returns
        -------
        tuple
            A tuple containing the x-grid and KDE values.
        """
        # Compute the midpoints of the bins
        bin_midpoints = (bins[:-1] + bins[1:]) / 2

        # Repeat the midpoints according to the counts
        data = np.repeat(bin_midpoints, n)

        if log_scale:
            # Transform data to log10 scale, handling non-positive values
            data = np.log10(data[data > 0])

        # Compute the KDE
        if len(data) == 0 or data.size <= 1:
            return np.array([]), np.array([])
        kde = gaussian_kde(data, **kwargs)

        # Evaluate the KDE on a grid in log space if necessary
        if log_scale:
            min_bin = bins[0] if bins[0] > 0 else 1e-9
            max_bin = bins[-1]
            x_grid = np.linspace(np.log10(min_bin), np.log10(max_bin), n_points)
        else:
            x_grid = np.linspace(bins[0], bins[-1], n_points)

        kde_values = kde(x_grid)

        if log_scale:
            # Transform x_grid back to base 10 scale
            x_grid = np.power(10, x_grid)

        return x_grid, kde_values


    def add_ridge_data(self, hist=False, kde=False, bins=100, log_bins=False, order=None, kde_params=None):
        """
        Computes histogram and/or KDE data for the ridge plot and adds it to the instance.

        Parameters
        ----------
        hist : bool, optional
            Whether to compute histogram data, by default False.
        kde : bool, optional
            Whether to compute KDE data, by default False.
        bins : int or array-like, optional
            The number of bins or pre-computed bins for the histogram, by default 100.
        log_bins : bool, optional
            Whether to use log-scaled bins, by default False.
        order : list, optional
            The order for the ridges on the y-axis, by default None.
        kde_params : dict, optional
            Parameters for KDE computation, by default None.
        """
        if not hist and not kde:
            raise ValueError("At least one of 'hist' or 'kde' must be True.")

        plotby_unique = self.df[self.y].unique()
        if order is not None:
            assert all([i in order for i in plotby_unique]), 'All unique values in y must be in order'
            plotby_unique = order
        
        self.order = plotby_unique

        if self.data is None:
            self.data = {}

        for unique_y in plotby_unique:
            df_subset = self.df[self.df[self.y] == unique_y]
            if unique_y not in self.data:
                self.data[unique_y] = {}

            hue_values = df_subset[self.hue].unique() if self.hue else [None]
            for hue_value in hue_values:
                if self.hue:
                    values = df_subset[self.x][df_subset[self.hue] == hue_value].values
                else:
                    values = df_subset[self.x].values

                if isinstance(bins, int):
                    if log_bins:
                        min_val = values.min() if len(values) > 0 else 1e-9
                        if min_val <= 0: min_val = 1e-9
                        max_val = values.max() if len(values) > 0 else 1
                        nphist_bins = np.logspace(np.log10(min_val), np.log10(max_val), bins)
                    else:
                        min_val = values.min() if len(values) > 0 else 0
                        max_val = values.max() if len(values) > 0 else 1
                        nphist_bins = np.linspace(min_val, max_val, bins)
                else:
                    nphist_bins = bins
                
                key = hue_value if self.hue else 'default'
                if key not in self.data[unique_y]:
                    self.data[unique_y][key] = {}

                if hist:
                    nphist_n, _ = np.histogram(values, bins=nphist_bins)
                    self.data[unique_y][key]['hist'] = (nphist_n, nphist_bins)
                
                if kde:
                    if 'hist' in self.data[unique_y][key]:
                        nphist_n, nphist_bins = self.data[unique_y][key]['hist']
                    else: # hist must be computed for kde
                        nphist_n, _ = np.histogram(values, bins=nphist_bins)
                    x_grid, kde_values = self._get_kde(nphist_n, nphist_bins, log_scale=log_bins, **(kde_params or {}))
                    self.data[unique_y][key]['kde'] = (x_grid, kde_values)


    def find_breaks(self, breaktype="peaks", height=0.01, distance=10, prominence=0.01, window=3, tolerance=0.001, buffer=0.05):
        """
        Finds natural breaks (peaks, valleys, or plateaus) in the KDE values and adds them to the instance.

        Parameters
        ----------
        breaktype : str, optional
            Type of breaks to find. Options are "peaks", "valleys", or "plateaus", by default "peaks".
        height : float, optional
            Required height of peaks, by default 0.01.
        distance : int, optional
            Required minimal horizontal distance in samples between neighbouring peaks, by default 10.
        prominence : float, optional
            Required prominence of peaks, by default 0.01.
        window : int, optional
            Window size for moving average when finding plateaus, by default 1.
        tolerance : float, optional
            Tolerance of KDE slope to be within zero for plateau detection, by default 0.01.
        buffer : float, optional
            Buffer distance (as a proportion of max KDE x values) to skip before scanning for plateau, by default 0.05.

        Raises
        ------
        ValueError
            If KDE data has not been computed yet or if breaktype is not recognized.
        """
        if self.data is None:
            raise ValueError("KDE data not available. Please run `add_ridge_data` with `kde_params` first.")

        if breaktype not in ["peaks", "valleys", "plateaus"]:
            raise ValueError("breaktype must be 'peaks', 'valleys', or 'plateaus'.")

        breaks = []
        for feature in self.data:
            for sample in self.data[feature]:
                x_values = self.data[feature][sample]['kde'][0]
                y_values = self.data[feature][sample]['kde'][1]
                if not y_values.any(): continue
                
                if breaktype == "valleys":
                    y_values = -(y_values - y_values.max())
                
                if breaktype in ["peaks", "valleys"]:
                    detected_peaks, _ = find_peaks(x=y_values, height=height, distance=distance, prominence=prominence)
                    if len(detected_peaks) > 0:
                        breaks.append((feature, sample, *x_values[detected_peaks]))
                        
                elif breaktype == "plateaus":
                    # Find the first peak
                    detected_peaks, _ = find_peaks(x=y_values, height=height, distance=distance, prominence=prominence)
                    if len(detected_peaks) > 0:
                        first_peak_idx = detected_peaks[0]
                        max_x_value = np.max(x_values)
                        min_x_value = np.min(x_values)
                        slope_tolerance = tolerance
                        
                        # Determine starting point for scanning, accounting for buffer
                        scan_start_idx = first_peak_idx + window
                        if buffer is not None:
                            # Find the index that corresponds to the buffer distance
                            peak_x_value = x_values[first_peak_idx]
                            buffer_x_value = peak_x_value + buffer*(max_x_value - min_x_value)
                            # Find the closest index to the buffer distance
                            buffer_idx = np.argmin(np.abs(x_values - buffer_x_value))
                            scan_start_idx = max(scan_start_idx, buffer_idx)
                        
                        # Scan rightward from the starting point to find where it levels off (plateau start)
                        plateau_start_idx = first_peak_idx  # Default to peak if no plateau found
                        for i in range(scan_start_idx, len(y_values) - window):
                            # Calculate moving window average slope
                            window_y = y_values[i-window:i+window+1]
                            window_x = np.arange(len(window_y))
                            if len(window_y) > 1:
                                slope = np.polyfit(window_x, window_y, 1)[0]
                                if abs(slope) <= slope_tolerance:
                                    plateau_start_idx = i
                                    break
                        
                        # # Add both the peak and plateau start
                        # plateau_points = [x_values[first_peak_idx], x_values[plateau_start_idx]]
                        
                        # Add just the plateau start
                        plateau_points = [x_values[plateau_start_idx]]

                        breaks.append((feature, sample, *plateau_points))
        
        if not breaks:
            self.breaks = pd.DataFrame(columns=['feature', 'sample']).set_index(['feature', 'sample'])
            return

        breaks_df = pd.DataFrame(breaks)
        cols_length = len(breaks_df.columns)
        breaks_df.columns = ['feature', 'sample'] + [i for i in range(cols_length - 2)]
        breaks_df = breaks_df.set_index(['feature', 'sample'])
        self.breaks = breaks_df


    def normalize_by_anchors(self, anchor_dict: Dict[str, Tuple[float, ...]]):
        """
        Normalizes features in the DataFrame based on provided anchor points.

        This function performs a piecewise linear transformation for each specified
        feature (column) to align its anchor points with a target set of anchors.
        The results are stored in `self.normalized_df` and `self.target_anchors`.

        Parameters
        ----------
        anchor_dict : Dict[str, Tuple[float, ...]]
            A dictionary mapping feature names (column names) to a tuple
            of sorted anchor points. E.g., {'feature_A': (10, 50, 90)}.

        Raises
        ------
        ValueError
            If anchor_dict is invalid or features are not in the data.
        """
        if not anchor_dict:
            raise ValueError("Anchor dictionary is empty. Please provide valid anchor points for normalization.")

        it = iter(anchor_dict.values())
        first_len = len(next(it))
        if not all(len(val) == first_len for val in it):
            raise ValueError("All anchor point tuples in the dictionary must have the same length.")
        if first_len < 2:
            raise ValueError("Anchor point tuples must have at least two points.")

        for feature_name in anchor_dict.keys():
            if feature_name not in self.df.columns:
                raise ValueError(f"Feature '{feature_name}' from anchor_dict not found in data_matrix columns.")

        target_anchors = min(anchor_dict.values(), key=lambda anchors: anchors[0])
        normalized_matrix = self.df.copy()

        for feature_name, source_anchors in anchor_dict.items():
            source_anchors = tuple(sorted(source_anchors))
            original_values = normalized_matrix[feature_name].to_numpy(dtype=float)
            new_values = np.zeros_like(original_values)

            lower_bound = source_anchors[0]
            lower_shift = target_anchors[0] - lower_bound
            mask_below = original_values < lower_bound
            new_values[mask_below] = original_values[mask_below] + lower_shift

            upper_bound = source_anchors[-1]
            upper_shift = target_anchors[-1] - upper_bound
            mask_above = original_values > upper_bound
            new_values[mask_above] = original_values[mask_above] + upper_shift

            for i in range(len(source_anchors) - 1):
                x1, x2 = source_anchors[i], source_anchors[i+1]
                y1, y2 = target_anchors[i], target_anchors[i+1]
                
                if i == 0:
                    mask_segment = (original_values >= x1) & (original_values <= x2)
                else:
                    mask_segment = (original_values > x1) & (original_values <= x2)

                segment_values = original_values[mask_segment]
                
                if x1 == x2:
                    new_values[mask_segment] = y1
                else:
                    scale = (y2 - y1) / (x2 - x1)
                    new_values[mask_segment] = y1 + (segment_values - x1) * scale
            
            normalized_matrix[feature_name] = new_values

        self.normalized_df = normalized_matrix
        self.target_anchors = target_anchors


    def plot(self, log_bins=False, colors=None, alpha=0.8, hspace=-0.5,
            #  label_pos='bottom_right', 
             hist=True, kde=False, fill_between=False,
             legend=False, legend_params=None, subplots_kwargs=None,
             default_tick_color='k', num_cols=1):
        """
        Generate and display the ridge plot.

        Parameters
        ----------
        log_bins : bool, optional
            Whether to use log-scaled bins on the x-axis, by default False.
        colors : iterable, optional
            An iterable of colors for the plots, by default None.
        alpha : float, optional
            The alpha transparency for histograms and KDE fills, by default 0.8.
        hspace : float, optional
            The height space between subplots, by default -0.5.
        label_pos : str, optional
            Position of the y-labels (e.g., 'bottom_right'), by default 'bottom_right'.
        hist : bool, optional
            Whether to plot the histogram, by default True.
        kde : bool, optional
            Whether to plot the KDE curve, by default False.
        fill_between : bool, optional
            Whether to fill the area under the KDE curve, by default False.
        legend : bool, optional
            Whether to display a legend, by default False.
        legend_params : dict, optional
            Parameters for legend configuration, by default None.
        subplots_kwargs : dict, optional
            Keyword arguments for `svytools.genplot.subplots`, by default None.
        default_tick_color : str, optional
            Default color for axis ticks when no hue is used, by default 'k'.
        num_cols : int, optional
            Number of columns to arrange the ridge plots in, by default 1.

        Returns
        -------
        tuple
            A tuple containing the matplotlib Figure and Axes objects.
            
        Raises
        ------
        ValueError
            If required data for plotting has not been computed yet.
        """
        # Local import to avoid a circular import between genplot and decorate
        from svytools.decorate import decorate_plot, get_pm

        if self.data is None:
            raise ValueError("Plot data not available. Please run `add_ridge_data` first.")
        if hist and any('hist' not in v.get(key, {}) for v in self.data.values() for key in v):
            raise ValueError("Histogram data not available. Please run `add_ridge_data` with `hist=True` to compute it.")
        if kde and any('kde' not in v.get(key, {}) for v in self.data.values() for key in v):
            raise ValueError("KDE data not available. Please run `add_ridge_data` with `kde=True` to compute it.")

        plotby_unique = list(self.data.keys())

        default_subplots_kwargs = {'ar': 5, 'fss': 1.5}
        restricted_subplots_kwargs = ['num_plots', 'max_cols', 'sharex']
        if subplots_kwargs is None:
            subplots_kwargs = {}
        if any([i in subplots_kwargs for i in restricted_subplots_kwargs]):
            raise ValueError('Restricted keyword arguments: %s' % restricted_subplots_kwargs)
        subplots_kwargs = {**default_subplots_kwargs, **subplots_kwargs}

        # Calculate layout for multiple columns
        if num_cols > 1:
            num_rows = int(np.ceil(len(plotby_unique) / num_cols))
            fig, axes = subplots(len(plotby_unique), ncols=num_cols, sharex=True, **subplots_kwargs)
            axes_flat = axes.flat
        else:
            fig, axes = subplots(len(plotby_unique), 1, sharex=True, **subplots_kwargs)
            if len(plotby_unique) == 1:
                axes = np.array([axes])
            axes_flat = axes.flat
            
        if log_bins:
            if num_cols > 1:
                # Set xscale for all axes in the first row
                for ax in axes[0, :]:
                    ax.set_xscale('log')
            else:
                axes[0].set_xscale('log')
        if colors is None:
            colors = it.cycle(sns.color_palette())

        color_iter = iter(colors) 

        for unique_y, ax in zip(plotby_unique, axes_flat):
            if self.hue:
                hue_values = list(self.data[unique_y].keys())
                hue_colors = [next(color_iter) for _ in hue_values]
            else:
                hue_values = [None]
                c = next(color_iter)
                hue_colors = [c]

            labels = []

            for i, (hue_value, hue_color) in enumerate(zip(hue_values, hue_colors)):
                key = hue_value if self.hue else 'default'
                if key not in self.data[unique_y]: continue

                if self.hue:
                    labels.append(f'{hue_value}')
                else:
                    labels.append(str(unique_y))

                max_hist = None
                if hist:
                    nphist_n, nphist_bins = self.data[unique_y][key]['hist']
                    ax.hist(nphist_bins[:-1], bins=nphist_bins, weights=nphist_n, alpha=alpha,
                            facecolor=hue_color, linewidth=0, zorder=i)
                    max_hist = np.max(nphist_n)
                    
                if kde:
                    x_grid, kde_values = self.data[unique_y][key]['kde']
                    if len(x_grid) == 0: continue

                    kde_values_scaled = kde_values.copy()
                    if hist and max_hist is not None and np.max(kde_values_scaled) > 0:
                        kde_values_scaled *= max_hist / np.max(kde_values_scaled)
                    
                    ax.plot(x_grid, kde_values_scaled, color=hue_color, zorder=i)

                    if fill_between:
                        ax.fill_between(x_grid, kde_values_scaled, color=hue_color, alpha=alpha, zorder=i)

            if legend:
                update_legend_defaults = {'bbox_to_anchor': (1, 0), 'loc': 'lower right'}
                default_config = {'title': self.hue if self.hue else None, 'title_fontsize': 8}

                default_legend_pm = get_pm('legend', update=update_legend_defaults)
                preconfig = default_legend_pm.get_params(legend_params)
                config = get_config(preconfig, default_config)
                
                if self.hue:
                    cdict = dict(zip(hue_values, hue_colors))
                    ax = decorate_plot(ax, plot_type='legend', config=config, cdict=cdict)
                else:
                    cdict = {str(unique_y): c}
                    ax = decorate_plot(ax, plot_type='legend', config=config, cdict=cdict)

            
            # coords = {'bottom': 0.1, 'left': 0.1, 'top': 0.9, 'right': 0.9, 'center': 0.5}
            # va, ha = label_pos.split('_')

            axis_elements_color = default_tick_color if self.hue else c

            # ax.text(coords[ha], coords[va], '\n'.join(labels), transform=ax.transAxes,
            #         ha=ha, va=va, color=plot_colors, fontweight='bold')


            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(axis_elements_color)
            ax.spines['bottom'].set_linewidth(3)
            if len(ax.get_yticks()) > 1:
                max_ytick = ax.get_yticks()[-1]
                ax.set_yticks([0, max_ytick])
                ax.set_yticklabels([None, "{:.3g}".format(max_ytick)], color=axis_elements_color, fontweight='bold')
            ax.set_facecolor((0, 0, 0, 0))

        fig.subplots_adjust(hspace=hspace)
        ax.set_xlabel(self.x)

        return fig, axes
