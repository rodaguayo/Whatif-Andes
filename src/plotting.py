"""Shared Plotly styling for the figures/ notebooks.

Centralizes the house style (fonts, colors, layout defaults) and the small
pieces of boilerplate (basin-map geo styling, region-boundary ruler, panel
labels, figure export) that were previously copy-pasted across notebooks.
"""

from pathlib import Path

import plotly.graph_objects as go

# Typography
FONT_FAMILY = "Arial"
BASE_FONT_SIZE = 14

# Core palette
PLOT_BGCOLOR = "rgba(229,236,246,0.7)"   # light-blue plot background
WATER_COLOR = "#ecf1f8"                  # ocean/lake fill on basin maps
GEO_BGCOLOR = "rgba(210, 226, 230, 0.7)" # geo subplot background
FRAME_COLOR = "#83888f"                  # thin panel borders / ruler lines
MARKER_LINE_COLOR = "#9a9a9a"            # marker / bar outlines
MUTED_TEXT_COLOR = "#474747"             # secondary annotation / axis text
PANEL_LABEL_COLOR = "#181818"

POS_COLOR, NEG_COLOR = "#C7231D", "#2E86AB"  # positive/negative delta pairs
REFERENCE_LINE_COLOR = "#BAC7DA"             # dotted 1:1 / threshold reference lines

DEFAULT_MARGIN = dict(l=40, r=20, b=20, t=20)

# Sequential yellow -> green -> blue -> purple colorscale (viridis-like),
# used for glacier-area / performance-metric color axes.
SEQUENTIAL_COLORSCALE = [
    (0.0, "#fde725"),
    (0.1, "#6ece58"),
    (0.2, "#35b779"),
    (0.6, "#31688e"),
    (1.0, "#440154"),
]


def scenario_colors(cfg):
    """Warming-level scenario colors, in cfg['scenarios'] order."""
    return [s["color"] for s in cfg["scenarios"]]


# House-style template: dotted white gridlines on the light-blue plot background,
# outside ticks, no zerolines, Arial. Figure-specific fig.update_xaxes/yaxes(...)
# calls only need to set what's actually different (title, range, dtick, ...).
TEMPLATE = go.layout.Template(
    layout=go.Layout(
        xaxis=dict(ticks="outside", griddash="dot", gridcolor="white", zeroline=False),
        yaxis=dict(ticks="outside", griddash="dot", gridcolor="white", zeroline=False),
        font=dict(family=FONT_FAMILY, size=BASE_FONT_SIZE),
        plot_bgcolor=PLOT_BGCOLOR,
        margin=dict(DEFAULT_MARGIN),
    )
)


def base_layout(**overrides):
    """Default go.Figure layout kwargs shared across figures (house template + overrides).

    Usage: fig.update_layout(**base_layout(width=900, height=700))
    """
    layout = dict(template=TEMPLATE)
    layout.update(overrides)
    return layout


def geo_style(lonaxis_range, lataxis_range=(-56, -7),
              countrycolor="rgba(0,0,0,0.2)", countrywidth=0.2):
    """Shared basemap styling for go.Scattergeo panels (river/basin maps).

    Usage: fig.update_geos(**geo_style([-79, -67]), row=1, col=1)
    """
    return dict(
        showframe=True,
        framecolor=WATER_COLOR, bgcolor=GEO_BGCOLOR,
        landcolor="white", showland=True, resolution=50,
        showlakes=True, showocean=True, lakecolor=WATER_COLOR, oceancolor=WATER_COLOR,
        showcoastlines=True, coastlinecolor="rgba(0,0,0,0)",
        showcountries=True, countrycolor=countrycolor, countrywidth=countrywidth,
        lonaxis_range=list(lonaxis_range), lataxis_range=list(lataxis_range),
    )


def add_region_boundary_ruler(fig, regions, boundary_lats, x=0.05,
                               lat_range=(-56, -7), tick_half_width=0.005,
                               line_y_range=(0.005, 0.995), line_color=FRAME_COLOR,
                               label_bgcolor=WATER_COLOR, font_size=14, borderpad=1):
    """Draw the vertical region-boundary ruler (line + ticks + labels) used
    alongside latitude-ordered maps/panels.

    `regions` is cfg["regions"] (name -> [lat_min, lat_max]); `boundary_lats`
    are the tick latitudes, given explicitly since figures nudge the outer
    ticks slightly inward from the axis edges for legibility.
    """
    lat_min, lat_max = lat_range

    def y_pos(lat):
        return (lat - lat_min) / (lat_max - lat_min)

    fig.add_shape(type="line", x0=x, x1=x, y0=line_y_range[0], y1=line_y_range[1],
                  xref="paper", yref="paper", line=dict(width=1, color=line_color))

    for boundary_lat in boundary_lats:
        yp = y_pos(boundary_lat)
        fig.add_shape(type="line", x0=x - tick_half_width, x1=x + tick_half_width,
                      y0=yp, y1=yp, xref="paper", yref="paper",
                      line=dict(width=1, color=line_color))

    for region, (rmin, rmax) in regions.items():
        yp = y_pos((rmin + rmax) / 2)
        fig.add_annotation(
            text=region.replace(" ", "<br>"), bgcolor=label_bgcolor, borderpad=borderpad,
            x=x, y=yp, xref="paper", yref="paper", showarrow=False,
            font=dict(size=font_size, family=FONT_FAMILY),
            xanchor="center", yanchor="middle",
        )


def add_panel_label(fig, text, x, y=0.995, size=16, color=PANEL_LABEL_COLOR,
                     xanchor="right", yanchor="top"):
    """Bold a)/b)/c) panel tag, in paper-relative coordinates."""
    fig.add_annotation(text=f"<b>{text}</b>", xref="paper", yref="paper", x=x, y=y,
                        showarrow=False, font=dict(size=size, family=FONT_FAMILY, color=color),
                        xanchor=xanchor, yanchor=yanchor)


def save_figure(fig, path, width, height, scale=None):
    """Write a figure to disk at the given pixel size, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(width=width, height=height)
    if scale is not None:
        kwargs["scale"] = scale
    fig.write_image(str(path), **kwargs)
