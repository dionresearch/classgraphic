"""template.py.

Various templates in `plotly` format.
"""
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px


#: Logo template
DION_LOGO = go.layout.Template(
    layout_images=[
        dict(
            visible=True,
            source="http://dionresearch.com/img/dion_research_logo_small_transparent.png",
            layer="below",
            sizex=0.7,
            sizey=0.7,
            xref="paper",
            yref="paper",
            sizing="contain",
            opacity=0.1,
            x=0.5,
            y=0.9,
            xanchor="center",
            yanchor="top",
            name="logo",
        )
    ]
)

#: Hide Legend
HIDE_LEGEND = go.layout.Template(layout_showlegend=False)

#: sequential colorscale
seq_colorscale = px.colors.get_colorscale("blues")

#: Dion Research blue gradient template
DION_TEMPLATE = {
    "data": {
        "heatmap": [
            {
                "colorbar": {"outlinewidth": 0, "ticks": ""},
                "colorscale": seq_colorscale,
                "type": "heatmap",
            }
        ],
        "heatmapgl": [
            {
                "colorbar": {"outlinewidth": 0, "ticks": ""},
                "colorscale": seq_colorscale,
                "type": "heatmapgl",
            }
        ],
        "histogram": [
            {
                "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
                "type": "histogram",
            }
        ],
        "pie": [{"automargin": True, "type": "pie"}],
        "table": [
            {
                "cells": {
                    "fill": {"color": seq_colorscale[0][1]},
                    "line": {"color": "black"},
                },
                "header": {
                    "fill": {"color": seq_colorscale[3][1]},
                    "line": {"color": "black"},
                },
                "type": "table",
            }
        ],
    },
    "layout": {
        "colorscale": {
            "sequential": seq_colorscale,
        },
        "font": {"color": seq_colorscale[-1][1]},
        "title": {"x": 0.05},
        "title_font_size": 24,
        "margin": {"t": 140, "b": 40},
    },
}

pio.templates["dion_logo"] = go.layout.Template(DION_LOGO)
pio.templates["dion"] = go.layout.Template(DION_TEMPLATE)
pio.templates["hide_legend"] = go.layout.Template(HIDE_LEGEND)


def set_default_template(template="dion"):
    """set_default_template.

    Args:
        template (str): template to use for all charts. Defaults to `DEFAULT_TEMPLATE`.
    """
    pio.templates.default = template
