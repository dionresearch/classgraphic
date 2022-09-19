"""common functions for table and viz."""
from collections import Counter

import numpy as np
import pandas as pd

import plotly.express as px

from classgraphic.template import pio


def complementary_hex_color(rgb):
    """Return Complementary hexadecimal color of hex or rgb color.

    Args:
        rgb (str): color to convert, formatted as either  `#aabbcc` or `rgb(111,222,123)`

    Returns:
        hex color (str)
    """
    if rgb[0] == "#":
        # hex, remove hash
        rgb = rgb[1:]
        comp = ["%02X" % (255 - int(a, 16)) for a in (rgb[0:2], rgb[2:4], rgb[4:6])]
    elif rgb[:3] == "rgb":
        # rgb already in 0-255 range
        rgb = rgb[4:-1].split(",")
        comp = ["%02X" % (255 - int(a)) for a in rgb]
    return "".join(["#"] + comp)


def get_labels(model, y, as_str=True):
    """Get labels from the model.

    We use a one hot strategy and ultimately get the resulting column names.

    Args:
        model: scikit-learn model
        y (Series or array): the target column for classification
        as_str (bool): default to list of string, else return one hot encoded dataframe

    Returns:
        (list(str) or dataframe)
    """
    onehot = pd.get_dummies(y, columns=model.classes_)
    if as_str:
        # in case we want labels as string
        labels = sorted(str(val) for val in onehot.columns)
    else:
        labels = onehot
    return labels


def heatmap_colorscale(colorscale=None, template=None):
    """Get heatmap colorscale from colorscale or template.

    Args:
        colorscale (str or list): plotly named colorscale (`blues`, `burgyl_r`...) or 2d list mapping
        template (str): template name to use to get heatmap colorscale from.

    Returns:
        list (colorscale)
    """
    template = table_template(template)

    # get colors from colorscale if it was specified
    if type(colorscale) == list:
        colorscale_list = colorscale
    elif colorscale:
        colorscale_list = px.colors.get_colorscale(colorscale)

    if colorscale and len(colorscale_list) < 4:
        raise ValueError("colorscale must have at least 4 values.")

    # get colors from template heatmap colorscale
    if colorscale is None:
        colorscale_list = pio.templates[template].data.heatmap[0].colorscale

    return colorscale_list


def light_theme(colorscale_list):
    """Evaluate if a theme has a colorscale going from dark to light.

    Args:
        colorscale_list (list): list of rgb or hex colors.

    Returns:
        bool
    """
    rgb = colorscale_list[-1][1]
    if rgb[0] == "#":
        # hex, remove hash
        rgb = rgb[1:]
        values = [int(a, 16) for a in (rgb[0:2], rgb[2:4], rgb[4:6])]
    elif rgb[:3] == "rgb":
        # rgb already in 0-255 range
        values = [int(a) for a in rgb[4:-1].split(",")]
    else:
        return False
    lightness = sum(values) / (3.0 * 255)
    return lightness > 0.5


def missing_matrix(X):
    """Find all missing (None or NaN) values.

    For each cell, returns True if value is missing, False if not.

    Args:
        X (DataFrame or 2d array): data array to evaluate for missing values.

    Returns:
        array of bool
    """
    if type(X) == pd.DataFrame:
        return X.isnull()
    return np.isnan(X)


def table_template(template=None):
    """Get template to use with a table.

    Args:
        template (str): template name. Template has to be valid. If not specified, will use plotly default template.

    Returns:
        template (str)
    """
    if template is None:
        default_template = pio.templates.default
        return default_template
    else:
        if template in pio.templates:
            return template
        else:
            raise ValueError(f"{template} unavailable. Choose from:\n{pio.templates}")


def value_count(y_list):
    """Get class counts.

    Args:
        y_list (DataFrame or array): usually the target data (y, y_test etc)

    Returns:
        counts (dict)
    """
    count_dict = Counter([str(val) for val in y_list])
    count_dict = {k: [v] for k, v in count_dict.items()}
    return count_dict


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=500, random_state=0)
    print(pd.DataFrame(value_count(y)).T.reset_index())

    # add 20 missing values
    index = np.random.choice(X.size, 50, replace=False)  # 10%
    X.ravel()[index] = np.nan

    np.set_printoptions(threshold=np.inf)
    print(missing_matrix(X))

    print(table_template())
    print(complementary_hex_color("#112233"))
    print(table_template("plotly_dark"))
    try:
        print(table_template("potly_dark"))
    except ValueError as ex:
        print(ex)
    print(heatmap_colorscale(template="plotly"))
    print(light_theme(heatmap_colorscale(template="plotly")))
    print(heatmap_colorscale(template="ggplot2"))
    print(light_theme(heatmap_colorscale(template="ggplot2")))
    print(heatmap_colorscale(template="plotly_dark"))
    print(light_theme(heatmap_colorscale(template="plotly_dark")))
