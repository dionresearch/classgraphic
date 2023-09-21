"""Classification tables, in plotly and pandas format."""
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
)

from classgraphic.common import (
    complementary_hex_color,
    get_labels,
    heatmap_colorscale,
    light_theme,
    pio,
    value_count,
)


def class_imbalance_table(
    y,
    y2=None,
    condition=None,
    sort=False,
    ascending=False,
    extended=False,
    colorscale=None,
    template=None,
    **kwargs,
):
    """Class imbalance table.

    Given a `target` Series or array (eg. y, y_test, y_train), this visualize the balance/imbalance of
    classes.

    Args:
        y (series or array): usually, the target column for classification
        y2 (series or array): optional. If provided an additional count column will allow for comparison
        condition (str): added to title. Something like 'train', 'test', 'train and test', 'real and test'
        sort (bool): if True, data will be sorted by class count
        ascending (bool): if True, sort ascending, else descending
        extended (bool): return a dataframe, besides the Plotly table figure
        colorscale (str or list): plotly named colorscale (`blues`, `burgyl_r`...) or 2d list mapping
        template (str): template name to use - uses heatmap `colorscale`, not table colors to provide alternating colors
        **kwargs: any Plotly kwargs for go.Table

    Returns:
        Figure (, Dataframe if extended)
    """
    if condition is None:
        condition = ""
    else:
        condition = " - " + condition  # add separator

    # data wrangling
    values = value_count(y)  # dict of classes, count as list, for Pandas
    df = pd.DataFrame(values).T.reset_index()
    df.columns = ["class", "count(y)"]

    if y2 is not None:
        # we are comparing
        values2 = value_count(y2)
        df2 = pd.DataFrame(values2).T.reset_index()
        df2.columns = ["class", "count(y2)"]
        df = pd.merge(df, df2, on="class", how="outer")

    if sort:
        df.sort_values(by="class", ascending=ascending, inplace=True)
    total = y.shape

    title = (
        f"<b>Data set classes{condition}</b><br><sub>Total observations: {total})</sub>"
    )

    fig = table(df, title=title, format=[None, ".3~f", ".3~f", ".3~f", None], named_index=False,
                colorscale=colorscale, template=template)

    if extended:
        return fig, df
    return fig


def classification_table(model, y, y_pred, extended=False, colorscale=None, template=None, **kwargs):
    """Classification report table.

    Args:
        y (series or array): usually, the target column for classification
        y_pred (series or array): predicted values
        condition (str): added to title. Something like 'train', 'test', 'train and test', 'real and test'
        sort (bool): if True, data will be sorted by class count
        ascending (bool): if True, sort ascending, else descending
        extended (bool): return a dataframe, besides the Plotly table figure
        colorscale (str or list): plotly named colorscale (`blues`, `burgyl_r`...) or 2d list mapping
        template (str): template name to use - uses heatmap `colorscale`, not table colors to provide alternating colors
        **kwargs: any Plotly kwargs for go.Table

    Returns:
        Figure (, Dataframe if extended)
    """
    # data wrangling
    labels = get_labels(model, y, as_str=False).columns

    # data wrangling
    cr_dict = classification_report(y, y_pred, labels=labels, output_dict=True)
    accuracy = cr_dict.pop("accuracy")
    balanced_accuracy = balanced_accuracy_score(
        y,
        y_pred,
    )
    df = pd.DataFrame.from_dict(cr_dict).T
    df.reset_index(inplace=True)
    metric = df["index"].copy()
    df["index"] = "<b>" + df["index"] + "</b>"
    df.rename({"index": "metric"}, axis=1, inplace=True)

    # plot
    if round(accuracy, 3) == round(balanced_accuracy, 3):
        title = (
            f"<b>{model} - Classification Report</b><br>"
            f"<sub>Accuracy: {accuracy:.3f}"
        )
    else:
        title = f"<b>{model} - Classification Report</b><br>" \
                f"<sub>Accuracy: {accuracy:.3f}, Balanced Accuracy: {balanced_accuracy:.3f}"

    fig = table(df, title, format=[None, ".3~f", ".3~f", ".3~f", None], colorscale=colorscale, template=template, **kwargs)
    if extended:
        df["metric"] = metric
        return fig, df
    return fig


def confusion_matrix_table(
    model,
    y,
    y_pred,
    normalize=None,
    extended=False,
    colorscale=None,
    template=None,
    **kwargs,
):
    """Confusion matrix table.

    Args:
        model: scikit-learn model
        y (series or array): the target column for classification
        y_pred (series or array): predicted values
        normalize (str)): one of None, 'true' (or synonym 'real'), 'pred', 'all'
        extended (bool): return a dataframe, besides the Plotly table figure
        colorscale (str or list): plotly named colorscale (`blues`, `burgyl_r`...) or 2d list mapping
        template (str): template name to use - uses heatmap `colorscale`, not table colors to provide alternating colors
        **kwargs: any Plotly kwargs for go.Heatmap

    Returns:
        Figure (table)(, Dataframe if extended)
    """
    # figure out what colorscale to use
    colorscale_list = heatmap_colorscale(colorscale, template)
    light_scale = light_theme(colorscale_list)

    # data wrangling
    labels = get_labels(model, y, as_str=False).columns
    x_labels = [str(label) for label in labels]
    y_labels = [str(label) for label in labels]

    if normalize is None:
        normalized = ""
    else:
        normalized = f"Normalized ({normalize}) - "
    if normalize == "real":
        # CM doesn't know about our synonym
        normalize = "true"
    cm_array = confusion_matrix(y, y_pred, labels=labels, normalize=normalize)
    # (tn, fp), (fn, tp) = cm_array
    df = pd.DataFrame(cm_array)
    df.columns = x_labels
    df.index = y_labels
    mcc = matthews_corrcoef(y, y_pred)
    max = cm_array.max()

    title = f"<b>{model} - Confusion Matrix</b><br><sub>{normalized}Matthews Correlation Coefficient (MCC)={mcc:.3f}</sub>"
    # plot
    data = go.Heatmap(
        z=cm_array,
        x=x_labels,
        y=y_labels,
        xgap=2,
        ygap=2,
        showscale=False,
        colorscale=colorscale_list,
        **kwargs,
    )
    annotations = []

    for real, row in enumerate(cm_array):
        for pred, value in enumerate(row):
            if real < pred:
                hovertext = "FP"
            else:
                hovertext = "FN"
            if real == pred == 1:
                hovertext = "TP"
            elif real == pred:
                # both 0
                hovertext = "TN"
            if cm_array.shape[0] > 2:
                hovertext = None
            if light_scale:
                text_color = "black" if value > max / 2 else "white"
            else:
                text_color = "black" if value < max / 2 else "white"
            annotations.append(
                {
                    "x": x_labels[pred],
                    "y": y_labels[real],
                    "font": {
                        "color": text_color,
                        "size": 24,
                    },
                    "text": str(value),
                    "xref": "x1",
                    "yref": "y1",
                    "hovertext": hovertext,
                    "showarrow": False,
                }
            )
    layout = {
        "title": title,
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
        "annotations": annotations,
    }
    fig = go.Figure(data=data, layout=layout)

    if extended:
        return fig, df
    return fig


def describe(X, colorscale=None, template=None, transpose=False, **kwargs):
    """Dataframe description table.

    This is equivalent to pandas `df.describe()`, with extra quantiles and non numerical feature
    stats, such as top and frequency.

    Args:
        X (dataframe or array): tabular data
        colorscale (str or list): plotly named colorscale (`blues`, `burgyl_r`...) or 2d list mapping
        template (str): template name to use - uses heatmap `colorscale`, not table colors to provide alternating colors
        transpose (bool): columns become rows, rows become columns
        **kwargs: any Plotly kwargs for `go.Figure`

    Returns:
        Figure (table)
    """
    if type(X) == pd.DataFrame:
        df = X
    elif type(X) == pd.Series:
        df = X.to_frame()
    else:
        df = pd.DataFrame(X)

    try:
        stats = df.describe(percentiles=[.01, .25, .5, .75, .99], include="all", datetime_is_numeric=True).T
    except TypeError: # pandas 2.x
        stats = df.describe(percentiles=[.01, .25, .5, .75, .99], include="all").T
    stats.rename(
        columns={"25%": "25% (Q1)", "50%": "50% (median)", "75%": "75% (Q3)"},
        inplace=True
    )
    stats.fillna("", inplace=True)
    stats.index.name = "variable"
    if transpose:
        stats = stats.T
    title = f"<b>Statistics</b><br><sub>Dimensions: {df.shape}</sub>"
    tab = table(stats.reset_index(), title, format=None, colorscale=colorscale, template=template, **kwargs)
    return tab


def prediction_table(model, y, y_pred, colorscale=None, template=None, **kwargs):
    """Real vs Prediction table.

    Args:
        model: scikit-learn model
        y (series or array): the target column for classification
        y_pred (series or array): predicted values
        normalize (str)): one of None, 'true' (or synonym 'real'), 'pred', 'all'
        extended (bool): return a dataframe, besides the Plotly table figure
        colorscale (str or list): plotly named colorscale (`blues`, `burgyl_r`...) or 2d list mapping
        template (str): template name to use - uses heatmap `colorscale`, not table colors to provide alternating colors
        **kwargs: any Plotly kwargs for go.Heatmap

    Returns:
        Figure (table)
    """
    if type(y) == pd.DataFrame:
        df = y
    elif type(y) == pd.Series:
        df = y.to_frame()
    else:
        df = pd.DataFrame(y)
    name = df.columns.values[0]
    df.reset_index(inplace=True)
    df2 = pd.DataFrame(y_pred)
    df = pd.concat([df, df2], axis=1, ignore_index=True)
    df.drop(0, axis=1, inplace=True)  # removing "index"
    df.columns = [name + " (real)", name + " (pred)"]
    df["match"] = df[name + " (real)"] == df[name + " (pred)"]
    df.reset_index(inplace=True)
    df.loc[df["match"] == True, "match"] = "✔️"  # NOQA
    df.loc[df["match"] == False, "match"] = "❌"  # NOQA
    tab = table(df, title=f"<b>{model} - Real vs Predicted table</b>",
                colorscale=colorscale, template=template, named_index=False, **kwargs)
    return tab


def table(df, title=None, format=None, colorscale=None, template=None, named_index=True, **kwargs):
    """Build generic plotly table.

    Args:
        df (dataframe): tabular data
        title (str): title for the table. Can use <b></b>, <sub></sub>, <br> etc. Font size is defined by the template.
        colorscale (str or list): plotly named colorscale (`blues`, `burgyl_r`...) or 2d list mapping
        template (str): template name to use - uses heatmap `colorscale`, not table colors to provide alternating colors
        **kwargs: any Plotly kwargs for `go.Figure`

    Returns:
        Figure (table)
    """
    if type(df) == pd.Series:
        df = df.to_frame()
    elif type(df) != pd.DataFrame:
        df = pd.DataFrame(df)

    if len(df.columns) == 1:
        named_index = False
    colorscale_list = heatmap_colorscale(colorscale, template)

    # plot
    header_color = colorscale_list[3][1]
    index_color = colorscale_list[2][1]
    row_even_color = colorscale_list[1][1]
    row_odd_color = colorscale_list[0][1]
    row_text_color = complementary_hex_color(row_even_color)

    if named_index:
        row_fill_color = [index_color, [row_odd_color, row_even_color, ] * int((df.shape[0] / 2) + 1)]
        font_size = [14] + [13] * (len(df.columns) - 1)
    else:
        row_fill_color = [[row_odd_color, row_even_color, ] * int((df.shape[0] / 2) + 1)]
        font_size = [13] * len(df.columns)
    data = [
        go.Table(
            header=dict(
                values=[f"<b>{col}</b>" for col in df.columns.values],
                line_color="black",
                fill_color=[header_color],
                align=["left", "center"],
                font=dict(color="white", size=14),
            ),
            cells=dict(
                values=[df[col] for col in df.columns.values],
                # line_color='darkslategray',
                fill_color=row_fill_color,
                align=["left", "center"],
                font=dict(color=row_text_color, size=font_size),
                format=format,

            ),
        )
    ]
    layout = {
        "title": title,
    }

    fig = go.Figure(data=data, layout=layout, **kwargs)
    return fig


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    random_state = 42
    np.random.seed(random_state)
    pio.templates.default = "dion"

    # binary classification
    X, y = make_classification(n_samples=500, random_state=random_state)

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.5, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # CT
    ct = classification_table(model, y_test, y_pred)
    ct.show()

    # CM
    fig = confusion_matrix_table(model, y_test, y_pred)
    fig.show()

    fig = confusion_matrix_table(
        model, y_test, y_pred, normalize="true", template="plotly_dark"
    )
    fig.show()

    fig = confusion_matrix_table(
        model, y_test, y_pred, normalize="pred", colorscale="burgyl_r"
    )
    fig.show()

    fig = confusion_matrix_table(
        model, y_test, y_pred, normalize="all", colorscale="blues"
    )
    fig.show()

    df = px.data.iris()

    X = df.drop(columns=["species", "species_id"])
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=random_state
    )

    # dataframe stats
    tab = describe(df)
    tab.show()

    # Fit the model
    lr_model = LogisticRegression(max_iter=200)
    lr_model.fit(X_train, y_train)
    y_probs = lr_model.predict_proba(X_test)
    y_pred = lr_model.predict(X_test)

    # class imbalance
    tab = class_imbalance_table(y_train)
    tab.show()

    table(y_test, title="y_test").show()
    table(y_pred, title="y_pred").show()
    # predictions
    tab = prediction_table(y_test, y_pred)
    tab.show()

    # CT
    tab, cl_df = classification_table(lr_model, y_test, y_pred, extended=True)
    tab.show()
    print(cl_df)

    # CM
    tab, cm_df = confusion_matrix_table(
        lr_model, y_test, y_pred, extended=True, template="plotly_white"
    )
    tab.show()
    print(cm_df)

    #
    from sklearn.datasets import load_digits

    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.5, random_state=42
    )

    tab = class_imbalance_table(y_train, y_test, template="plotly_dark")
    tab.show()
