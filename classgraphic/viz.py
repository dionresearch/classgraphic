"""Classification visualizations, in plotly format and some also in pandas format."""
import warnings
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    det_curve,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)
import scipy as sp
from classgraphic.common import get_labels, missing_matrix, pio, value_count
from classgraphic.table import confusion_matrix_table
from classgraphic.template import set_default_template


def class_imbalance(
    y,
    y2=None,
    condition=None,
    sort=False,
    ascending=False,
    reference=True,
    always_bar=False,
    barmode="stack",
    **kwargs,
):
    """Class imbalance visualization.

    Given a `target` Series or array (eg. y, y_test, y_train), this visualize the balance/imbalance of
    classes. For binary classification, the default is a pie chart. For multiclass, it is a bar chart.

    Args:
        y (series or array): usually, the target column for classification
        condition (str): added to title. Something like 'train', 'test'
        sort (bool): if True, data will be sorted by class count
        ascending (bool): if True, sort ascending, else descending
        reference (bool): add a reference line to better compare classes
        always_bar (bool): it True, for binary classification use bar instead of pie chart
        barmode
        **kwargs: any Plotly kwargs for `px.pie` or `px.bar`

    Returns:
        Figure
    """
    if condition is None:
        condition = ""
        for_text = ""
    else:
        for_text = " for "
    title = f"<b>Data set classes counts{for_text}{condition}</b>"

    # data wrangling
    values = value_count(y)  # dict of classes, count as list, for Pandas
    df = pd.DataFrame(values).T.reset_index()
    df.columns = ["class", "count"]

    if y2 is not None:
        values2 = value_count(y2)
        df2 = pd.DataFrame(values2).T.reset_index()
        df2.columns = ["class", "count"]
        text = "y"
        text2 = "y2"
        if "," in condition:
            text, text2 = condition.split(",")

        df["set"] = text
        df2["set"] = text2
        df = pd.concat([df, df2], axis='index')
        pattern_shape_map = {}
        pattern_shape_map[text] = ""
        pattern_shape_map[text2] = "/"

    else:
        df["set"] = ""
        pattern_shape_map = {}
    # plot
    if not always_bar and len(values.keys()) == 2:
        # binary, use pie
        fig = px.pie(df, values="count", names="class", facet_col="set", title=title, **kwargs)
        if reference:
            fig.add_shape(
                type="line",
                line=dict(width=2, dash="dash", color="black"),
                x0=0.5,
                x1=0.5,
                y0=-0.1,
                y1=1.1,
            )
    else:

        # bar chart it is
        if sort:
            df = df.sort_values(by="count", ascending=ascending)
        if y2 is not None:
            # have to make two different bar() calls, else legend is wrong
            fig = px.bar(df, x="count", y="class", color="class", barmode=barmode, pattern_shape="set",
                         pattern_shape_map=pattern_shape_map, **kwargs)
        else:
            fig = px.bar(df, x="count", y="class", color="class", barmode=barmode, **kwargs)
        title += f"<br><sub>{barmode=}</sub>"
        if reference:
            if y2 is not None:
                fig.add_vline(
                    x=df["count"][df["set"] == text].min(),
                    line_width=1, line_dash="solid", line_color="black",
                    annotation_text=f"{text} min()"
                )
                fig.add_vline(
                    x=df["count"][df["set"] == text2].min(),
                    line_width=1, line_dash="dash", line_color="black",
                    annotation_text=f"{text2} min()"
                )
            else:
                fig.add_vline(
                    x=df["count"].min(), line_width=1, line_dash="solid", line_color="black",
                    annotation_text=f"{condition} min()"
                )
    fig.update_layout(
        dict(
            title=title,
        )
    )
    return fig


def class_error(model, y, y_pred, normalize=False, reference=True, **kwargs):
    """Classification error chart.

    Given a `model`, a `target` Series or array and a prediction Series or array, this visualize the
    prediction errors (for each real class, a breakdown of how many predicted by class). If normalize is False,
    it also show balance/imbalance of classes. If normalize is True, the bars all represent 100% of the class,
    and each segment, the percent of that predicted class for the real class.

    Args:
        model (sklearn model): A scikit learn model (displayed in title)
        y (series or array): the target column for classification
        y_pred (series or array): predicted values
        normalize (bool):  if True, each segment is in percent and each true class is equal to 100%
        reference (bool): add a reference line to better compare classes
        **kwargs: any Plotly kwargs for `px.bar`

    Returns:
        Figure
    """
    # labels
    labels = get_labels(model, y, as_str=False).columns
    x_labels = [str(label) + " (pred)" for label in labels]

    # data wrangling
    cm_array = confusion_matrix(y, y_pred, labels=labels)
    df = pd.DataFrame(cm_array, columns=labels).T.reset_index()
    df.columns = ["Real"] + x_labels
    df["Real"] = df["Real"].astype(str)  # in case classes are 0, 1 etc.
    df["total"] = df[x_labels].sum(axis=1)

    # Better labeling
    xaxis_title = "Count of predicted classes"
    normalized = ""
    if normalize:
        df[x_labels] = df[x_labels].divide(df["total"], axis=0) * 100
        df["total"] = 100
        xaxis_title = "Percentage of predicted classes"
        normalized = " (normalized)"

    # plot
    fig = px.bar(
        df,
        y="Real",
        x=x_labels,
        text_auto=True,
        text="total",
        orientation="h",
        **kwargs,
    )
    if reference and not normalize:
        # no point adding the line when normalized since they all align at 100%
        fig.add_vline(
            x=df["total"].min(), line_width=1, line_dash="dash", line_color="black"
        )
    fig.update_layout(
        dict(
            title=f"<b>{model} - Classification Error chart{normalized}</b>",
            xaxis=dict(title=xaxis_title),
        )
    )
    return fig


def det(model, y, y_pred, **kwargs):
    """Plot detection error tradeoff (DET) curve.

    A visualization of error rates for binary classification, plotting the probit of false negative
     rate vs. false positive rate. The interval -3, 3 from the probit is mapped to a percentage.

    Args:
        model (sklearn model): A scikit learn model (displayed in title)
        y (series or array): the target column for classification
        y_pred (series or array): predicted values
        **kwargs: any Plotly kwargs for `px.line`

    Returns:
        Figure
    """
    fpr, fnr, _ = det_curve(y, y_pred)
    df = pd.DataFrame(
        {
            "False Positive Rate": sp.stats.norm.ppf(fpr),
            "False Negative Rate": sp.stats.norm.ppf(fnr),
        }
    )

    # plot
    fig = px.line(
        df,
        x="False Positive Rate",
        y="False Negative Rate",
        range_y=[-3, 3],
        range_x=[-3, 3],
        **kwargs,
    )
    ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
    tickvals = sp.stats.norm.ppf(ticks)
    ticktext = [f"{s:.0%}" if (s * 100).is_integer() else f"{s:.1%}" for s in ticks]
    fig.update_xaxes(
        tickmode="array", tickvals=tickvals, ticktext=ticktext, constrain="domain"
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(title=f"<b>{model} - DET Curve</b>", legend_orientation="v")
    return fig


def feature_importance(model, y, transpose=False, extended=False, **kwargs):
    """Feature Importance.

    Args:
        model (sklearn model): A scikit learn model (displayed in title)
        y (series or array): the target column for classification
        transpose (bool): columns become rows, rows become columns
        extended (bool): return a dataframe, besides the Plotly figure
        **kwargs: any Plotly kwargs for `px.histogram`

    Returns:
        Figure
    """
    try:
        features = model.feature_names_in_
    except AttributeError:
        features = None
    labels = get_labels(model, y)
    # depends on the scikit-learn model...
    try:
        importance = model.feature_importances_
    except AttributeError:
        importance = model.coef_
    df = pd.DataFrame(importance)
    if features is not None:
        df.columns = features
    try:
        df.index = labels
    except ValueError:
        # binary class, single value
        df.index = ["Positive"]

    if transpose:
        df = df.T
        legend_title = "class"
        xaxis_title = "coefficient"
        yaxis_title = "feature"
    else:
        legend_title = "feature"
        xaxis_title = "coefficient"
        yaxis_title = "class"

    title = "<b>Feature Importance</b>"
    fig = px.bar(df, orientation="h", title=title, **kwargs)
    fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title, legend_title=legend_title)
    if extended:
        return fig, df
    return fig


def missing(X, warning=True, **kwargs):
    """Missing data heatmap.

    Missing is defined here as NaN on numerical data types.

    Args:
        X (sklearn model): A scikit learn model (displayed in title)
        warning (bool): Issue a warning if there are some missing values.
        **kwargs: any Plotly kwargs for `px.imshow`

    Returns:
        Figure
    """
    # data wrangling
    data = missing_matrix(X)

    if type(data) == pd.DataFrame:
        missing_count = data.sum().sum()
    else:
        missing_count = sum(data.flatten())

    missing_text = f"missing: {missing_count}"

    # plot
    if missing_count:
        fig = px.imshow(
            data,
            contrast_rescaling="infer",
            aspect="auto",
            labels=dict(color="Missing Value (255 = True)"),
            **kwargs,
        )
        if warning:
            warnings.warn("The dataset has missing (None or NaN) values.")
    else:
        fig = px.imshow(
            data,
            zmin=0, zmax=1,
            aspect="auto",
            labels=dict(color="Missing Value"),
            **kwargs,
        )
    fig.update_coloraxes(showscale=False)

    fig.update_layout(
        title=f"<b>Missing Data (NaN, None)</b><br><sub>{missing_text}</sub>", xaxis_title="Feature", yaxis_title="Row"
    )
    return fig


def precision_recall(model, y, y_probs, **kwargs):
    """Precision / Recall curve.

    Args:
        model (sklearn model): A scikit learn model (displayed in title)
        y (series or array): the target column for classification
        y_probs (series or array): predicted value probabilities
        **kwargs: any Plotly kwargs for `px.line`

    Returns:
        Figure
    """
    onehot = get_labels(model, y, as_str=False)

    fig = go.Figure()

    try:
        for i in range(y_probs.shape[1]):
            y_true = onehot.iloc[:, i]
            y_prob = y_probs[:, i]

            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

            pr_auc = auc(recall, precision)
            p_score = average_precision_score(y_true, y_prob)

            name = (
                f"{onehot.columns[i]} (AUC={pr_auc:.3f}, AP={p_score:.3f})"
            )
            fig.add_trace(go.Scatter(x=recall, y=precision, name=name, mode="lines"))
        name = ""
    except IndexError:
        # Single / binary class
        precision, recall, thresholds = precision_recall_curve(y, y_probs)

        pr_auc = auc(recall, precision)
        p_score = average_precision_score(y, y_probs)
        name = f"AUC={pr_auc:.3f}, AP={p_score:.3f}"
        fig = px.line(
            x=recall,
            y=precision,
            labels=dict(x="Recall", y="Precision"),
        )

    fig.update_layout(
        title=f"<b>{model} - Precision Recall Curve</b><br><sub>{name}</sub>",
        xaxis_title="Recall",
        yaxis_title="Precision",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain="domain"),
        legend_orientation="v",
        **kwargs,
    )
    fig.add_shape(
        type="line",
        line=dict(dash="dash", color="red"),
        x0=0,
        x1=1,
        y0=1,
        y1=0,
        name="random",
    )
    return fig


def prediction_histogram(
    model,
    y,
    y_probs,
    nbins=50,
    barmode="overlay",
    reference=True,
    extended=False,
    **kwargs,
):
    """Prediction Histogram chart.

    Args:
        model (sklearn model): A scikit learn model (displayed in title)
        y (series or array): the target column for classification
        y_probs (series or array): predicted value probabilities
        nbins (int): number of bins, default to 50
        barmode (str): one of ['stack', 'group', 'overlay', 'relative', None], defaults to overlay
        reference (bool): add a reference line to better compare classes
        extended (bool): return a dataframe, besides the Plotly figure
        **kwargs: any Plotly kwargs for `px.histogram`

    Returns:
        Figure
    """
    # data wrangling
    labels = get_labels(model, y)
    df1 = pd.DataFrame({"Real": y})
    df1["Real"] = df1["Real"].astype(str)
    df2 = pd.DataFrame(y_probs)
    if y_probs.shape == y.shape:
        color = "Real"
        y = None
        x = "Score"
        df = pd.concat([df1, df2], axis=1, ignore_index=True)
        df.columns = ["Real", "Score"]
        pattern = None
        legend_title = "Real"
    else:
        pattern = "Real"
        color = None
        x = labels
        y = None
        df = pd.concat([df1, df2], axis=1, ignore_index=True)
        df.columns = ["Real"] + labels
        legend_title = "Predicted, Real"
        df.sort_values(by=["Real"], inplace=True)
        df["sum"] = df[labels].fillna(0).sum(axis=1)
    # Plot
    fig = px.histogram(
        df,
        x=x,
        y=y,
        color=color,
        range_x=[0, 1],
        nbins=nbins,
        title=f"<b>{model} - Prediction Histogram</b><br><sub>{barmode=:}</sub>",
        barmode=barmode,
        pattern_shape=pattern,
        **kwargs,
    )
    fig.update_layout(xaxis_title="Score", legend_title=legend_title)
    if reference:
        fig.add_vline(0.5, line_width=1, line_dash="dash", line_color="black")
    if extended:
        return fig, df
    return fig


def roc(model, y, y_probs, ovo=False, ovr=True, **kwargs):
    """Plot Receiver Operating Characteristic (ROC) curve.

    Args:
        model (sklearn model): A scikit learn model (displayed in title)
        y (series or array): the target column for classification
        y_probs (series or array): predicted value probabilities
        ovo (bool): if True, display One vs One macro average and weighted AUC (multiclass)
        ovr (bool): if True, display One vs Rest macro average and weighted AUC (multiclass)
        **kwargs: any Plotly kwargs for `go.Figure`

    Returns:
        Figure
    """
    labels = get_labels(model, y, as_str=False)

    # Create an empty figure, and iteratively add new lines
    # every time we compute a new class
    fig = go.Figure(**kwargs)
    auc_str = ""
    try:
        for i in range(y_probs.shape[1]):
            y_true = labels.iloc[:, i]
            y_score = y_probs[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)
            name = f"{labels.columns[i]} (AUC={auc_score:.3f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))
        ovo_text = ""
        if ovo:
            macro_roc_auc_ovo = roc_auc_score(y, y_probs, multi_class="ovo", average="macro")
            weighted_roc_auc_ovo = roc_auc_score(y, y_probs, multi_class="ovo", average="weighted")
            ovo_text = f"macro-average OVO AUC={macro_roc_auc_ovo:.3f}," \
                       f" weighted-average OVO AUC={weighted_roc_auc_ovo:.3f}"
        ovr_text = ""
        if ovr:
            macro_roc_auc_ovr = roc_auc_score(y, y_probs, multi_class="ovr", average="macro")
            weighted_roc_auc_ovr = roc_auc_score(y, y_probs, multi_class="ovr", average="weighted")
            if ovo:
                ovr_text = ", "
            ovr_text += f"macro-average OVO AUC={macro_roc_auc_ovr:.3f}," \
                        f" weighted-average OVO AUC={weighted_roc_auc_ovr:.3f}"
        title = f"<b>{model} ROC Curve</b><br><sub>{ovo_text}{ovr_text}</sub></b>"
    except IndexError:
        # Single / binary class
        fpr, tpr, thresholds = roc_curve(y, y_probs)
        auc_score = roc_auc_score(y, y_probs)
        auc_str = f"AUC = {auc_score:.3f}"
        df = pd.DataFrame(
            {"False Positive Rate": fpr, "True Positive Rate": tpr}
        )
        title = f"<b>{model} ROC Curve</b><br><sub>{auc_str}</sub></b>"
        fig = px.line(df, x="False Positive Rate", y="True Positive Rate")

    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain="domain"),
        legend_orientation="v",
        **kwargs,
    )
    # random threshold line
    fig.add_shape(
        type="line",
        line=dict(dash="dash", color="red"),
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        name="random",
    )
    return fig


def threshold(model, y, y_probs, **kwargs):
    """Plot True Positive Rate vs False Positive Rate for all Threshold.

    Args:
        model (sklearn model): A scikit learn model (displayed in title)
        y (series or array): the target column for classification
        y_probs (series or array): predicted value probabilities

        **kwargs: any Plotly kwargs for `go.Figure`

    Returns:
        Figure
    """
    labels = get_labels(model, y, as_str=False)
    fig = go.Figure()
    try:
        for i in range(y_probs.shape[1]):
            y_true = labels.iloc[:, i]
            y_score = y_probs[:, i]

            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            # Evaluating model performance at various thresholds
            df = pd.DataFrame(
                {"False Positive Rate": fpr, "True Positive Rate": tpr},
                index=thresholds,
            )
            df.index.name = "Thresholds"
            df.columns.name = "Rate"
            df.sort_index(inplace=True)
            name = f"{labels.columns[i]}"
            fig.add_trace(
                go.Scatter(
                    x=df["False Positive Rate"],
                    y=df["True Positive Rate"],
                    name=name,
                    mode="lines",
                )
            )

    except IndexError:
        # Single / binary class
        fpr, tpr, thresholds = roc_curve(y, y_probs)

        # Evaluating model performance at various thresholds
        df = pd.DataFrame(
            {"False Positive Rate": fpr, "True Positive Rate": tpr}, index=thresholds
        )
        df.index.name = "Thresholds"
        df.columns.name = "Rate"
        df.sort_index(inplace=True)

        fig = px.line(df)

    fig.update_layout(
        title=f"<b>{model} TPR and FPR at every threshold</b>",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain="domain", range=[0, 1]),
        legend_orientation="v",
        **kwargs,
    )
    return fig


def view(X, y_real, y_pred=None, nobs=10, extended=False, **kwargs):
    """Receiver Operating Characteristic.

    Args:
        X (DataFrame or serues or array): feature data to view
        y_real (series or array): the target column for classification matching X
        y_pred (series or array): optional, predicted value probabilities matching X
        nobs (int): number of observations to display
        extended (bool): if True, return dataframe
        **kwargs: any Plotly kwargs for `px.scatter`

    Returns:
        Figure
    """
    if type(X) == pd.DataFrame:
        df = X
    elif type(X) == pd.Series:
        df = X.to_frame()
    else:
        df = pd.DataFrame(X)
    feature_count = len(df.columns)
    if type(y_real) == pd.DataFrame:
        df2 = y_real
    elif type(y_real) == pd.Series:
        df2 = y_real.to_frame()
    else:
        df2 = pd.DataFrame(y_real)
    df2.columns = ["Real"]
    legend_title = "Real"

    df[list(df2.columns)] = df2.copy()

    if y_pred is not None:
        if type(y_pred) == pd.DataFrame:
            df2 = y_pred
        elif type(y_pred) == pd.Series:
            df2 = y_pred.to_frame()
        else:
            df2 = pd.DataFrame(y_pred)
        df2.columns = ["Predicted"]
        name = list(df.columns)
        name.append("Predicted")
        df.reset_index(inplace=True)
        df = pd.concat([df, df2], axis=1, ignore_index=True)
        df.drop(0, axis=1, inplace=True)  # removing "index"
        df.columns = name
        legend_title = "Predicted, Real"

    df = df.head(nobs)
    df.sort_values(by=["Real"], inplace=True)
    symbol = None
    scatter_args = {}
    params = ["x", "y", "z", "size"]
    for i in range(feature_count):
        try:
            param = params.pop()
        except IndexError:
            break
        scatter_args[param] = df.columns[i]
    if y_pred is not None:
        symbol = df.columns[-1]
        scatter_args["color"] = df.columns[-2]
    else:
        scatter_args["color"] = df.columns[-1]
    scatter_args["symbol"] = symbol

    if feature_count == 2:
        fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color=df.columns[2], symbol=symbol, **kwargs)
    elif feature_count > 2:
        fig = px.scatter_3d(df, **scatter_args, **kwargs)
    fig.update_layout(
        title="Scatter view of X, for y and predicted",
        legend_title=legend_title
    )
    if extended:
        return fig, df
    return fig


def voronoi_diagram(centroid_source, data=None, labels=None, plot_voronoi=True, plot_delauney=False, **kwargs):
    """Plot a 2D voronoi diagram.

    Args:
        centroid_source: either a fitted clustering alrorithm with cluster_centers, or cluster center coordinates
        data: optional, original data (X) used for calculating the clusters
        labels: labels (y) for the clusters
        plot_voronoi: if False, will not plot boundaries, aka Voronoi tessalation
        plot_delauney: if True, plots Delauney triangulation
        **kwargs: for now, passed to initial scatter plot

    Returns:
        Plotly figure
    """
    try:
        centers = centroid_source.cluster_centers
    except AttributeError:
        centers = centroid_source
    voronoi = sp.spatial.Voronoi(centers)
    if plot_delauney:
        triangles = sp.spatial.Delaunay(centers)

    voronoi_points = pd.DataFrame(voronoi.points)
    voronoi_points["type"] = "centroids"
    voronoi_vertices = pd.DataFrame(voronoi.vertices)
    voronoi_vertices["type"] = "vertices"
    if data is not None:
        data_points = pd.DataFrame(X)
        data_points["type"] = "data"
        voronoi_df = pd.concat([data_points, voronoi_vertices, voronoi_points], axis=0)
    else:
        voronoi_df = pd.concat([voronoi_vertices, voronoi_points], axis=0)

    center = voronoi.points.mean(axis=0)
    ptp_bound = voronoi.points.ptp(axis=0)

    # Voronoi
    finite_segments = []
    infinite_segments = []
    # This loop is borrowed from scipy.spatial.voronoi_plot_2d, which is matplotlib
    # oriented. This is further massaged to use plotly.
    for pointidx, simplex in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(voronoi.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = voronoi.points[pointidx[1]] - voronoi.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = voronoi.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (voronoi.furthest_site):
                direction = -direction
            far_point = voronoi.vertices[i] + direction * ptp_bound.max()
            infinite_segments.append([voronoi.vertices[i], far_point])

    if data is not None and labels is not None:
        fig = px.scatter(data_points, x=0, y=1, color="type", hover_name=labels, **kwargs)
        fig.add_trace(go.Scatter(x=voronoi_vertices[0], y=voronoi_vertices[1], mode="markers", name="vertices"))
        fig.add_trace(go.Scatter(x=voronoi_points[0], y=voronoi_points[1], mode="markers", marker_symbol="x",
                                 name="centroids"))
    else:
        fig = px.scatter(voronoi_df, x=0, y=1, color="type")

    for plot in fig.data:
        if plot["name"] == "vertices":
            vertices_color = plot.marker.color
        elif plot["name"] == "centroids":
            centroids_color = plot.marker.color

    if plot_delauney:
        first = True
        for triangleidx in triangles.simplices:
            x, y = zip(triangles.points[triangleidx].T)
            fig.add_trace(
                go.Scatter(x=x[0], y=y[0], mode="lines+markers", line_color="red", marker_color=centroids_color,
                           name="delauney", legendgroup="delauney", showlegend=first))
            first = False

    if plot_voronoi:
        first = True
        for line in finite_segments:
            x, y = zip(*line)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line_color="black", marker_color=vertices_color,
                                     name="boundary", legendgroup="boundary", showlegend=first))
            first = False
        for line in infinite_segments:
            x, y = zip(*line)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line_color="black", marker_color=vertices_color,
                                     name="boundary", legendgroup="boundary", showlegend=first))
    return fig


if __name__ == "__main__":
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification, make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    set_default_template("dion")

    random_state = 0
    np.random.seed(random_state)

    binary = False
    if binary:
        # Data
        X, y = make_classification(n_samples=500, random_state=random_state)

        # Class split, pie chart
        fig = class_imbalance(y, reference=True)
        fig.show()

        # class split, bar chart
        fig = class_imbalance(y, reference=True, always_bar=True)
        fig.show()

        # binary classification
        model = LogisticRegression()
        model.fit(X, y)

        # predictions
        y_score = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)

        # Figures

        # prediction histogram
        fig = prediction_histogram(model, y, y_score)
        fig.show()

        # class error
        fig = class_error(model, y, y_pred, reference=True)
        fig.show()

        # normalized class error
        fig = class_error(model, y, y_pred, normalize=True)
        fig.show()

        # precision recall curve
        fig = precision_recall(model, y, y_score)
        fig.show()

        # roc curve
        fig = roc(model, y, y_score)
        fig.show()

        # threshold
        fig = threshold(model, y, y_score)
        fig.show(
            config={
                "displaylogo": False,
                "modeBarButtonsToAdd": [
                    "hoverClosestCartesian",
                    "hoverCompareCartesian",
                ],
            }
        )

        # for DET
        X, y = make_classification(n_samples=1000, random_state=random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=random_state
        )
        svc_model = SVC(random_state=0).fit(X_train, y_train)
        y_pred = svc_model.decision_function(X_test)

        # DET
        fig = det(svc_model, y_test, y_pred, template="plotly")
        fig.show()

        # add 50 missing values
        index = np.random.choice(X.size, 50, replace=False)
        X.ravel()[index] = np.nan

        # Missing values
        fig = missing(X)
        fig.show()

    multivar = False
    if multivar:
        df = px.data.iris()
        X = df.drop(columns=["species", "species_id"])
        y = df["species"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.6, random_state=random_state
        )

        # Class split
        fig = class_imbalance(y_train, y_test, condition="train,test", sort=True)
        fig.show()

        fig = class_imbalance(y_train, condition="train", sort=True)
        fig.show()

        # Fit the model
        lr_model = LogisticRegression(max_iter=200)
        lr_model.fit(X_train, y_train)
        y_probs = lr_model.predict_proba(X_test)
        y_pred = lr_model.predict(X_test)

        # class error
        fig = class_error(lr_model, y_test, y_pred)
        fig.show()

        # confusion matrix
        tab = confusion_matrix_table(lr_model, y_test, y_pred)
        tab.show()

        # feature importance
        fig = feature_importance(lr_model, y)
        fig.show()

        fig = feature_importance(lr_model, y, transpose=True)
        fig.show()

        # view
        fig = view(X_test, y_test, y_pred)
        fig.show()

        # prediction histogram
        fig, ph_df = prediction_histogram(
            lr_model, y_test, y_probs, extended=True, template="seaborn"
        )
        fig.show()
        ph_df.to_csv("results.csv")

        print(y_probs.shape)
        # precision recall curve
        fig = precision_recall(lr_model, y_test, y_probs)
        fig.show()

        # multiclass roc
        fig = roc(lr_model, y_test, y_probs)
        fig.show()

        # threshold
        fig = threshold(lr_model, y_test, y_probs)
        fig.show()

        # add missing values
        X = X.mask(np.random.random(X.shape) < 0.05)

        # Missing values
        fig = missing(X, template="plotly_white")
        fig.show()

    # Clustering
    X, y, centers = make_blobs(n_samples=300, n_features=2, centers=9, return_centers=True)
    fig = voronoi_diagram(centers, X, y)
    fig.show()

    _, _, centers = make_blobs(n_samples=144, n_features=2, centers=12, return_centers=True)
    fig = voronoi_diagram(centers, plot_voronoi=False, plot_delauney=True)
    fig.show()
