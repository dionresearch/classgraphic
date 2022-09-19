"""Essential components for quick visualization."""
from classgraphic.table import (
    class_imbalance_table,
    classification_table,
    confusion_matrix_table,
    describe,
    prediction_table,
    table
)

from classgraphic.template import set_default_template

from classgraphic.viz import (
    class_imbalance,
    class_error,
    det,
    feature_importance,
    missing,
    precision_recall,
    roc,
    prediction_histogram,
    threshold,
    view
)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

set_default_template()
