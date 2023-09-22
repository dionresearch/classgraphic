# classgraphic
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![image](https://img.shields.io/pypi/v/classgraphic.svg)](https://pypi.python.org/pypi/classgraphic) 
![Dev](https://github.com/dionresearch/classgraphic/actions/workflows/dev.yml/badge.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dionresearch/classgraphic/HEAD?labpath=notebooks%2FClassGraphic_iris_demo.ipynb)

Interactive classification diagnostic plots for scikit-learn.

![coin sorting machine](https://github.com/dionresearch/classgraphic/raw/main/docs/source/sorter_patent.jpg)

> We classify things for the purpose of doing something to them. Any classification which does not assist manipulation is worse than useless. - Randolph S. Bourne,
  "Education and Living", The Century Co (April 1917)

# Major features:

Plotly based tables for:

- class_imbalance_table 
- classification_table
- confusion_matrix_table
- describe (dataframe stats)
- prediction_table
- table

And the following charts:

- class_imbalance 
- class_error
- det
- feature_importance
- missing
- precision_recall
- roc
- prediction_histogram
- threshold

For clustering:
- Delauney triangulations
- Voronoi tessalations

# Try it

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dionresearch/classgraphic/HEAD?labpath=notebooks%2FClassGraphic_iris_demo.ipynb)

By trying it on binder, you'll see all the details and interactivity. The quickstart below
has static images, but if you run these commands in a jupyter notebook, ipython or IDE you will
be able to interact with them.

# Quickstart

```python
from classgraphic.essential import *

# loading the data
df = px.data.iris()

# let's see what kind of data we have
describe(df, transpose=True).show()
```
![dataframe describe tale](https://github.com/dionresearch/classgraphic/raw/main/docs/source/describe.png)
```python
# any missing?
missing(df)
```
![dataframe describe tale](https://github.com/dionresearch/classgraphic/raw/main/docs/source/missing.png)
```python
# features
X = df.drop(columns=["species", "species_id"])

#target
y = df["species"]

# Let's check our classes we will be training on and predicting
class_imbalance_table(y, condition="all")
```
![dataframe describe tale](https://github.com/dionresearch/classgraphic/raw/main/docs/source/imbalance_table.png)
```python
# train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=random_state
)

# we want to see total count for each, default for bars is to be stacked, so that works
# we could also pass to class_imbalance barmode="overlay" if we prefer
class_imbalance(y_train, y_test, condition="train,test")
```
![dataframe describe tale](https://github.com/dionresearch/classgraphic/raw/main/docs/source/class_imbalance.png)
```python
# model
model = LogisticRegression(max_iter=max_iter, random_state=random_state)
model.fit(X_train, y_train)

# predictions
y_score = model.predict_proba(X_test)
y_pred = model.predict(X_test)

confusion_matrix_table(model, y_test, y_pred).show()
classification_table(model, y_test, y_pred)
```
![dataframe describe tale](https://github.com/dionresearch/classgraphic/raw/main/docs/source/confusion.png)
![dataframe describe tale](https://github.com/dionresearch/classgraphic/raw/main/docs/source/classification_table.png)
```python
feature_importance(model, y, transpose=True)
```
![dataframe describe tale](https://github.com/dionresearch/classgraphic/raw/main/docs/source/feature.png)

This concludes the quickstart. There are many more visualizations and tables to explore.

See the `notebooks` and `docs` folders on [github](https://github.com/dionresearch/classgraphic) and the documentation
[web site](https://dionresearch.github.io/classgraphic/) for more information.

# Requirements

- Python 3.8 or later
- numpy
- pandas
- plotly>=5.0
- scikit-learn
- nbformat

# Install

If you use conda, create an environment named `classgraphic`, then activate it:

- in Linux:
`source activate pilot`

- In Windows:
`conda activate pilot`

If you use another environment management create and activate your environment
using the normal steps.

Then execute:

```sh
python setup.py install
```

or for installing in [development mode](https://pip.pypa.io/en/latest/cli/pip_install/#install-editable):


```sh
python -m pip install -e . --no-build-isolation
```

or alternatively

```sh
python setup.py develop
```

To install from github instead:
```shell
pip install git+https://github.com/dionresearch/classgraphic
```


# See also

- [stemgraphic](https://github.com/dionresearch/stemgraphic) python package for visualization of data and text
- [Hotelling](https://github.com/dionresearch/hotelling) one and two sample Hotelling T2 tests, T2 and f statistics and univariate and multivariate control charts and anomaly detection
