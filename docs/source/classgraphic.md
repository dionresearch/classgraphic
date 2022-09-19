# classgraphic package
Interactive classification diagnostic plots for scikit-learn.

![coin sorting machine](sorter_patent.jpg)

> We classify things for the purpose of doing something to them. Any classification which does not assist manipulation is worse than useless. - Randolph S. Bourne,
  "Education and Living", The Century Co (April 1917)

## Module contents

```{eval-rst}
.. automodule:: classgraphic
   :members:
   :inherited-members:
   :show-inheritance:
```
## Submodules

### classgraphic.common

```{eval-rst}
.. automodule:: classgraphic.common
   :members:
   :inherited-members:
   :show-inheritance:
```

### classgraphic.essential

By starting a python script or a notebook with `from classgraphic.essential import *`,
all the basic elements are loaded to start exploring data, build linear classification
models and evaluate them.

Specifically, essential exposes the following tables:

- class_imbalance_table 
- classification_table
- confusion_matrix_table
- describe
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

For templates:

- set_default_template

`classgraphic.essential` also sets the default template to "dion", which unifies
the look of all visualizations and tables around a gradient of blue, with light background and
lines.

Finally, it also imports `numpy as np`, `pandas as pd`, `plotly.express as px`, `plotly.graph_objects as go`
and `make_classification`, `train_test_split` and `LogisticRegression` from `sklearn`.

### classgraphic.table

```{eval-rst}
.. automodule:: classgraphic.table
   :members:
   :inherited-members:
   :show-inheritance:
```

### classgraphic.template

```{eval-rst}
.. automodule:: classgraphic.template
   :members:
   :inherited-members:
   :show-inheritance:
```

### classgraphic.viz

```{eval-rst}
.. automodule:: classgraphic.viz
   :members:
   :inherited-members:
   :show-inheritance:
```
