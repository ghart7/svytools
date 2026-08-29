# Svytools

The **svytools** package is a general-purpose toolkit of plotting, data-wrangling, and clustering utilities. It was factored out of the [Survey](https://github.com/survey-genomics/survey) toolkit so it can be installed and imported on its own, without requiring Scanpy, MuData, AnnData, or any other single-cell-specific dependencies.

It currently provides:

- `genutils` – general utility functions and classes (config merging, `ParamManager`, DataFrame helpers, pickling, normalization, etc.)
- `genplot` – general Matplotlib/Seaborn plotting helpers (subplot layout, colorbars, color utilities, the `Ridge` plot class, etc.)
- `decorate` – reusable plot decoration helpers (`decorate_plot`, plot labels, legends, colorbars)
- `matrix` – KMeans/hierarchical clustering convenience functions and the `DFClust` class

> **Note:** This package intentionally duplicates code that also lives in the `survey` package. It was created as a temporary "fork" so that ongoing paper-submission notebooks relying on `survey` are not disrupted. Once the submission is complete, the duplicated functions will be removed from `survey` in favor of importing from `svytools`.

## System Requirements

Tested using Python 3.10+ on macOS (arm64) and Linux (x86_64).

## Installation

1. [Install mamba (or the latest conda).](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)

2. Clone this repository and create a conda environment from the `environment.yml` file:

    ```shell
    git clone <this_repo>
    cd svytools/
    mamba env create -f environment.yml
    mamba activate svytools-env
    ```

   This installs the package itself in editable mode (`pip install -e .`), so changes to the source are picked up immediately.

   Alternatively, if you already have an environment and just want to install the package into it:

    ```shell
    pip install -e /path/to/svytools
    ```

## Environment

To use the environment as a python kernel for Jupyter Lab:

```
(svytools-env) python -m ipykernel install --user --name svytools-env --display-name "svytools-env"
```

## Support

Please report issues or feature requests via GitHub.
