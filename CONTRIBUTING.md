# Contributing to Bloom

## Here you can find about initial setup of project Bloom for development.

### Installation

1. Create a virtual environment for python dependencies. Use following command in terminal.

    ```python -m venv venv```

2. Install all dependencies including dev dependencies specified in pyproject.toml. Using '-e' flag makes the project editable and easy to modify.

    ```pip install -e .[dev]```

3. Install pre-commit to the directory.

    ```pre-commit install```

After these steps, code formatting and other hooks defined in .pre-commit-config.yaml should run locally before each commit on modified files. This will modifiy satged files and you might need to restage them again for commit if any issues are fixed.


### How to use installed package

With editable package structure, you can import the package to anywhere you want as shown below.

```import bloom```