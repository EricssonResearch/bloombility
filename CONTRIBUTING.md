# Contributing to Bloom

## Initial setup of project Bloom for development

### Installation

1. Create a virtual environment for python dependencies. Use following command in terminal.

    ```shell
    python -m venv venv
    ```

2. To activate or re-activate the environment, run the following command.

    - in Linux / MacOS / git console:

        ```shell
        source venv/Scripts/activate
        ```

    - in Windows cmd.exe:

        ```shell
        venv\Scripts\activate.bat
        ```

    - in Windows PowerShell:

        ```shell
        venv\Scripts\Activate.ps1
        ```

    (Note that you need to use ```python```, not ```python3``` to run your scripts.)

3. Install all dependencies including dev dependencies specified in ```pyproject.toml```. Using '-e' flag makes the project editable and easy to modify.

    ```shell
    pip install -e .[dev]
    ```

4. Install ```pre-commit``` to the directory.

    ```shell
    pre-commit install
    ```

After these steps, code formatting and other hooks defined in ```.pre-commit-config.yaml``` should run locally before each commit on modified files. This will modifiy staged files and you might need to restage them again for commit if any issues are fixed.

### How to use installed package

With editable package structure, you can import the package to anywhere you want as shown below.

```python
import bloom
```
