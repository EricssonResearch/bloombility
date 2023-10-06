# Bloom - Federated vs. Split learning performance

![Structure of split vs federated learning](readme_resources/splitVsFedStructure.jpg)

Figure: Difference between split and federated learning architecture. <a href="https://par.nsf.gov/servlets/purl/10347871">Source</a>

## Table of Contents

1. [Code structure](#code_struct)
2. [Requirements](#req)
3. [Running the program](#run)

### Code structure <a name="code_struct"></a>
The project includes three main learning types: centralized, split and federated (averaging) learning.
For split and federated learning, a server-client architecture is launched, and training/testing data as well as parts of (split learning) or the entire model (federated learning) transmitted between them. In addition, helper classes and methods provide the following functionalities:

- read the project configurations
- load and/or distribute the datasets
- set up clients
- save and load snapshots of models
- evaluate the results




### Requirements <a name="req"></a>
Requirements can be found in ```requirements.txt``` (for now).

### Running the program <a name="run"></a>
Add after restructure.
