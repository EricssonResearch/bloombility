import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from bloom import ROOT_DIR


"""
Visualization of non-iid-ness using the data_distributor module.

    When visualization=True,
    a csv file is created in load_data/experiments,
    which contains a timestamp of the experiment and the sizes of each trainloader.

    In this script, the csv file is loaded, and the sizes of each dataset contrasted like in the paper on non-iid data splits
    seen here: https://colab.research.google.com/drive/1Bj14Quxo8afOTdAxP3s7aFdTK7RP9bGC?usp=sharing#scrollTo=DadGsLdqJLmy

 """


# Read the CSV file
dataset = os.path.join(ROOT_DIR, "load_data", "experiments", "dataset_sizes.csv")
if not os.path.exists(dataset):
    print(
        "No experiments found. Try running any type of learning with data_distributor flag visualization=True first."
    )
    quit()
df = pd.read_csv(
    dataset, header=None, low_memory=False, on_bad_lines="warn", names=range(14)
)

# Ignore the first two columns
df = df.iloc[:, 2:]


# Get the number of experiments
num_experiments = len(df)

# Iterate over each row in the DataFrame
max_dataset_len = 0
for i, row in df.iterrows():
    # Get the dataset sizes for the current experiment
    dataset_sizes = row.values
    log_sizes = np.log(dataset_sizes)

    # the dataset sizes become size of each dot on the plot.
    # X and y positions are determined by experiment run number and number of datasets in the experiment
    height = [i + 1] * len(dataset_sizes)
    positions = [i + 1 for i, _ in enumerate(dataset_sizes)]
    if len(dataset_sizes) > max_dataset_len:
        max_dataset_len = len(dataset_sizes)

    # add the line of dots representing one experiment to the plot
    plt.scatter(positions, height, s=log_sizes * 10, c=positions)

# Set the x and y axis labels
plt.xlabel("Number of Datasets")
plt.ylabel("Experiments")

locs, labels = plt.yticks()  # Get the current locations and labels.
plt.yticks(np.arange(1, 4, step=1))


locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(1, max_dataset_len + 1, step=1))  # Set label locations.
# Show the plot
plt.show()
