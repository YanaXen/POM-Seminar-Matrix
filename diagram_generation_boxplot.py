from matplotlib import pyplot as plt
import pandas as pd
import ast
import numpy as np


def create_diagram(x, labels, xlabel, ylabel, name, flexibility):
    """
    Creates and saves a boxplot for the given data groups.

    Args:
        x (list): Data groups to plot.
        labels (list): Labels for the boxplot groups.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        name (str): Title of the diagram.
        flexibility (str): Identifier used in the output file name.

    Returns:
        None: The diagram is saved as a PNG file.
    """

    # Create a new figure and axis for the boxplot
    fig, ax = plt.subplots()

    # Set axis labels
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Draw the boxplot for the provided data groups
    bplot = ax.boxplot(x,  
                    labels=labels) 
    
    # Set the plot title
    ax.set_title(name)

    # Use a fixed upper bound for utilization metrics.
    # For all other metrics, derive the upper limit from the data.
    if ylabel in ("Utilization", "Active Interval Utilization"):
        ymax = 1
    else:
        ymax = np.nanmax([np.nanmax(np.asarray(group, dtype=float)) for group in x])
        ymax += 2
    ax.set_ylim(0, ymax)
    
    # Compute the median of each group for inspection and debugging output
    medians = []
    for group in x:
        arr = np.asarray(group, dtype=float)
        arr = arr[~np.isnan(arr)]
        med = np.median(arr) if arr.size else np.nan
        medians.append(med)

    # print(ylabel)
    # print(medians)

    # Save the figure and close it to free resources
    fig.savefig(f"diagrams/diagram_boxplot_{flexibility}_{ylabel}.png")
    plt.close(fig)


def get_usage(series, active_interval_utilization=False):
    """
    Computes the average machine utilization for each run in a series.

    Args:
        series (list): Per-run utilization data.
        active_interval_utilization (bool, optional): If True, uses the active
            interval utilization value instead of the regular utilization value.
            Defaults to False.

    Returns:
        avg_utils (list): Average utilization value for each run.
    """
    avg_utils = []
    #  Select the tuple index for the utilization value depending on the input type
    index = 3 if active_interval_utilization else 2
    
    # Process each run separately
    for run_usage in series:           
        # Extract the relevant utilization values from the machine tuples
        rels = [u[index] for u in run_usage]
        # Ignore the first and last machine if more than two entries are present
        if len(rels) > 2:
            rels = rels[1:-1]    
        # Compute the average utilization for the current run      
        avg_utils.append(sum(rels) / len(rels) if rels else None)
    return avg_utils

def experiment_four_diagramm(x, labels, xlabel, ylabel, name, flexibility):
    """
    Creates and saves a line plot for experiment 4 across different layer settings.

    Args:
        x (list): X-values for the plotted series.
        labels (list): Labels for the plotted lines.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        name (str): Title of the diagram.
        flexibility (str): Identifier used in the output file name.

    Returns:
        None: The diagram is saved as a PNG file.
    """

    # Create a square figure for the diagram
    figure = plt.figure(figsize=(8,8))

    # Define the layer settings shown on the y-axis
    y = ["3 Layers", "4 Layers", "5 Layers", "6 Layers"]

    # Plot one line for each input series
    for x_values, label in zip(x, labels):
        plt.plot(x=x_values, y=y, label=label)

    # Set axis labels, legend, and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(name)

    # Save the figure and close it to free resources
    figure.savefig(f"diagrams/diagram_{flexibility}_{ylabel}_batch_sizes.png")
    plt.close(figure)

def plot_experiment4_utilization(data, ylabel="Utilization", flexibility="Line"):
    """
    Plots median utilization values for experiment 4 across layer settings and
    batch sizes.

    Args:
        data (pandas.DataFrame): Data filtered to experiment 4.
        ylabel (str, optional): Label for the y-axis. Defaults to
            "Utilization".
        flexibility (str, optional): Identifier used in the output file name.
            Defaults to "Line".

    Returns:
        None: The diagram is saved as a PNG file.
    """

    # Split the experiment version into layer setting and batch size
    tmp = data.copy()
    tmp[["layer", "batch"]] = tmp["experiment_version"].str.split("-", n=1, expand=True)
    tmp["layer"] = tmp["layer"].str.strip()
    tmp["batch"] = tmp["batch"].str.strip()

    # Reduce the stored usage information to one scalar utilization value per run
    tmp["util"] = get_usage(tmp["usage"])

    # Define the plotting order for layers and batch sizes
    layer_order = ["3 Layers", "4 Layers", "5 Layers", "6 Layers"]
    batch_order = ["Batch Size = 15", "Batch Size = 30", "Batch Size = 45"]

    # Aggregate utilization by taking the median for each (batch, layer) pair
    agg = (
        tmp.groupby(["batch", "layer"])["util"]
           .median()
           .reset_index()
    )

    # Create the figure and x positions for the layer categories
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(layer_order))

    # Plot one line per batch size across all layer settings
    for b in batch_order:
        ys = []
        for layer in layer_order:
            val = agg[(agg["batch"] == b) & (agg["layer"] == layer)]["util"]
            ys.append(val.iloc[0] if len(val) else np.nan)
        ax.plot(x, ys, marker="o", label=b)

    # Configure axes, title, limits, and legend
    ax.set_xticks(x)
    ax.set_xticklabels(layer_order)
    ax.set_xlabel("Line flexibility")
    ax.set_ylabel(ylabel)
    ax.set_title("Comparison of utilization depending on batch size")
    ax.set_ylim(0, 1)
    ax.legend()

    # Save the figure and close it to free resources
    fig.savefig(f"diagrams/diagram_{flexibility}_{ylabel}_batch_sizes.png")
    plt.close(fig)


def diagram_generation(data, labels, flexibility, experiment):
    """
    Generates the diagrams for a given experiment based on the stored results.

    Args:
        data (pandas.DataFrame): Experiment results used for diagram creation.
        labels (list): Labels of the experiment versions to compare.
        flexibility (str): Name of the flexibility dimension shown in the plots.
        experiment (str): Name of the experiment to evaluate.

    Returns:
        None: The generated diagrams are saved as image files.
    """

    # Unpack the labels depending on how many comparison groups are given
    if len(labels)==3:
        label1, label2, label3 = labels
    else:
        label1, label2, label3, label4 = labels

    # Convert serialized list-like columns back to Python objects if needed
    cols_to_parse = ["usage", "active_usage", "downtime_per_machine", "downtime_over_makespan"]
    for col in cols_to_parse:
        data[col] = data[col].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)

    # For all experiments except experiment 4, create boxplots for makespan
    # and standard utilization.
    if experiment != "Experiment 4":
        # Collect makespan values per experiment version
        values_1 = data[data["experiment_version"] == label1]["makespan"]
        values_2 = data[data["experiment_version"] == label2]["makespan"]
        values_3 = data[data["experiment_version"] == label3]["makespan"]
        if len(labels)>3:
            values_4 = data[data["experiment_version"] == label4]["makespan"]
            x = [values_1, values_2, values_3, values_4]
        else:
            x = [values_1, values_2, values_3]

        create_diagram(x, labels, f"{flexibility} Flexibility", "Makespan", f"Makespan comparison for {flexibility} Flexibility levels", flexibility)

        # Collect usage values and reduce them to one utilization value per run
        values_1 = data[data["experiment_version"] == label1]["usage"]
        values_2 = data[data["experiment_version"] == label2]["usage"]
        values_3 = data[data["experiment_version"] == label3]["usage"]

        values_1 = get_usage(values_1)
        values_2 = get_usage(values_2)
        values_3 = get_usage(values_3)
        if len(labels)>3:
            values_4 = data[data["experiment_version"] == label4]["usage"]
            values_4 = get_usage(values_4)
            x = [values_1, values_2, values_3, values_4]
        else:
            x = [values_1, values_2, values_3]
        create_diagram(x, labels, f"{flexibility} Flexibility", "Utilization", f"Utilization comparison for {flexibility} Flexibility levels", flexibility)

    # For experiment 3, additionally compare active interval utilization
    if experiment == "Experiment 3":
        values_1 = data[data["experiment_version"] == label1]["active_usage"]
        values_2 = data[data["experiment_version"] == label2]["active_usage"]
        values_3 = data[data["experiment_version"] == label3]["active_usage"]

        values_1 = get_usage(values_1, True)
        values_2 = get_usage(values_2, True)
        values_3 = get_usage(values_3, True)
        if len(labels)>3:
            values_4 = data[data["experiment_version"] == label4]["active_usage"]
            values_4 = get_usage(values_4, True)
            x = [values_1, values_2, values_3, values_4]
        else:
            x = [values_1, values_2, values_3]
        create_diagram(x, labels, f"{flexibility} Flexibility", "Active Interval Utilization", f"Active Interval Utilization comparison for {flexibility} Flexibility levels", flexibility)

    # For experiment 4, create the dedicated utilization plot across batch sizes
    if experiment == "Experiment 4":
        plot_experiment4_utilization(data, ylabel="Utilization", flexibility=flexibility)


def run_diagram_creation(data):
    """
    Runs diagram generation for all experiments contained in the input data.

    Args:
        data (pandas.DataFrame): Result data containing all experiments.

    Returns:
        None: The generated diagrams are saved as image files.
    """
    # Determine all experiment names present in the data
    experiments = pd.unique(data["experiment"])

    # Process each experiment separately
    for experiment in experiments:
        # Build a filtered copy for the current experiment
        subset = data[data["experiment"] == experiment].copy()

        # Define the flexibility type and labels used for the current experiment
        if "Experiment 1" in experiment:
            flexibility = "Resource"
            labels = ["No Flexibility", "Low Flexibility", "High Flexibility"]
        elif "Experiment 2" in experiment:
            flexibility = "Layout"
            labels = ["4 Layers", "5 Layers", "6 Layers"]
        elif "Experiment 3" in experiment:
            flexibility = "Line"
            labels = ["3 Layers", "4 Layers", "5 Layers", "6 Layers"]
        elif "Experiment 4" in experiment:
            flexibility = "Line"
            labels = ["Batch Size = 15", "Batch Size = 30", "Batch Size = 45"]

        # Generate the diagrams for the selected experiment subset
        diagram_generation(subset, labels, flexibility, experiment)


if __name__ == "__main__":
    data = pd.read_csv("data/results.csv")
    run_diagram_creation(data)