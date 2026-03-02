from matplotlib import pyplot as plt
import pandas as pd
import ast

def get_title(experiment, ylabel):
    if experiment=="Experiment 1":
        return f"{ylabel} comparison for Line Flexibility levels"
    elif experiment=="Experiment 2":
        return f"{experiment}: Varying Machine Flexibility"
    elif experiment=="Experiment 3":
        return f"{experiment}: Varying Batch Size"
    elif experiment=="Experiment 4":
        return f"{experiment}: Practical Example"
    elif experiment=="Experiment 5":
        return f"{experiment}: Varying Machine Flexibility 2"
    else:
        return "Title"

def create_diagram(x, y, experiment, xlabel, ylabel):
        fig = plt.figure(figsize=(8, 5))
        plt.plot(x, y, color="blue", marker="o")

        # if we would have a linear development
        x = pd.Series(x).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True) 

        if experiment == "Experiment 3" and ylabel not in ["Utilization", "Active Interval Utilization"]:
            x_steps = [0 if index==0 else ((x[index]-x[index-1])/x[index-1]) for index in range(len(x))]
            y_lin = []
            for index, increase in enumerate(x_steps):
                if index==0:
                    y_lin.append(y[0])
                else:
                    y_lin.append(y_lin[index-1] * (1+increase))

            plt.plot(x, y_lin, color="red", linestyle = 'dotted', label="linear development")
            plt.legend()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if (ylabel == "Utilization" or ylabel == "Active Interval Utilization"):
            ymax = 1
        else:
            ymax = max(y) + 2

        plt.ylim(0, ymax)
        plt.title(get_title(experiment, ylabel))
        name = experiment.replace(" ", "")
        ylabel = ylabel.replace(" ", "")
        fig.savefig(f"diagrams/diagram_{name}_{ylabel}.png")
        plt.close(fig)

def diagram_generation(data):
    experiments = pd.unique(data["experiment"])

    cols_to_parse = ["usage", "active_usage", "downtime_per_machine", "downtime_over_makespan"]

    for col in cols_to_parse:
        data[col] = data[col].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)

    for experiment in experiments:
        experiment_data = data[data["experiment"]==experiment].copy()
        if experiment == "Experiment 3":
            x = experiment_data["total_work_content"]
            xlabel = "Total work content"
        else:
            x = experiment_data["experiment_version"]
            xlabel = "Line Flexibility"

        # create makespan diagram
        y = experiment_data["makespan"]

        create_diagram(x, y, experiment, xlabel, "Makespan")

        # create usage diagram
        y = experiment_data["usage"]

        avg_util = []
        for variant in y:
            util_sum = 0
            counter = 0
            for index, u in enumerate(variant): 
                if index == 0 or index == len(variant)-1:
                    continue
                util_sum += u[2]
                counter +=1
            avg_util.append(util_sum / counter)

        create_diagram(x, avg_util, experiment, xlabel, "Utilization")

        # create active usage diagram
        y = experiment_data["active_usage"]

        avg_util = []
        for variant in y:
            util_sum = 0
            counter = 0
            for index, u in enumerate(variant): 
                if index == 0 or index == len(variant)-1:
                    continue
                util_sum += u[3]
                counter +=1
            avg_util.append(util_sum / counter)


        create_diagram(x, avg_util, experiment, xlabel, "Active Interval Utilization")


if __name__ == "__main__":
    data = pd.read_csv("data/experiment_results_backup_9.csv")
    diagram_generation(data)