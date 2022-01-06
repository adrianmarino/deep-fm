import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics(logs, warmup_count=0):
    metric_names = logs.keys()
    epochs = logs['epoch'][warmup_count:]

    sns.set_style("darkgrid")
    for name in metric_names:
        if 'epoch' != name:
            sns.lineplot(
                x=epochs,
                y=logs[name][warmup_count:],
                label=name.capitalize()
            )
    plt.xlabel("Epocs")
    plt.title("Metrics")
    plt.tight_layout()
    plt.show()