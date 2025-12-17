import matplotlib.pyplot as plt
import numpy as np


def plot_before_after(before, after, metric = "mean_vo2"):
    data_pairs = [(before, after, metric)]

    fig, axes = plt.subplots(2, 1, sharey=False)

    for ax, (left, right, title) in zip(axes, data_pairs):

       
        x_left = np.zeros_like(left)
        x_right = np.ones_like(right)

        # individual lines
        for l, r in zip(left, right):
            ax.plot([0, 1], [l, r], color='gray', alpha=0.7)
    
        # mean line
        ax.plot([0, 1], [np.mean(left), np.mean(right)],
                lw=3)
    
        # mean markers
        ax.scatter([0, 1], [np.mean(left), np.mean(right)],
                  s=80, zorder=3)
    
        # Optional: individual markers
        ax.scatter(x_left, left, color='gray', s=30)
        ax.scatter(x_right, right, color='gray', s=30)
    
        # formatting
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['period before \n perturbation', 'period after \n perturbation'])
        ax.set_title(title)
        ax.grid(True, axis='y')
       # ax.set_ylabel('vo2 [L/min]')
    
    return fig

def plot_response(responses):
    fig = plt.figure()
    mean_res = np.mean(responses[:,:,0], axis=0)
    time = np.arange(-1000, 4500)
    plt.plot(time, mean_res, linewidth = 3, label='mean response')
    for i in range(responses.shape[0]):
        plt.plot(time, responses[i, :,0], color='gray', alpha = 0.5)
#plt.vlines(1000, 0.25, 1.5, color='red', linestyles = 'dashed', label='time of perturbation')
    plt.vlines(0, min(mean_res), max(mean_res), color='red', linestyles = 'dashed', label='time of perturbation')
    ticks = plt.gca().get_xticks()

# Replace labels with ticks/100
    plt.gca().set_xticklabels([t/100 for t in ticks])

    plt.xlabel('time [s]')
    #plt.ylabel('Ventilation rate [L/min]')
    plt.legend()
    return fig

def subplot_response(ax, responses):
    mean_res = np.mean(responses[:, :, 0], axis=0)
    time = np.arange(-1000, 4500)

    ax.plot(time, mean_res, linewidth=3, label='mean response')

    for i in range(responses.shape[0]):
        ax.plot(time, responses[i, :, 0], color='gray', alpha=0.5)

    ax.vlines(
        0,
        min(mean_res),
        max(mean_res),
        color='red',
        linestyles='dashed',
        label='time of perturbation'
    )

    ticks = ax.get_xticks()
    ax.set_xticklabels([t / 100 for t in ticks])

    ax.set_xlabel('time [s]')
    ax.legend()

