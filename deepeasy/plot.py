"""画图。

Create:   2019-02-25
Modified: 2019-02-25
"""

import matplotlib.pyplot as plt


def plot_nn_history(neural_network) -> None:

    fig = plt.figure(figsize=(10, 3), dpi=120)
    subplot_kw = {
        'xlabel': 'Epochs',
    }
    ax1, ax2 = fig.subplots(1, 2, subplot_kw=subplot_kw)
    ax1.set_title('Cost')
    ax2.set_title('Accuracy')

    epochs_pre: int = 0
    for i in range(neural_network.train_count + 1):
        cost_history = neural_network.history[f'cost{i}']
        accuracy_history = neural_network.history[f'accuracy{i}']
        epochs = len(cost_history)
        if epochs > epochs_pre:
            ax1.set_xlim(0, epochs)
            ax2.set_xlim(0, epochs)

        x_axis = list(range(1, epochs + 1))

        ax1.plot(
            x_axis, cost_history, label=f'Train {i+1}'
        )
        ax1.grid(True)

        ax2.plot(
            x_axis, accuracy_history, label=f'Train {i+1}'
        )
        ax2.grid(True)

        epochs_pre = epochs

    ax1.legend(loc='upper right', shadow=True)
    ax2.legend(loc='lower right', shadow=True)

    plt.show()
