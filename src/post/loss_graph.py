import matplotlib.pyplot as plt


def make_loss_graph(data, legend, filename, title, xlabel, ylabel):
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

    t = [i for i in range(len(data[0]))]
    for i in range(len(data)):
        plt.plot(t, data[i])
    # plt.plot(data)
    # plt.plot(validationLoss)
    # plt.figlegend(['Train', 'Test'], loc='upper left')
    # plt.figlegend([f'Fold {i}' for i in range(len(data))], loc='upper right')
    plt.figlegend(legend, loc='upper left')
    # plt.show()
    plt.savefig(filename)