import matplotlib.pyplot as plt


def make_loss_graph(data, filename, title, xlabel, ylabel):
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

    plt.plot(data)
    # plt.plot(data)
    # plt.plot(validationLoss)
    # plt.figlegend(['Train', 'Test'], loc='upper left')
    # plt.figlegend([f'Fold {i}' for i in range(len(data))], loc='upper right')
    # plt.show()
    plt.savefig(filename)
    plt.clf()

def make_eval_graph(data2, legend, filename, title, xlabel, ylabel):
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

    t = [i for i in range(len(data2[0]))]
    for i in range(len(data2)):
        plt.plot(t, data2[i])
    # plt.plot(data)
    # plt.plot(validationLoss)
    # plt.figlegend(['Train', 'Test'], loc='upper left')
    # plt.figlegend([f'Fold {i}' for i in range(len(data))], loc='upper right')
    plt.figlegend(legend, loc='upper left')
    # plt.show()
    plt.savefig(filename)
    plt.clf()