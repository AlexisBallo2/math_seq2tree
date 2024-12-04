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



# customTrain = [0.2051839464882943, 0.24727124727124727, 0.37530281779931157, 0.2994949494949495, 0.29484126984126985, 0.4001421464108031, 0.38095187499114563, 0.3396633458020073, 0.32822753272274274, 0.4145044010715653]
# customEval = [0.06255243464545791, 0.12754790429209034, 0.09326487466022351, 0.12951441323534346, 0.14853518574448807, 0.21059767106278737, 0.12378267727104934, 0.1025940467800933, 0.12352763515554213, 0.14337393872277593]

# regTrain = [0.06533094098883573, 0.11757022558239598, 0.16115685960984968, 0.23765432098765435, 0.20134822525843996, 0.24294871794871797, 0.14317048853439682, 0.19401903128318224, 0.18856837606837606, 0.19499062120710584]
# regEval = [0.08021745696164301, 0.13925299506694858, 0.15390449343937718, 0.16281083257827444, 0.22239001308768752, 0.18926809624484045, 0.15217960334239405, 0.07572066176717342, 0.19108694922648414, 0.15540789959394613]

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# # Plot the lines on their respective subplots
# ax2.plot(customTrain, label='Train')
# ax2.plot(customEval, label='Eval')

# ax1.plot(regTrain, label='Train')
# ax1.plot(regEval, label='Eval')

# # Basic configurations
# ax2.set_title('mGTS')
# ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Token Accuracy')
# ax2.legend()

# ax1.set_title('GTS')
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Token Accuracy')
# ax1.legend()

# # Automatically adjust the layout for better appearance
# plt.tight_layout()

# # Display the plots
# plt.show()
