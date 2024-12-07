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



# customTrain = [0.4078822484410969, 0.6152076604343536, 0.6806112553292354, 0.7214262968057772, 0.7502410090972808, 0.7746797233980562, 0.7878475019927135, 0.7999760423141481, 0.815591856146939, 0.8255619379434703, 0.8334906061039741, 0.8407678241814737, 0.8501107056104581, 0.8548294890482456, 0.8584613965092498, 0.8648973547322968, 0.8698798780180202, 0.8755205452157925, 0.8761832320690065, 0.892735091038875]
# customEval = [0.19309334721295937, 0.24173030314524105, 0.31334369725441014, 0.31466574630591543, 0.3271703245851534, 0.3353425458901757, 0.34231992631769004, 0.3377015060653457, 0.35571419982349883, 0.40065212579683396, 0.3869394980513865, 0.40295119022484094, 0.39790543068077017, 0.4122757699735885, 0.3974897544290436, 0.4072627133548105, 0.40374538082085015, 0.3789861825883885, 0.39761106048018, 0.42251759989289916]

# regTrain = [0.41138731697318953, 0.6179958149766459, 0.6985423764402593, 0.7396135514332564, 0.7703172631057482, 0.7908389455632092, 0.8061481043706259, 0.8182113699454806, 0.8288065105772937, 0.8391981526547491, 0.8459742917593739, 0.8529060862880983, 0.8605904584120457, 0.8671975749016017, 0.8725912486113498, 0.875754705015643, 0.8798436566079542, 0.8844290635166373, 0.8871118676700139, 0.9034052247215254]
# regEval = [0.42797385290425555, 0.5232760987520009, 0.5954469025145291, 0.6281424298420487, 0.6491221006232696, 0.6735115317982145, 0.6695864207784843, 0.681401144376213, 0.6873153781417409, 0.7029142918402908, 0.702247905507754, 0.6971033836852619, 0.7049593490192407, 0.7139779333318644, 0.7179547889799848, 0.7218354127824508, 0.7155964434920637, 0.7250546292552268, 0.7241661950240992, 0.7330201110094259]

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# # Plot the lines on their respective subplots
# ax2.plot(customTrain, label='Train')
# ax2.plot(customEval, label='Eval')

# ax1.plot(regTrain, label='Train')
# ax1.plot(regEval, label='Eval')

# ax1.set_ylim([0,1])
# ax2.set_ylim([0,1])


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
