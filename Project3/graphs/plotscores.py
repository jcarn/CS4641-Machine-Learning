import matplotlib.pyplot as plt
import numpy as np

def read_datafile(file_name):
    # the skiprows keyword is for heading, but I don't know if trailing lines
    # can be specified
    data = np.loadtxt(file_name, delimiter=',')
    return data




def plot_curve(title, iterations, y_train, y_test, ylim=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    train_scores_mean = y_train#np.mean(y_train, axis=0)
    train_scores_std = np.std(y_train, axis=0)
    test_scores_mean = y_test#np.mean(y_test, axis=0)
    test_scores_std = np.std(y_test, axis=0)
    plt.grid()

    plt.fill_between(iterations, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(iterations, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(iterations, train_scores_mean, ',-', color="r",
             label="Training Error")
    plt.plot(iterations, test_scores_mean, ',-', color="g",
             label="Test Error")
    t = title.replace(" ", "_")
    t = t.replace(",", "")
    t = t.replace(":", "")
    plt.savefig("plots/" + str(t) + ".png")
    plt.legend(loc="best")
    plt.show()
    return plt
1
def plot_clusters(title, clusters, y_acc, y_time, ylim=None, err_tit = "Error"):
    # plt.figure()
    if ylim is not None:
        plt.ylim(*ylim)
    fig, ax1 = plt.subplots()
    plt.title(title)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(err_tit)
    ax1.plot
    train_scores_mean = y_acc#np.mean(y_train, axis=0)
    ax1.plot(clusters, train_scores_mean, ',-', color="r", label="Reward")
    plt.grid()

    ax2 = ax1.twinx()
    # train_scores_std = np.std(y_train, axis=0)
    test_scores_mean = y_time#np.mean(y_test, axis=0)
    ax2.set_ylabel("Runtime (s)")
    # test_scores_std = np.std(y_test, axis=0)
    #
    # plt.fill_between(iterations, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(iterations, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax2.plot(clusters, test_scores_mean, ',-', color="g",
             label="Steps")
    t = title.replace(" ", "_")
    t = t.replace(",", "")
    t = t.replace(":", "")
    fig.savefig(str(t) + ".png")
    fig.legend(loc="best")
    fig.show()
    plt.show()
    return plt

def plot_curve_notest(title, iterations, rhc, sa, ga, mimic, ylim=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.grid()

    plt.plot(iterations, rhc, ',-', color="r",
             label="RHC")
    plt.plot(iterations, sa, ',-', color="g",
             label="SA")
    plt.plot(iterations, ga, ',-', color="b",
             label="GA")
    plt.plot(iterations, mimic, ',-', color="y",
             label="MIMIC")
    t = title.replace(" ", "_")
    t = t.replace(",", "")
    t = t.replace(":", "")
    plt.legend(loc="best")
    plt.savefig("plots/" + str(t) + ".png")
    plt.show()
    return plt



# train_results = read_datafile("backprop_train.csv")
# test_results = read_datafile("backprop_test.csv")
def plot_method(plt_title, filename):
    train_results = read_datafile(filename + "_train.csv")
    test_results = read_datafile(filename + "_test.csv")
    its = np.array([5*i for i in range(0, len(train_results))])
    plot_curve(plt_title, its, train_results, test_results)

def plot_method_notest(plt_title, filename):
    rhc_results = read_datafile(filename + "_rhc_train.csv")
    sa_results = read_datafile(filename + "_sa_train.csv")
    ga_results = read_datafile(filename + "_ga_train.csv")
    mimic_results = read_datafile(filename + "_mimic_train.csv")

    its = np.array([5*i for i in range(0, len(rhc_results))])
    plot_curve_notest(plt_title, its, rhc_results, sa_results, ga_results, mimic_results)

def clusterplot(title, filename, err_tit = "Error"):
    file = read_datafile(filename)
    plot_clusters(title, file[:, 0], file[:, 1], file[:,2], err_tit=err_tit)

# clusterplot("K-Means Big-Set", "./results/kmeansmulti.csv")
# clusterplot("EM Big-Set", "./results/emmulti.csv", err_tit = "Log-Likelihood")
#clusterplot("K-Means Small-Set", "./results/kmeanssingle.csv", err_tit = "Log-Likelihood")
#clusterplot("EM Small-Set", "./results/emsingle.csv", err_tit = "Error (SSE)")
clusterplot("Maze Grid VI", "./results/emsingle.csv", err_tit = "Error (SSE)")
