import matplotlib.pyplot as plt
import os


def plot_ssd(K, sum_of_squared_distances, id_data):
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    name = os.path.join(id_data, "ssd.png")
    plt.savefig(name, dpi=200)
    plt.close()


def plot_trks_mrg(n_trks, out_path):
    plt.hist(n_trks, bins=200)
    plt.title("# tracks per merged vertex ")
    plt.xlabel("# Trks")
    plt.ylabel("Merge")
    name = os.path.join(out_path, "nTrks_mrg.png")
    plt.savefig(name, dpi=300)
    plt.close()


def plot_mrg_evt(n_merge, out_path):
    plt.hist(n_merge, bins=100)
    plt.title("# Merge vertex per event")
    plt.xlabel("# Merge")
    plt.ylabel("Event")
    name = os.path.join(out_path, "nMerge_evt.png")
    plt.savefig(name, dpi=300)
    plt.close()


def plot_truth_mrg(n_truth, out_path):
    if n_truth == n_truth:
        truth_bins = [0, 1, 2, 3, 4, 5,
                      6, 7, 8, 9, 10]
        plt.hist(n_truth, bins=truth_bins)
        plt.title("n tot truth per merge vertex")
        plt.xlabel("#Truth")
        plt.ylabel("Merge")
        name = os.path.join(out_path, "nTruth_mrg.png")
        plt.savefig(name, dpi=300)
        plt.close()
