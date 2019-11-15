# from sklearn.metrics import silhouette_score, silhouette_samples
# from sklearn.cluster import KMeans
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def plot_surface(k, scr, ratio, id_data):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(k)
    y = np.arange(scr)
    z = np.arange(ratio)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    name = os.path.join(id_data, "surface.png")
    plt.savefig(name, dpi=200)
    plt.close()


def plot_ssd(k, sum_of_squared_distances, labels, id_data):
    plt.plot(k, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    #plt.legend(labels)
    name = os.path.join(id_data, "ssd.png")
    plt.savefig(name, dpi=200)
    plt.close()


def plot_scr_vs_ratio(scr, ratio, id_data):
    plt.scatter(ratio, scr, s=0.1)
    plt.xlabel('ratio hs/pu')
    plt.ylabel('silhouette score')
    plt.title('ratio vs silhouette score')
    name = os.path.join(id_data, "ratio_vs_score.png")
    plt.savefig(name, dpi=400)
    plt.close()


def plot_scr_vs_percent(scr, percent, id_data):
    plt.scatter(scr, percent, s=0.1, alpha=0.7)
    plt.xlabel('silhouette score')
    plt.ylabel('hs tracks %')
    plt.title('hs tracks percentage vs silhouette score')
    name = os.path.join(id_data, "perc_vs_score.png")
    plt.savefig(name, dpi=400)
    plt.close()


def plot_cn_vs_scr(k, scr, agg_scr, id_data):
    plt.scatter(k, scr, s=1, c="red", marker='1')
    plt.scatter(k, agg_scr, s=1, c="blue", marker='2')
    plt.xlabel('k')
    plt.ylabel('silhouette score')
    plt.title('silhouette score vs cluster number')
    name = os.path.join(id_data, "cluster_vs_score.png")
    plt.savefig(name, dpi=400)
    plt.close()


def plot_score(scr, agg_scr, id_data):
    plt.plot(scr, c="red", marker='o',
             linestyle='dotted',
             linewidth=0.5, markersize=1)
    plt.plot(agg_scr, c="blue", marker='o',
             linestyle='dashed',
             linewidth=0.1, markersize=0.1)
    plt.xlabel('silhouette score')
    plt.title('silhouette score')
    name = os.path.join(id_data, "score.png")
    plt.savefig(name, dpi=400)
    plt.close()


def plot_scatter_score_km_vs_agg(km_scr, agg_scr, id_data):
    plt.scatter(km_scr, agg_scr,
                marker='.',
                s=0.1)
    plt.xlabel("Kmeans slh score")
    plt.ylabel("Agg slh score")
    plt.title("score comparison")
    name = os.path.join(id_data, "comparison_score.png")
    plt.savefig(name, dpi=400)
    plt.close()


def plot_correlation(df, id_data):
    plt.figure(figsize=(15, 15))
    data1 = df[["clus_pt", "clus_eta", "clus_phi", "clus_d0", "clus_z0"]]
    cor = data1.corr()
    fig = sns.heatmap(cor, xticklabels=1, yticklabels=1,
                      cmap="YlGnBu", linewidths=.8)  # square = True, cbar = True,
    name = os.path.join(id_data, "correlation_matrix.png")
    fig.figure.savefig(name, dpi=300)
    plt.close()
    return None


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


def plot_linked_tracks_merge(n_tracks, out_path, png_name):
    plt.hist(n_tracks, bins=300)
    plt.title(png_name)
    plt.xlabel("tracks")
    plt.ylabel("count")
    name = os.path.join(out_path, png_name + ".png")
    plt.savefig(name, dpi=300)
    plt.close()


def plot_linked_tracks_merge_both(n_tracks1, n_tracks2, out_path, png_name):
    plt.hist(n_tracks1, bins=300, color="red", label="HS", alpha=0.8)
    plt.hist(n_tracks2, bins=300, color="blue", label="PU", alpha=0.8)
    plt.title(png_name)
    plt.xlabel("tracks")
    plt.ylabel("count")
    red_patch = mpatches.Patch(color='red', label='HS Collection')
    blue_patch = mpatches.Patch(color='blue', label='PU Collection')
    plt.legend(handles=[red_patch, blue_patch])
    name = os.path.join(out_path, png_name + ".png")
    plt.savefig(name, dpi=300)
    plt.close()



