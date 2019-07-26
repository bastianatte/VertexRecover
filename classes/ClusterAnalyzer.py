from utils.misc import get_logger, tracks_csv_creator, plot_direc_creator, df_creator
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans, AgglomerativeClustering
from utils.plot_utils import plot_trks_mrg, plot_ssd, plot_mrg_evt, plot_truth_mrg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import logging
import time
import os
ca_logger = get_logger(__name__)
ca_logger.setLevel(logging.DEBUG)


class ClusterAnalyzer(object):
    def __init__(self, num_clusters, output_path):
        self.num_clusters = num_clusters
        self.output_path = output_path

    def df_exe(self, df, df_columns, feat_column, lab_columns):
        """
        Looping over the whole dataset, this method is able to
        identify the temporary dataframe for each merged vertex.
        This is done using event, merged_vertex and truth
        vertices informations.

        :param df: dataframe
        :param df_columns: dataframe columns name
        :param lab_columns: dataframe labels name
        :return: None
        """
        z = 0
        t = 0
        n_empty_mrg = 0
        n_tot_merg_evt = []
        n_tot_truth_mrg = []
        n_trks_merg = []
        n_evt_max = int(df["v_nEvent"].max())
        for event in range(1, 10):
            df_temp = df.loc[(df["v_nEvent"] == event), df_columns]
            n_merge_max = df_temp["v_nMergeEvent"].max()
            if n_merge_max != n_merge_max:
                n_empty_mrg = n_empty_mrg + 1
                continue
            n_tot_merg_evt.append(n_merge_max)
            for mrg in range(1, int(n_merge_max)):
                start = time.time()
                evt_str = str(event)
                mrg_str = str(mrg)
                df_temp1 = df_temp.loc[(df_temp["v_nEvent"] == event) &
                                       (df_temp["v_nMergeEvent"] == mrg),
                                       df_columns]
                n_truth_max_temp = df_temp1["v_closeThruth"].max()
                n_tot_truth_mrg.append(n_truth_max_temp)
                n_trks = df_temp1["v_closeThruth"].count()
                if n_trks >= self.num_clusters:
                    t = t + 1
                    n_row_str = str(n_truth_max_temp)
                    ca_logger.info(
                        "Event {}, Merge {}, nTruth {}, nTrks {}".format(
                            event, mrg, n_truth_max_temp, n_trks
                        )
                    )
                    df_temp1.drop(lab_columns, axis=1, inplace=True)
                    n_trks_merg.append(n_trks)
                    event_dir = os.path.join(self.output_path,
                                             "TempDF_Plt",
                                             "evt_" + evt_str)
                    plot_out_dir = os.path.join(event_dir, 'MrgVtx' +
                                                mrg_str + '_nTruth' +
                                                n_row_str)
                    if not os.path.exists(plot_out_dir):
                        os.makedirs(plot_out_dir)
                    self.cluster_exe(df_temp1, plot_out_dir)
                    # self.agg_clus(df_temp1)
                    stop = time.time()
                    ca_logger.info("Merged vertex cluster"
                                   "done in: {} sec".format(stop - start))
                else:
                    z += 1
                    ca_logger.info(
                        "Event {}, Merge {}, nTrks {},"
                        "BAD MERGED VERTEX!".format(
                            event, mrg, n_trks
                        )
                    )
                    continue
                if plot_out_dir is not None:
                    filename_path = os.path.join(plot_out_dir,
                                                 "df_temp_Tracks" +
                                                 str(n_trks) + ".csv")
                    df_temp1.to_csv(filename_path,
                                    header=True,
                                    index=False)
        plot_mrg_evt(n_tot_merg_evt, self.output_path)
        plot_truth_mrg(n_tot_truth_mrg, self.output_path)
        plot_trks_mrg(n_trks_merg, self.output_path)
        ca_logger.info("{} empty events!".format(n_empty_mrg))
        ca_logger.info("{} Merged vertices with less tracks"
                       "than nearby clusters=[{}]".format(z, self.num_clusters))
        ca_logger.info("{} Merged vertices with more tracks"
                       "than nearby clusters=[{}]".format(t, self.num_clusters))

    def df_linked_tracks_exe(self, df, df_columns, lab_columns):
        n_empty_mrg = 0
        n_mrg = 0
        n_event = 0
        n_evt_max = int(df["event_numb"].max())
        for event in range(1, n_evt_max+1):
            n_event += 1
            df_temp = df.loc[(df["event_numb"] == event), df_columns]
            n_merge_max = df_temp["merge_vtx"].max()
            if n_merge_max != n_merge_max:
                n_empty_mrg = n_empty_mrg + 1
                continue
            ca_logger.info("{} linked merged vertices!! ".format(n_merge_max))
            for mrg in range(1, (int(n_merge_max)+1)):
                ca_logger.info("event[{}] - Mrg Vtx[{}]".format(event, mrg))
                n_mrg += 1
                start = time.time()
                df_temp1 = df_temp.loc[(df_temp["event_numb"] == event) &
                                       (df_temp["merge_vtx"] == mrg),
                                       df_columns]
                n_trks = df_temp1["linked_type"].count()
                df_full, hs_df, pu_df = df_creator(df_temp1, lab_columns)
                if n_trks > 10:
                    ca_logger.info(
                        "Event {}, Merge {}, nTrks {} - "
                        "linked HS tracks shape: {}, "
                        "linked PU tracks shape: {}  ".format(event, mrg, n_trks,
                                                              hs_df.shape,
                                                              pu_df.shape)
                    )
                    plot_out_dir = plot_direc_creator(self.output_path, event,
                                                      mrg, n_trks)
                    tracks_csv_creator(plot_out_dir, df_full, hs_df,
                                       pu_df, n_trks)
                    self.cluster_exe(df_full, plot_out_dir)
                stop = time.time()
                ca_logger.info("Merged vertex cluster"
                               "done in: {} sec".format(stop - start))
        ca_logger.info("events: {}, mrg_vtxs: {} ".format(n_event, n_mrg))
        ca_logger.info("{} empty events!".format(n_empty_mrg))

    def cluster_exe(self, df, plot_out_dir):
        """
        Loops over the cluster's number to retrieve
        the results using different sklearn methods.
        :param df: dataframe
        :param plot_out_dir: plot directory
        :return: None
        """
        ssd = []
        km_silh_score = []
        silh_score_list = []
        kvalue_list =[]
        cluster_numbers = range(2, self.num_clusters)
        for cnt, k in enumerate(cluster_numbers):
            ssd_new, km_silh_score = self.kmean_analyzer(k, df, ssd,
                                                         plot_out_dir)
            cls, cls_silh_score = self.agg_clus_analyzer(k, df)
        plot_ssd(cluster_numbers, ssd_new,
                 km_silh_score, plot_out_dir)

    def kmean_analyzer(self, k, df, ssd, plot_out_dir):
        """
        Run kmean analysis to exctact useful features
        for clustering.
        :param k: cluster number/s
        :param df: dataframe
        :param ssd: sum of squared distances
        :param plot_out_dir: output_plot_path
        :return: ssd, silhouette_score
        """
        km = KMeans(n_clusters=k, random_state=10)
        km = km.fit(df)
        ssd.append(km.inertia_)
        km_lab = km.fit_predict(df)
        km_silh_score = silhouette_score(df, km_lab)
        km_silh_value = silhouette_samples(df, km_lab)
        self.plt_silh(k, df, km, km_lab, km_silh_score,
                      km_silh_value, plot_out_dir)
        ca_logger.info("KMEANS - "
                       "clus: {}, "
                       "sil_score: {}".format(k, km_silh_score))
        return ssd, km_silh_score

    def agg_clus_analyzer(self, k, df):
        """
        Run agglomerative clustering method to exctact
        useful features for clustering.
        :param k: cluster number/s
        :param df: dataframe
        :return: cls, cls silhouette score
        """
        cls = AgglomerativeClustering(n_clusters=k)
        cls = cls.fit(df)
        cls_lab = cls.labels_
        cls_silh_score = silhouette_score(df, cls_lab)
        ca_logger.info("AGGCLS - "
                       "clus: {}, "
                       "sil_score: {}".format(k, cls_silh_score))
        return cls, cls_silh_score

    def plt_silh(self, k, df, km, km_lab, silh_score, silh_value, out_path):
        """
        Plotting the silhouette parameter given by kmeans clustering.
        :param k: cluster number
        :param df: dataframe
        :param km: kmean object
        :param km_lab: kmeand labels
        :param silh_score: silhouette score
        :param silh_value: silhouette sample
        :param out_path: plot directory
        :return: None
        """
        ncs = str(k)
        # nCol = df.shape[1]
        y_lower = 10
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(20, 7)
        ax1.set_xlim([-0.1, 1.2])
        ax1.set_ylim([0, len(df["clus_d0"])+(self.num_clusters+1)*10])
        for i in range(k-1): # k-1
            n_clusters_str = str(i)
            ith_cluster_silhouette_values = silh_value[km_lab == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / k)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color,
                              alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # 10 for the 0 samples
        ax1.set_title("Silhouette plot for the various clusters")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silh_score, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        colors = cm.nipy_spectral(km_lab.astype(float) / k)
        ax2.scatter(df["clus_d0"], df["clus_z0"], marker='o',
                    s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
        centers = km.cluster_centers_
        ax2.scatter(centers[:, 3], centers[:, 4], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("d0")
        ax2.set_ylabel("z0")
        plt.suptitle(("Silhouette analysis "
                      "with n_clusters = %d" % k),
                     fontsize=14, fontweight='bold')
        name = os.path.join(out_path, "silhouette_plt_"
                            +ncs+"_"+n_clusters_str+"_Clus.png")
        plt.savefig(name, dpi=300)
        plt.close()
        return None
