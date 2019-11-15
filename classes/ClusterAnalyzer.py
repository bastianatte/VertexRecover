from utils.misc import get_logger, tracks_csv_creator, plot_direc_creator, df_creator
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans, AgglomerativeClustering
from utils.plot_utils import plot_trks_mrg, plot_ssd, plot_mrg_evt, plot_truth_mrg, plot_linked_tracks_merge, \
    plot_linked_tracks_merge_both, plot_cn_vs_scr, plot_score, plot_scr_vs_ratio,\
    plot_scatter_score_km_vs_agg, plot_scr_vs_percent
from utils.plot_centroids import plot_centroid_d0z0, plot_centroid_d0z0_focused, plot_colored_data
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
        bad_vtx = 0
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
                    # self.cluster_exe(df_temp1, plot_out_dir)
                    # self.agg_clus(df_temp1)
                    stop = time.time()
                    ca_logger.info("Merged vertex cluster "
                                   "done in: {} sec".format(stop - start))
                else:
                    bad_vtx += 1
                    ca_logger.info(
                        "Event {}, Merge {}, nTrks {},"
                        "NOT ENOUGH TRACKS, BAD MERGED VERTEX!".format(
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
                       "than nearby clusters=[{}]".format(bad_vtx, self.num_clusters))
        ca_logger.info("{} Merged vertices with more tracks"
                       "than nearby clusters=[{}]".format(t, self.num_clusters))

    def df_linked_tracks_exe(self, df, df_columns, lab_columns):
        n_empty_mrg = 0
        n_mrg = 0
        n_event = 0
        n_good_merge = 0
        n_hs_tracks_mrg = []
        n_pu_tracks_mrg = []
        n_full_tracks_mrg = []
        km_slh_list = []
        agg_slh_list = []
        k_list = []
        ratio_list = []
        percentage_list = []
        n_evt_max = int(df["event_numb"].max())
        ca_logger.info("EVENT NUMBER IS: {}".format(n_evt_max))
        # for event in range(1, n_evt_max + 1):
        for event in range(1, 100):
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
                df_full, hs_df, pu_df = df_creator(df_temp1, lab_columns)
                n_trks = df_temp1["linked_type"].count()
                if n_trks > self.num_clusters:
                    n_good_merge += 1
                    n_tot_trk = df_full.shape[0]
                    n_full_tracks_mrg.append(n_tot_trk)
                    n_hs_trk = hs_df.shape[0]
                    n_hs_tracks_mrg.append(n_hs_trk)
                    n_pu_trk = pu_df.shape[0]
                    n_pu_tracks_mrg.append(n_pu_trk)
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
                    percentage = (n_hs_trk*100)/(n_pu_trk+n_hs_trk)
                    km_slh_list, agg_slh_list, k_list, ratio_list, percentage_list = self.cluster_exe(df_full,
                                                                                                      agg_slh_list,
                                                                                                      km_slh_list,
                                                                                                      k_list,
                                                                                                      n_hs_trk/n_pu_trk,
                                                                                                      ratio_list,
                                                                                                      percentage,
                                                                                                      percentage_list,
                                                                                                      plot_out_dir)
                    ca_logger.info("hs tracks percentage: {}%".format((n_hs_trk*100)/(n_pu_trk+n_hs_trk)))
                    ca_logger.info("ratio hs/pu: {}".format(n_hs_trk/n_pu_trk))
                else:
                    ca_logger.info(" bad vertex with {} tracks".format(n_trks))
                stop = time.time()
                ca_logger.info("Merged vertex cluster "
                               "done in: {} sec".format(stop - start))
        plot_scr_vs_ratio(km_slh_list, ratio_list,
                          self.output_path)
        plot_cn_vs_scr(k_list, km_slh_list, agg_slh_list,
                       self.output_path)
        plot_score(km_slh_list, agg_slh_list,
                   self.output_path)
        plot_scatter_score_km_vs_agg(km_slh_list, agg_slh_list,
                                     self.output_path)
        plot_scr_vs_percent(km_slh_list, percentage_list,
                            self.output_path)
        plot_linked_tracks_merge(n_hs_tracks_mrg,
                                 self.output_path,
                                 "hs_trk_mrg")
        plot_linked_tracks_merge(n_pu_tracks_mrg,
                                 self.output_path,
                                 "pu_trk_mrg")
        plot_linked_tracks_merge(n_full_tracks_mrg,
                                 self.output_path,
                                 "full_trk_mrg")
        plot_linked_tracks_merge_both(n_hs_tracks_mrg,
                                      n_pu_tracks_mrg,
                                      self.output_path, "both_hs_pu")
        ca_logger.info(
            "events: {}, mrg_vtxs: {}, good merge_vtxs: {}".format(n_event,
                                                                   n_mrg,
                                                                   n_good_merge)
        )
        ca_logger.info("{} empty events!".format(n_empty_mrg))

    def cluster_exe(self, df, agg_slh_scr_list,
                    km_slh_scr_list, k_vl_list,
                    hs_pu_ratio,
                    hs_pu_ratio_list,
                    percentage,
                    percentage_list,
                    plot_out_dir):
        """
        Loops over the cluster's number to retrieve
        the results using different sklearn methods.
        :param df: dataframe
        :param agg_slh_scr_list: agglomerative silhouette score list
        :param km_slh_scr_list: silhouette score list
        :param k_vl_list: cluster number list
        :param hs_pu_ratio: ratio hs/pu tracks
        :param hs_pu_ratio_list: ratio hs/pu tracks list
        :param percentage: hs tracks percentage
        :param percentage_list: hs tracks percentage list
        :param plot_out_dir: plot directory
        :return: None
        """
        ssd = []
        km_sl_scr = []
        km_better = 0
        agg_better = 0
        cluster_numbers = range(2, self.num_clusters)
        for cnt, k in enumerate(cluster_numbers):
            ssd_new, km_sl_scr = self.kmean_analyzer(k, df, ssd,
                                                     plot_out_dir)
            cls, cls_slh_scr = self.agg_clus_analyzer(k, df)
            km_slh_scr_list.append(km_sl_scr)
            agg_slh_scr_list.append(cls_slh_scr)
            k_vl_list.append(k)
            hs_pu_ratio_list.append(hs_pu_ratio)
            percentage_list.append(percentage)
            if km_sl_scr < cls_slh_scr:
                km_better += 1
            else:
                agg_better += 1
            # ca_logger.info(
            #     "kmean counter: {}, agg counter: {}".format(km_better,
            #                                                 agg_better)
            # )
            # plot_surface(k, km_sl_scr,
            #              hs_pu_ratio,
            #              plot_out_dir)
        plot_ssd(cluster_numbers, ssd_new,
                 km_sl_scr, plot_out_dir)
        return km_slh_scr_list, agg_slh_scr_list, k_vl_list, hs_pu_ratio_list, percentage_list

    def kmean_analyzer(self, k, df, ssd, plot_out_dir):
        """
        Run k-mean analysis to extract useful features
        for clustering.
        :param k: cluster number/s
        :param df: full df
        :param ssd: sum of squared distances
        :param plot_out_dir: output_plot_path
        :return: ssd, silhouette_score
        """
        km = KMeans(n_clusters=k, random_state=10)
        km = km.fit(df)
        ssd.append(km.inertia_)
        km_lab = km.fit_predict(df)
        maxim = max(km_lab)
        km_slh_scr = silhouette_score(df, km_lab)
        km_slh_vlu = silhouette_samples(df, km_lab)
        centroids = km.cluster_centers_
        # ca_logger.info(
        #     "Centroids: {},"
        #     "labels: {},"
        #     "max: {}".format(centroids, km_lab, maxim)
        # )

        # lab_size = np.arange(km_lab.size)
        # print("Lab size", lab_size.size)
        # for lab in km_lab:
        #     print(df["clus_d0"], )
        # print(km_lab.shape[0])
        # for i in range(km_lab.shape[0]):
        #     if km_lab == 0:
        #
        #         print("yes")
        #     elif df[i]

        # ca_logger.info("kmean centroid shape: {}".format(centroids.shape))
        centroids = np.array(centroids)
        # self.plt_sil(k, df, km, km_lab, km_slh_scr,
        #              km_slh_vlu, plot_out_dir)

        plot_centroid_d0z0(centroids, df,
                           plot_out_dir, k)
        plot_centroid_d0z0_focused(centroids, df,
                                   plot_out_dir, k)
        plot_colored_data(km_lab, maxim,
                          df, plot_out_dir)
        # ca_logger.info("KMEANS - "
        #                "clus: {}, "
        #                "sil_score: {}".format(k, km_slh_scr))
        return ssd, km_slh_scr

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
        # ca_logger.info("AGGCLS - "
        #                "clus: {}, "
        #                "sil_score: {}".format(k, cls_silh_score))
        return cls, cls_silh_score

    def plt_sil(self, k, df, km, km_lab, silh_score,
                silh_value, out_path):
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
                    c="white", alpha=1,
                    s=200, edgecolor='k')
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
