from utils.misc import get_logger
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans, AgglomerativeClustering
from utils.plot_utils import plot_trks_mrg, plot_ssd,plot_mrg_evt, plot_truth_mrg
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

    def df_exe(self, df, df_columns, lab_columns):
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

    def df_clus_exe(self, df, df_columns, lab_columns):
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
        n_empty_mrg = 0
        hs = 0
        pu = 0
        n_mrg = 0
        n_event = 0
        n_tot_merg_evt = []
        n_evt_max = int(df["event_numb"].max())
        for event in range(1, 2):
            n_event += 1
            df_temp = df.loc[(df["event_numb"] == event), df_columns]
            print(df_temp)
            # n_merge_max = df_temp["merge_vtx"].max()
            # ca_logger.info("temp merge max: {} ".format(n_merge_max))
            # if n_merge_max != n_merge_max:
            #     n_empty_mrg = n_empty_mrg + 1
            #     continue
            # n_tot_merg_evt.append(n_merge_max)
            # for mrg in range(1, int(n_merge_max)):
            #     n_mrg += 1
            #     start = time.time()
            #     df_temp1 = df_temp.loc[(df_temp["event_numb"] == event) &
            #                            (df_temp["merge_vtx"] == mrg),
            #                            df_columns]
            #     ca_logger.info(
            #         "Event {}, Merge {}".format(event, mrg)
            #     )
            #
            #     # print(df_temp1)
            #     # print(df_temp1.shape)
            #     # print(df_temp1["check_type"])
            #     df_temp1.drop(lab_columns, axis=1, inplace=True)
            #
            #     # if df_temp1["check_type"].any() == 0:
            #     #     hs += 1
            #     # elif df_temp1["check_type"].any() == 1:
            #     #     pu += 1
            #
            #     stop = time.time()
            #     # ca_logger.info("Merged vertex cluster"
            #     #                "done in: {} sec".format(stop - start))
        ca_logger.info("HS vtx: {}, PU vtx: {}:".format(hs, pu))
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
        sum_of_squared_distances = []
        km_silh_score = []
        silh_score_list = []
        kvalue_list =[]
        cluster_numbers = range(2, self.num_clusters)
        for cnt, k in enumerate(cluster_numbers):
            ## KMeans
            km = KMeans(n_clusters=k, random_state=10)
            km = km.fit(df)
            sum_of_squared_distances.append(km.inertia_)
            km_lab = km.fit_predict(df)
            km_silh_score = silhouette_score(df, km_lab)
            km_silh_value = silhouette_samples(df, km_lab)
            silh_score_list.append(km_silh_score)
            kvalue_list.append(k)
            ## agg_clus
            cls = AgglomerativeClustering(n_clusters=k)
            cls = cls.fit(df)
            cls_lab = cls.labels_
            cls_silh_score = silhouette_score(df, cls_lab)
            cls_silh_value = silhouette_samples(df, cls_lab)
            ## plot silhouette
            self.plt_silh(k, df, km, km_lab, km_silh_score,
                          km_silh_value, plot_out_dir)
            # cls_plot_path = os.path.join(plot_out_dir, "cls_silhoue")
            # self.plt_silh(k, df, cls, cls_lab, cls_silh_score,
            #               cls_silh_value, cls_plot_path)
            ca_logger.info("KMEANS - "
                           "clus: {}, "
                           "lab prediction: {}".format(k, km_lab))
            # ca_logger.info("clus: {}, "
            #                "silhouette score: {}".format(k, km_silh_score))
            # ca_logger.info("clus: {},"
            #                "silhouette sample: {}".format(k, km_silh_value))
            ca_logger.info("AGG_CL - "
                           "clus: {}, "
                           "lab prediction: {}".format(k, cls_lab))
        plot_ssd(cluster_numbers, sum_of_squared_distances,
                 km_silh_score, plot_out_dir)

    def plt_silh(self, k, df, km, km_lab, silh_score, silh_value, out_path):
        """
        Plotting the silhouette paramente given by kmeans clustering.
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
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1.2])
        ax1.set_ylim([0, len(df["px"])+(self.num_clusters+1)*7])
        for i in range(k-1):
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
        ax2.scatter(df["px"], df["py"], marker='o',
                    s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
        centers = km.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for px")
        ax2.set_ylabel("Feature space for py ")
        plt.suptitle(("Silhouette analysis "
                      "with n_clusters = %d" % k),
                     fontsize=14, fontweight='bold')
        name = os.path.join(out_path, "silhouette_plt_"
                            +ncs+"_"+n_clusters_str+"_Clus.png")
        plt.savefig(name, dpi=300)
        plt.close()
        return None
