from utils.misc import get_logger
from sklearn.metrics import silhouette_score
from utils.plot_utils import plot_trks_mrg, plot_ssd, plot_mrg_evt, plot_truth_mrg
from sklearn.cluster import KMeans
import logging
import os

ca_logger = get_logger(__name__)
ca_logger.setLevel(logging.DEBUG)


class ClusterAnalyzer(object):
    def __init__(self, num_clusters, output_path):
        self.num_clusters = num_clusters
        self.output_path = output_path

    def plot_kmeans_elbow(self, df, plot_out_dir):
        sum_of_squared_distances = []
        silh_avg_list = []
        kvalue_list =[]
        cluster_numbers = range(2, self.num_clusters)
        for cnt, k in enumerate(cluster_numbers):
            km = KMeans(n_clusters=k)
            km = km.fit(df)
            sum_of_squared_distances.append(km.inertia_)
            cls_lab = km.fit_predict(df)
            ca_logger.info("clus: {}, lab.fit_predict ={}".format(k, cls_lab))
            silh_avg = silhouette_score(df, cls_lab)
            silh_avg_list.append(silh_avg)
            kvalue_list.append(k)
        plot_ssd(cluster_numbers, sum_of_squared_distances, silh_avg_list)

    def execute(self, df, df_columns, lab_columns):
        z = 0
        t = 0
        n_empty_mrg = 0
        n_tot_merg_evt = []
        n_tot_truth_mrg = []
        n_trks_merg = []
        n_evt_max = int(df["v_nEvent"].max())
        for event in range(1, n_evt_max):
            df_temp = df.loc[(df["v_nEvent"] == event), df_columns]
            n_merge_max = df_temp["v_nMergeEvent"].max()
            # check nan merge vertices
            if n_merge_max != n_merge_max:
                n_empty_mrg = n_empty_mrg + 1
                continue
            n_tot_merg_evt.append(n_merge_max)
            for mrg in range(1, int(n_merge_max)):
                evt_str = str(event)
                mrg_str = str(mrg)
                df_temp1 = df_temp.loc[(df_temp["v_nEvent"] == event) &
                                       (df_temp["v_nMergeEvent"] == mrg), df_columns]
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
                    event_dir = os.path.join(self.output_path, "TempDF_Plt", "evt_" + evt_str)
                    plot_out_dir = os.path.join(event_dir, 'MrgVtx'+mrg_str+'_nTruth'+n_row_str)
                    if not os.path.exists(plot_out_dir):
                        os.makedirs(plot_out_dir)
                    self.plot_kmeans_elbow(df_temp1, plot_out_dir)
                    # silhouette_ana(df_temp1)
                else:
                    z += 1
                    ca_logger.info(
                        "Event {}, Merge {}, nTrks {} BAD MERGED VERTEX!".format(
                            event, mrg, n_trks
                        )
                    )
                    continue
                if plot_out_dir is not None:
                    filename_path = os.path.join(plot_out_dir, "df_temp_Tracks" + str(n_trks) + ".csv")
                    df_temp1.to_csv(filename_path, header=True, index=False)
        plot_mrg_evt(n_tot_merg_evt, self.output_path)
        plot_truth_mrg(n_tot_truth_mrg, self.output_path)
        plot_trks_mrg(n_trks_merg, self.output_path)
        ca_logger.info("{} empty events!".format(n_empty_mrg))
        ca_logger.info("{} Merged vertices with less tracks than nearby clusters=[{}]".format(z, self.num_clusters))
        ca_logger.info("{} Merged vertices with more tracks than nearby clusters=[{}]".format(t, self.num_clusters))