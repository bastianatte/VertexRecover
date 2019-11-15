from classes.ClusterAnalyzer import ClusterAnalyzer
from classes.MainDfPlotter import MainDfPlotter
from utils.plot_utils import plot_correlation
import os
import sys
import argparse
import logging
from utils.misc import (
    load_data, print_df_info, load_config, get_logger
)
parser = argparse.ArgumentParser(
    description=("Run ClusterAnalysis using the parameters "
                 "specified in the configuration file")
)
parser.add_argument(
    '-c', '--conf',
    type=str,
    metavar='',
    required=True,
    help='Specify the path of the configuration file'
)
parser.add_argument(
    '-i', '--input',
    type=str,
    metavar='',
    required=True,
    help='Specify the path of the root file'
)
parser.add_argument(
    '-o', '--output',
    type=str,
    metavar='',
    required=True,
    help='Specify the output path'
)
parser.add_argument(
    '-p', '--plot',
    type=str,
    metavar='',
    required=True,
    help='type yes/not for plotting reason'
)
args = parser.parse_args()
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)
try:
    CONF = load_config(args.conf)
    KMEANS_CONF = CONF.get("kmeans")
    EXEC_CONF = CONF.get("execute")
except Exception as e:
    logger.error("{}. could not get conf settings".format(e.args[0]))
    sys.exit(1)
logger.info("Reading settings")
ROOTFILE_PATH = args.input
OUPUT_PATH = args.output
PLOT = args.plot
N_CLUSTERS = KMEANS_CONF["n_clusters"]
DF_COLUMNS = EXEC_CONF["df_columns"]
FEAT_COLUMNS = EXEC_CONF["feat"]
LAB_COLUMNS = EXEC_CONF["lab"]
DF_COLUMNS_LOC = EXEC_CONF["df_columns_loc"]
FEAT_COLUMNS_LOC = EXEC_CONF["feat_loc"]
LAB_COLUMNS_LOC = EXEC_CONF["lab_loc"]
PLOT_DIR = os.path.join(OUPUT_PATH, "Main_Plots")
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
logger.info("Initializing dataset")

feat, dataset = load_data(ROOTFILE_PATH, DF_COLUMNS, "Clustering")
feat_loc, dataset_loc = load_data(ROOTFILE_PATH, DF_COLUMNS, "Clustering_local")
print_df_info(dataset)
print_df_info(dataset_loc)

logger.info("Initializing plots")
if PLOT == "yes":
    """
    Plot full dataset variables
    """
    plot_correlation(dataset, PLOT_DIR)

    logger.info("...plot class activated, waiting... ")
    MPL = MainDfPlotter(dataset, PLOT_DIR)
    MPL.plot_pt()
    MPL.plot_eta()
    MPL.plot_phi()
    MPL.plot_d0()
    MPL.plot_z0()
    # MPL.plot_pz()
    # MPL.plot_px()
    # MPL.plot_py()
    # MPL.plot_rap()
    # MPL.plot_eta_phi()


logger.info("Initializing Analyzer")
logger.info("... clustering all tracks ...")
CA = ClusterAnalyzer(N_CLUSTERS, OUPUT_PATH)
CA.df_linked_tracks_exe(dataset, DF_COLUMNS, LAB_COLUMNS)
logger.info("... clustering all tracks DONE!!! ...")
logger.info("... clustering best 2 vertices ...")
CA = ClusterAnalyzer(N_CLUSTERS, OUPUT_PATH)
CA.df_linked_tracks_exe(dataset_loc, DF_COLUMNS, LAB_COLUMNS)
logger.info("... clustering best 2 vertices DONE!! ...")

# CA.df_exe(dataset, DF_COLUMNS, LAB_COLUMNS)  ### method for window matching