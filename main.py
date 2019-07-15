from classes.ClusterAnalyzer import ClusterAnalyzer
from classes.MainDfPlotter import MainDfPlotter
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
    help='Specify the ouput path'
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
N_CLUSTERS = KMEANS_CONF["n_clusters"]
DF_COLUMNS = EXEC_CONF["df_columns"]
LAB_COLUMNS = EXEC_CONF["lab"]
logger.info("Initializing Analyzer")
PLOT_DIR = os.path.join(OUPUT_PATH, "Main_Plots")
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
feat, dataset = load_data(ROOTFILE_PATH, DF_COLUMNS)
print_df_info(dataset)
MPL = MainDfPlotter(dataset, PLOT_DIR)
MPL.plot_pz()
MPL.plot_px()
MPL.plot_py()
MPL.plot_pt()
MPL.plot_eta()
MPL.plot_phi()
MPL.plot_rap()
MPL.plot_eta_phi()
CA = ClusterAnalyzer(N_CLUSTERS, OUPUT_PATH)
CA.execute(dataset, DF_COLUMNS, LAB_COLUMNS)
