import matplotlib.pyplot as plt
import os


class MainDfPlotter(object):
    def __init__(self, full_df, out_path):
        self.full_df = full_df
        self.out_path = out_path

    def plot_pt(self):
        pt = self.full_df["clus_pt"]
        fig, ax = plt.subplots()
        ax.hist(pt, alpha=0.9, bins=20000)
        ax.set_title("pt")
        ax.set_xbound(0, (pt.values.mean() * 5))
        figname = os.path.join(self.out_path, "df_pt" + ".png")
        fig.savefig(figname, dpi=300)
        plt.close(fig)

    def plot_eta(self):
        eta = self.full_df["clus_eta"]
        fig, ax = plt.subplots()
        ax.hist(eta, alpha=0.9, bins=400)
        ax.set_title("eta")
        figname = os.path.join(self.out_path, "df_eta" + ".png")
        fig.savefig(figname, dpi=300)
        plt.close(fig)

    def plot_phi(self):
        phi = self.full_df["clus_phi"]
        fig, ax = plt.subplots()
        ax.hist(phi, alpha=0.9, bins=400)
        ax.set_title("phi")
        figname = os.path.join(self.out_path, "df_phi" + ".png")
        fig.savefig(figname, dpi=300)
        plt.close(fig)

    def plot_d0(self):
        d0 = self.full_df["clus_d0"]
        fig, ax = plt.subplots()
        ax.hist(d0, alpha=0.9, bins=400)
        ax.set_title("d0")
        ax.set_xbound(-2, 2)
        figname = os.path.join(self.out_path, "df_d0" + ".png")
        fig.savefig(figname, dpi=300)
        plt.close(fig)

    def plot_z0(self):
        z0 = self.full_df["clus_z0"]
        fig, ax = plt.subplots()
        ax.hist(z0, alpha=0.9, bins=400)
        ax.set_title("z0")
        ax.set_xlabel("[mm]")
        figname = os.path.join(self.out_path, "df_z0" + ".png")
        fig.savefig(figname, dpi=300)
        plt.close(fig)


    # def plot_px(self):
    #     px = self.full_df["px"]
    #     fig, ax = plt.subplots()
    #     ax.hist(px, alpha=0.9, bins=2000)
    #     ax.set_xbound(-5000, 5000)
    #     ax.set_title("px")
    #     figname = os.path.join(self.out_path, "df_px" + ".png")
    #     fig.savefig(figname, dpi=200)
    #     plt.close(fig)
    #
    # def plot_py(self):
    #     py = self.full_df["py"]
    #     fig, ax = plt.subplots()
    #     ax.hist(py, alpha=0.9, bins=2000)
    #     ax.set_xbound(-5000, 5000)
    #     ax.set_title("py")
    #     figname = os.path.join(self.out_path, "df_py" + ".png")
    #     fig.savefig(figname, dpi=200)
    #     plt.close(fig)
    #
    # def plot_pz(self):
    #     pz = self.full_df["pz"]
    #     fig, ax = plt.subplots()
    #     ax.hist(pz, alpha=0.9, bins=2000)
    #     ax.set_xbound(-2000000, 2000000)
    #     ax.set_title("pz")
    #     figname = os.path.join(self.out_path, "df_pz" + ".png")
    #     fig.savefig(figname, dpi=300)
    #     plt.close(fig)

    # def plot_rap(self):
    #     rap = self.full_df["rap"]
    #     fig, ax = plt.subplots()
    #     ax.hist(rap, alpha=0.9, bins=200)
    #     ax.set_xbound(-10, 10)
    #     ax.set_title("rapidity")
    #     figname = os.path.join(self.out_path, "df_rap" + ".png")
    #     fig.savefig(figname, dpi=300)
    #     plt.close(fig)
    #
    # def plot_eta_phi(self):
    #     eta = self.full_df["eta"]
    #     phi = self.full_df["phi"]
    #     plt.scatter(eta, phi, c="blue", alpha=0.9)
    #     plt.xlabel("eta")
    #     plt.ylabel("phi")
    #     figname = os.path.join(self.out_path, "df_eta_phi" + ".png")
    #     plt.savefig(figname, dpi=300)
    #     plt.close()
