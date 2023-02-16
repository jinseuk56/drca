# dimensionality reduction and unsupervised clustering for EELS-SI
# Jinseok Ryu (jinseuk56@gmail.com)
# https://doi.org/10.1016/j.ultramic.2021.113314


import numpy as np
import tifffile

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS

import ipywidgets as pyw
import time

try:
    import hyperspy.api as hys
except:
    print("Hyperspy cannot be imported")
    print("DM files not supported")
    
# create a customized colorbar
color_rep = ["black", "red", "green", "blue", "orange", "purple", "yellow", "lime", 
             "cyan", "magenta", "lightgray", "peru", "springgreen", "deepskyblue", 
             "hotpink", "darkgray"]

rgb_rep = {"black":[1,1,1,1], "red":[1,0,0,1], "green":[0,1,0,1], "blue":[0,0,1,1], "orange":[1,0.5,0,1], "purple":[1,0,1,1],
           "yellow":[1,1,0,1], "lime":[0,1,0.5,1], "cyan":[0,1,1,1]}

custom_cmap = mcolors.ListedColormap(color_rep)
bounds = np.arange(-1, len(color_rep))
norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(color_rep))
sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])

cm_rep = ["gray", "Reds", "Greens", "Blues", "Oranges", "Purples"]  
    
class DR_assisted_CA():
    def __init__(self, adr, dat_dim, dat_unit, cr_range=None, dat_scale=1, rescale=True, DM_file=True):
        self.file_adr = adr
        self.num_img = len(adr)
        self.dat_dim = dat_dim
        if dat_dim == 4:
            cr_range = None
        self.dat_unit = dat_unit
        self.cr_range = cr_range
        
        if cr_range:
            self.dat_dim_range = np.arange(cr_range[0], cr_range[1], cr_range[2]) * dat_scale
            self.num_dim = len(self.dat_dim_range)
        
        if dat_dim == 3:
            self.data_storage, self.data_shape = data_load_3d(adr, cr_range, rescale, DM_file)
        
        else:
            self.data_storage, self.data_shape = data_load_4d(adr, rescale)
            
        self.original_data_shape = self.data_shape.copy()
             
    def binning(self, bin_y, bin_x, str_y, str_x, offset=0, rescale_0to1=True):
        dataset = []
        data_shape_new = []
        
        for img in self.data_storage:
            print(img.shape)
            processed = binning_SI(img, bin_y, bin_x, str_y, str_x, offset, self.num_dim, rescale_0to1) # include the step for re-scaling the actual input
            print(processed.shape)
            data_shape_new.append(processed.shape)
            dataset.append(processed)

        data_shape_new = np.asarray(data_shape_new)
        print(data_shape_new)
        
        self.data_storage = dataset
        self.data_shape = data_shape_new
        
    def find_center(self, cbox_edge, center_remove, result_visual=True, log_scale=True):
        if self.dat_dim != 4:
            print("data dimension error")
            return
        
        self.center_pos = []
        
        for i in range(self.num_img):
            mean_dp = np.mean(self.data_storage[i], axis=(0, 1))
            cbox_outy = int(mean_dp.shape[0]/2 - cbox_edge/2)
            cbox_outx = int(mean_dp.shape[1]/2 - cbox_edge/2)
            center_box = mean_dp[cbox_outy:-cbox_outy, cbox_outx:-cbox_outx]
            Y, X = np.indices(center_box.shape)
            com_y = np.sum(center_box * Y) / np.sum(center_box)
            com_x = np.sum(center_box * X) / np.sum(center_box)
            c_pos = [np.around(com_y+cbox_outy), np.around(com_x+cbox_outx)]
            self.center_pos.append(c_pos)
        print(self.center_pos)
        
        if result_visual:
            np.seterr(divide='ignore')
            for i in range(self.num_img):
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                if log_scale:
                    ax.imshow(np.log(np.mean(self.data_storage[i], axis=(0, 1))), cmap="viridis")
                else:
                    ax.imshow(np.mean(self.data_storage[i], axis=(0, 1)), cmap="viridis")
                ax.scatter(self.center_pos[i][1], self.center_pos[i][0], c="r", s=10)
                ax.axis("off")
                plt.show()
        
        if center_remove != 0:
            self.center_removed = True
            data_cr = []
            for i in range(self.num_img):
                ri = radial_indices(self.data_storage[i].shape[2:], [center_remove, 100], center=self.center_pos[i])
                data_cr.append(np.multiply(self.data_storage[i], ri))
                
            self.data_storage = data_cr
            
            if result_visual:
                for i in range(self.num_img):
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    if log_scale:
                        ax.imshow(np.log(np.mean(self.data_storage[i], axis=(0, 1))), cmap="viridis")
                    else:
                        ax.imshow(np.mean(self.data_storage[i], axis=(0, 1)), cmap="viridis")
                    ax.scatter(self.center_pos[i][1], self.center_pos[i][0], c="r", s=10)
                    ax.axis("off")
                    plt.show()
                    
    def make_input(self, min_val=0.0, max_normalize=True, rescale_0to1=False, log_scale=False, radial_flat=True, w_size=0, radial_range=None):

        dataset_flat = []
        if self.dat_dim == 3:
            for i in range(self.num_img):
                dataset_flat.extend(self.data_storage[i].clip(min=min_val).reshape(-1, self.num_dim).tolist())

            dataset_flat = np.asarray(dataset_flat)
            print(dataset_flat.shape)
            
            
        if self.dat_dim == 4:
            self.radial_flat = radial_flat
            self.w_size = w_size
            self.radial_range = radial_range
            
            dataset = []
            
            if radial_flat:
                self.k_indx = []
                self.k_indy = []
                self.a_ind = []

                for r in range(radial_range[0], radial_range[1], radial_range[2]):
                    tmp_k, tmp_a = indices_at_r((radial_range[1]*2, radial_range[1]*2), r, (radial_range[1], radial_range[1]))
                    self.k_indx.extend(tmp_k[0].tolist())
                    self.k_indy.extend(tmp_k[1].tolist())
                    self.a_ind.extend(tmp_a.tolist())

                self.s_length = len(self.k_indx)

                for i in range(self.num_img):
                    flattened = circle_flatten(self.data_storage[i], radial_range, self.center_pos[i])

                    tmp = np.zeros((radial_range[1]*2, radial_range[1]*2))
                    tmp[self.k_indy, self.k_indx] = np.sum(flattened, axis=(0, 1))

                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    ax.imshow(tmp, cmap="viridis")
                    ax.axis("off")
                    fig.tight_layout()
                    plt.show()

                    dataset.append(flattened) 
                
            else:
                for i in range(self.num_img):
                    flattened = flattening(self.data_storage[i], flat_option="box", crop_dist=w_size, c_pos=self.center_pos[i])
                    dataset.append(flattened)

                self.s_length = (w_size*2)**2
                
            for i in range(self.num_img):
                print(dataset[i].shape)
                dataset_flat.extend(dataset[i].reshape(-1, self.s_length))
            dataset_flat = np.asarray(dataset_flat)
            print(dataset_flat.shape)
            
        if log_scale:
            dataset_flat[np.where(dataset_flat==0.0)] = 1.0
            dataset_flat = np.log(dataset_flat)
            print(np.min(dataset_flat), np.max(dataset_flat))
            
        if max_normalize:
            dataset_flat = dataset_flat / np.max(dataset_flat, axis=1)[:, np.newaxis]
            print(np.min(dataset_flat), np.max(dataset_flat))
            
        if rescale_0to1:
            for i in range(len(dataset_flat)):
                dataset_flat[i] = zero_one_rescale(dataset_flat[i])
            
        dataset_flat = dataset_flat.clip(min=min_val)
        print(np.min(dataset_flat), np.max(dataset_flat))
        total_num = len(dataset_flat)
        self.dataset_flat = dataset_flat
        self.ri = np.random.choice(total_num, total_num, replace=False)

        self.dataset_input = dataset_flat[self.ri]
        self.dataset_input = self.dataset_input.astype(np.float32)
        
    def ini_DR(self, method="nmf", num_comp=5, result_visual=True):
        self.DR_num_comp = num_comp
        if method=="nmf":
            self.DR = NMF(n_components=num_comp, init="nndsvda", solver="mu", max_iter=2000, verbose=True)
            
            self.DR_coeffs = self.DR.fit_transform(self.dataset_input)
            self.DR_comp_vectors = self.DR.components_
            
        elif method=="pca":
            self.DR = PCA(n_components=num_comp, whiten=False, 
                     random_state=np.random.randint(100), svd_solver="auto")
            
            self.DR_coeffs = self.DR.fit_transform(self.dataset_input)
            self.DR_comp_vectors = self.DR.components_
        
        elif method=="cae":
            print("in preparation...")
            return
            
        else:
            print(method+" not supported")
            return
        
        coeffs = np.zeros_like(self.DR_coeffs)
        coeffs[self.ri] = self.DR_coeffs.copy()
        self.DR_coeffs = coeffs
        self.coeffs_reshape = reshape_coeff(self.DR_coeffs, self.data_shape)
        
        if result_visual:
            if self.dat_dim == 3:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4)) # all loading vectors
                for i in range(self.DR_num_comp):
                    ax.plot(self.dat_dim_range, self.DR_comp_vectors[i], "-", c=color_rep[i+1], label="loading vector %d"%(i+1))
                ax.grid()
                ax.legend(fontsize="large")
                ax.set_xlabel(self.dat_unit, fontsize=10)
                ax.tick_params(axis="x", labelsize=10)
                ax.axes.get_yaxis().set_visible(False)

                fig.tight_layout()
                plt.show()
                
            elif self.dat_dim == 4:
                if self.radial_flat:
                    for i in range(self.DR_num_comp):
                        tmp = np.zeros((self.radial_range[1]*2, self.radial_range[1]*2))
                        tmp[self.k_indy, self.k_indx] = self.DR_comp_vectors[i]

                        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                        ax.imshow(tmp, cmap="viridis")
                        ax.axis("off")
                        fig.tight_layout()
                        plt.show()

                else:
                    for i in range(self.DR_num_comp):
                        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                        ax.imshow(self.DR_comp_vectors[i].reshape((self.w_size*2, self.w_size*2)), cmap="viridis")
                        ax.axis("off")
                        fig.tight_layout()
                        plt.show()
            
            min_val = np.min(coeffs)
            max_val = np.max(coeffs)
            if self.num_img != 1:
                for i in range(self.DR_num_comp):
                    fig, ax = plt.subplots(1, self.num_img, figsize=(5*self.num_img, 5))
                    for j in range(self.num_img):
                        tmp = ax[j].imshow(self.coeffs_reshape[j][:, :, i], vmin=min_val, vmax=max_val, cmap="viridis")
                        ax[j].set_title("loading vector %d map"%(i+1), fontsize=10)
                        ax[j].axis("off")
                        fig.colorbar(tmp, cax=fig.add_axes([0.92, 0.15, 0.04, 0.7]))
                    plt.show()
            else:            
                for i in range(self.DR_num_comp):
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5*self.num_img))
                    tmp = ax.imshow(self.coeffs_reshape[0][:, :, i], vmin=min_val, vmax=max_val, cmap="viridis")
                    ax.set_title("loading vector %d map"%(i+1), fontsize=10)
                    ax.axis("off")
                    fig.colorbar(tmp, cax=fig.add_axes([0.92, 0.15, 0.04, 0.7]))
                    plt.show()
                    
    def aug_DR(self, num_comp, method="tsne", perplex=[50]):
        start = time.time()
        embeddings = []
        self.num_comp_vis = num_comp # number of dimensions of final data before clustering
        
        if method=="tsne":
            for order, p in enumerate(perplex):
                tmp_tsne = TSNE(n_components=num_comp, perplexity=p, early_exaggeration=5.0, learning_rate=300.0, 
                            init="random", n_iter=1000, verbose=0)
                tmp_tsne.fit_transform(self.DR_coeffs)
                embeddings.append(tmp_tsne.embedding_)
                print("%d perplexity %.1f finished"%(order+1, p))
                print("%.2f min have passed"%((time.time()-start)/60))
                
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.scatter(tmp_tsne.embedding_[:, 0], tmp_tsne.embedding_[:, 1], s=1, c="black")
                ax.set_title("perplexity %.1f"%p)
                fig.tight_layout()
                plt.show()

        self.embeddings = embeddings
        
    def prepare_clustering(self, sel_ind, quick_visual=True):
        
        self.aDR_coeffs = self.embeddings[sel_ind-1]
        
        comp_axes = np.arange(self.num_comp_vis)
        
        if len(comp_axes) == 2:
            self.X = np.stack((self.aDR_coeffs[:, comp_axes[0]], self.aDR_coeffs[:, comp_axes[1]]), axis=1)
            print(self.X.shape)

        elif len(comp_axes) == 3:
            self.X = np.stack((self.aDR_coeffs[:, comp_axes[0]], self.aDR_coeffs[:, comp_axes[1]], self.aDR_coeffs[:, comp_axes[2]]), axis=1)
            print(self.X.shape)
            
        if quick_visual:    
            center = np.mean(self.X, axis=0)
            dist_from_center = np.sum((self.X-center)**2, axis=1)
            max_radius = np.sqrt(np.max(dist_from_center))

            X_shift = self.X - center

            r_point = [0, -2*max_radius]
            g_point = [3**(1/2)*max_radius, max_radius]
            b_point = [-3**(1/2)*max_radius, max_radius]

            red = np.sqrt(np.sum((X_shift-r_point)**2, axis=1))
            red = red / np.max(red)
            green = np.sqrt(np.sum((X_shift-g_point)**2, axis=1))
            green = green / np.max(green)
            blue = np.sqrt(np.sum((X_shift-b_point)**2, axis=1))
            blue = blue / np.max(blue)
            alpha = np.ones_like(red)
            point_colors = np.stack((red, green, blue, alpha), axis=1)

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.scatter(X_shift[:, 0], X_shift[:, 1], s=3, c=point_colors)
            ax.scatter(center[0], center[1], s=5, c="red")
            fig.tight_layout()
            plt.show()

            color_reshape = reshape_coeff(point_colors, self.data_shape)
            red_reshape = reshape_coeff(np.expand_dims(red, axis=1), self.data_shape)
            green_reshape = reshape_coeff(np.expand_dims(green, axis=1), self.data_shape)
            blue_reshape = reshape_coeff(np.expand_dims(blue, axis=1), self.data_shape)

            if self.num_img != 1:
                fig, ax = plt.subplots(4, self.num_img, figsize=(5*self.num_img, 5*4))
                for j in range(self.num_img):
                    ax[0][j].imshow(color_reshape[j])
                    ax[0][j].axis("off")
                    ax[1][j].imshow(red_reshape[j], cmap="Reds")
                    ax[1][j].axis("off")
                    ax[2][j].imshow(green_reshape[j], cmap="Greens")
                    ax[2][j].axis("off")
                    ax[3][j].imshow(blue_reshape[j], cmap="Blues")
                    ax[3][j].axis("off")
                plt.show()
            else:            
                fig, ax = plt.subplots(4, 1, figsize=(5*self.num_img, 5*4))
                ax[0].imshow(color_reshape[0])
                ax[0].axis("off")
                ax[1].imshow(red_reshape[0], cmap="Reds")
                ax[1].axis("off")    
                ax[2].imshow(green_reshape[0], cmap="Greens")
                ax[2].axis("off")    
                ax[3].imshow(blue_reshape[0], cmap="Blues")
                ax[3].axis("off")    
                plt.show()

    def cluster_analysis(self, method="optics"):
        
        self.fig = plt.figure(figsize=(10, 8))
        G = gridspec.GridSpec(2, 4)
        self.ax1 = plt.subplot(G[0, :])

        if self.num_comp_vis == 3:
            self.ax2 = plt.subplot(G[1, :2], projection="3d")

        elif self.num_comp_vis == 2:
            self.ax2 = plt.subplot(G[1, :2])

        self.ax3 = plt.subplot(G[1, 2:])

        self.optics_before = [-1, -1, -1]
        
        st = {"description_width": "initial"}
        msample_wg = pyw.FloatText(value=0.05, description="min. # of samples in a neighborhood", style=st)
        steep_wg = pyw.FloatText(value=0.001, description="min. steepness", style=st)
        msize_wg = pyw.FloatText(value=0.05, description="min. # of samples in a cluster", style=st)
        img_wg = pyw.Select(options=np.arange(self.num_img)+1, value=1, description="image selection", style=st)

        self.clustering_widgets = pyw.interact(self.clustering, msample=msample_wg, steep=steep_wg, msize=msize_wg,  img_sel=img_wg)
        plt.show()

    def clustering(self, msample, steep, msize, img_sel):
        start = time.time()
        if msample <= 0:
            print("'min_sample' must be larger than 0")
            return

        if steep <= 0:
            print("'steepness' must be larger than 0")
            return

        if msize <= 0:
            print("'min_cluster_size' must be larger than 0")
            return

        optics_check = [msample, steep, msize]

        if self.optics_before != optics_check:
            self.ax1.cla()

            print("optics activated")
            clust = OPTICS(min_samples=msample, xi=steep, min_cluster_size=msize).fit(self.X)

            space = np.arange(len(self.X))
            reachability = clust.reachability_[clust.ordering_]
            labels = clust.labels_[clust.ordering_]
            print("activated?")
        
            for klass, color in zip(range(0, len(color_rep)), color_rep[1:]):
                Xk = space[labels == klass]
                Rk = reachability[labels == klass]
                self.ax1.plot(Xk, Rk, color, alpha=0.3)

            self.ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
            self.ax1.set_ylabel('Reachability-distance')
            self.ax1.set_title('Reachability-Plot')
            #self.ax1.grid()

            self.ax2.cla()
            labels = clust.labels_
            self.labels = labels
            if self.num_comp_vis == 3:
                for klass, color in zip(range(0, len(color_rep)), color_rep[1:]):
                    Xo = self.X[labels == klass]
                    self.ax2.scatter(Xo[:, 0], Xo[:, 1], Xo[:, 2], color=color, alpha=0.3, marker='.')
                self.ax2.plot(self.X[labels == -1, 0], self.X[labels == -1, 1], self.X[labels == -1, 2], 'k+', alpha=0.1)
                self.ax2.set_title('Automatic Clustering\nOPTICS(# of clusters=%d)\n(%f, %f, %f)'%(len(np.unique(labels)), msample, steep, msize))

            elif self.num_comp_vis == 2:
                for klass, color in zip(range(0, len(color_rep)), color_rep[1:]):
                    Xo = self.X[labels == klass]
                    self.ax2.scatter(Xo[:, 0], Xo[:, 1], color=color, alpha=0.3, marker='.')
                self.ax2.plot(self.X[labels == -1, 0], self.X[labels == -1, 1], 'k+', alpha=0.1)
                self.ax2.set_title('Automatic Clustering\nOPTICS(# of clusters=%d)\n(%f, %f, %f)'%(len(np.unique(labels)), msample, steep, msize))

        self.ax3.cla()
        
        label_reshape, _, _ = label_arrangement(self.labels, self.data_shape)

        self.ax3.imshow(label_reshape[img_sel-1], cmap=custom_cmap, norm=norm)
        self.ax3.set_title("image %d"%(img_sel), fontsize=10)
        self.ax3.axis("off")

        self.fig.tight_layout()

        del self.optics_before[:]
        for i in range(len(optics_check)):
            self.optics_before.append(optics_check[i])
        print("minimum number of samples in a neighborhood: %f"%msample)
        print("minimum steepness: %f"%steep)
        print("minumum number of samples in a cluster: %f"%msize)
        print("%.2f min have passed"%((time.time()-start)/60))

        return self.labels
        
    def clustering_result(self, tf_map=False, normalize='max', log_scale=True):
        
        self.clustering_widgets.widget.close_all()
        label_selected = self.clustering_widgets.widget.result
        label_sort = np.unique(label_selected)
        label_reshape, selected, hist = label_arrangement(label_selected, self.data_shape)
        num_label = len(label_sort)
        print(label_sort) # label "-1" -> not a cluster
        print(hist) # number of data points in each cluster
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for klass, color in zip(range(0, len(color_rep)), color_rep[1:]):
            Xo = self.X[label_selected == klass]
            ax.scatter(Xo[:, 0], Xo[:, 1], color=color, alpha=0.3, marker='.')
        ax.plot(self.X[label_selected == -1, 0], self.X[label_selected == -1, 1], 'k+', alpha=0.1)
        #ax.axis("off")
        fig.tight_layout()
        plt.show()
        
        # clustering result - spatial distribution of each cluster
        row_n = 1
        col_n = self.num_img

        fig, ax = plt.subplots(row_n, col_n, figsize=(7, 10))
        if self.num_img != 1:
            for i, axs in enumerate(ax.flat):
                axs.imshow(label_reshape[i], cmap=custom_cmap, norm=norm)
                axs.set_title("image %d"%(i+1), fontsize=10)
                axs.axis("off")

        else:
            ax.imshow(label_reshape[0], cmap=custom_cmap, norm=norm)
            ax.set_title("image %d"%(1), fontsize=10)
            ax.axis("off")

        fig.tight_layout()
        plt.show()
        
        if tf_map:
            if self.num_img != 1:
                for i in range(num_label):
                    fig, ax = plt.subplots(1, self.num_img, figsize=(3*self.num_img, 3))
                    for j in range(self.num_img):
                        ax[j].imshow(selected[i][j], cmap="afmhot")
                        ax[j].set_title("label %d map"%(label_sort[i]+1), fontsize=10)
                        ax[j].axis("off")
                        fig.tight_layout()
                    plt.show()
            else:            
                for i in range(num_label):
                    fig, ax = plt.subplots(1, 1, figsize=(3*self.num_img, 3))
                    tmp = ax.imshow(selected[i][0], cmap="afmhot")
                    ax.axis("off")
                    fig.tight_layout()
                    plt.show()
        
        # clustering result - representative spectra (cropped)
        # average all of the spectra in each cluster
        
        if self.dat_dim == 3:
            lines = np.zeros((num_label, self.num_dim))

            for i in range(num_label):
                ind = np.where(label_selected == label_sort[i])
                print("number of pixels in the label %d cluster: %d"%(label_sort[i], hist[i]))
                lines[i] = np.mean(self.dataset_flat[ind], axis=0)

            fig, ax = plt.subplots(1, 2, figsize=(15, 8))

            # normalize representative spectra for comparison
            if normalize == 'max':
                denominator = np.max(lines, axis=1)
            elif normalize == 'min':
                denominator = np.min(lines, axis=1)
            lines = lines / denominator[:, np.newaxis]

            if -1 in label_sort:
                for i in range(1, num_label):
                    ax[0].plot(self.dat_dim_range, (lines[i]), label="cluster %d"%(i), c=color_rep[i])
                    ax[1].plot(self.dat_dim_range, (lines[i]+(i-1)*0.25), label="cluster %d"%(i), c=color_rep[i])

            else:
                for i in range(0, num_label):
                    ax[0].plot(self.dat_dim_range, (lines[i]), label="cluster %d"%(i+1), c=color_rep[i+1])
                    ax[1].plot(self.dat_dim_range, (lines[i]+i*0.25), label="cluster %d"%(i+1), c=color_rep[i+1])

            ax[0].legend(fontsize="x-large")
            ax[0].set_xlabel(self.dat_unit)
            ax[0].set_facecolor("lightgray")

            ax[1].set_xlabel(self.dat_unit)
            ax[1].set_facecolor("lightgray")

            fig.tight_layout()
            plt.show()
            
        elif self.dat_dim == 4:
            lines = np.zeros((num_label, self.s_length))

            for i in range(num_label):
                ind = np.where(label_selected == label_sort[i])
                print("number of pixels in the label %d cluster: %d"%(label_sort[i], hist[i]))
                lines[i] = np.mean(self.dataset_flat[ind], axis=0)

            row_n = num_label
            col_n = 1
            fig, ax = plt.subplots(row_n, col_n, figsize=(7, 50))


            if self.radial_flat:
                for i, la in enumerate(label_sort):
                    tmp = np.zeros((self.radial_range[1]*2, self.radial_range[1]*2))
                    tmp[self.k_indy, self.k_indx] = lines[i]

                    ax[i].imshow(tmp, cmap="viridis")
                    ax[i].axis("off")

                    if la == -1:
                        ax[i].set_title("not cluster")
                    else:
                        ax[i].set_title("cluster %d"%(la+1)) 

            else:
                for i, la in enumerate(label_sort):
                    if log_scale:
                        ax[i].imshow(np.log(lines[i].reshape((self.w_size*2, self.w_size*2))), cmap="viridis") # log scale - optional
                    else:
                        ax[i].imshow(lines[i].reshape((self.w_size*2, self.w_size*2)), cmap="viridis")
                    ax[i].axis("off")
                    if la == -1:
                        ax[i].set_title("not cluster")
                    else:
                        ax[i].set_title("cluster %d"%(la+1))

            fig.tight_layout()
            plt.show()

#####################################################
# functions #
#####################################################
def data_load_3d(adr, crop=None, rescale=True, DM_file=True):
    """
    load a spectrum image
    """
    storage = []
    shape = []
    for i, ad in enumerate(adr):
        if DM_file:
            if crop:
                temp = hys.load(ad)
                print(temp.axes_manager[2])
                temp = temp.isig[crop[0]:crop[1]]
                temp = temp.data
                if rescale:
                    temp = temp/np.max(temp)
                temp = temp.clip(min=0.0)
                print(temp.shape)
                
            else:
                temp = hys.load(ad).data
                if rescale:
                    temp = temp/np.max(temp)
                temp = temp.clip(min=0.0)
                print(temp.shape)
        
        else:
            if crop:
                temp = tifffile.imread(ad)
                temp = temp[:, :, crop[0]:crop[1]]
                temp = temp.data
                if rescale:
                    temp = temp/np.max(temp)
                temp = temp.clip(min=0.0)
                print(temp.shape)
                
            else:
                temp = tifffile.imread(ad)
                if rescale:
                    temp = temp/np.max(temp)
                temp = temp.clip(min=0.0)
                print(temp.shape)                
                
        shape.append(temp.shape)
        storage.append(temp)       
    
    shape = np.asarray(shape)
    return storage, shape


def data_load_4d(adr, rescale=False):
    storage = []
    shape = []   
    for i, ad in enumerate(adr):
        tmp = tifffile.imread(ad)
        if rescale:
            tmp = tmp / np.max(tmp)
        tmp = tmp.clip(min=0.0)
        print(tmp.shape)
        shape.append(list(tmp.shape[:2]))
        storage.append(tmp)
    
    shape = np.asarray(shape)
    return storage, shape

def zero_one_rescale(spectrum):
    """
    normalize one spectrum from 0.0 to 1.0
    """
    spectrum = spectrum.clip(min=0.0)
    min_val = np.min(spectrum)
    
    rescaled = spectrum - min_val
    
    if np.max(rescaled) != 0:
        rescaled = rescaled / np.max(rescaled)
    
    return rescaled

def binning_SI(si, bin_y, bin_x, str_y, str_x, offset, depth, rescale=True):
    """
    re-bin a spectrum image
    """
    si = np.asarray(si)
    rows = range(0, si.shape[0]-bin_y+1, str_y)
    cols = range(0, si.shape[1]-bin_x+1, str_x)
    new_shape = (len(rows), len(cols))
    
    binned = []
    for i in rows:
        for j in cols:
            temp_sum = np.mean(si[i:i+bin_y, j:j+bin_x, offset:(offset+depth)], axis=(0, 1))
            if rescale:
                binned.append(zero_one_rescale(temp_sum))
            else:
                binned.append(temp_sum)
            
    binned = np.asarray(binned).reshape(new_shape[0], new_shape[1], depth)
    
    return binned


def radial_indices(shape, radial_range, center=None):
    y, x = np.indices(shape)
    if not center:
        center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
    
    r = np.hypot(y - center[0], x - center[1])
    ri = np.ones(r.shape)
    
    if len(np.unique(radial_range)) > 1:
        ri[np.where(r <= radial_range[0])] = 0
        ri[np.where(r > radial_range[1])] = 0
        
    else:
        r = np.round(r)
        ri[np.where(r != round(radial_range[0]))] = 0
    
    return ri

def flattening(fdata, flat_option="box", crop_dist=None, c_pos=None):
    
    fdata_shape = fdata.shape
    if flat_option == "box":
        if crop_dist:     
            box_size = np.array([crop_dist, crop_dist])
        
            h_si = np.floor(c_pos[0]-box_size[0]).astype(int)
            h_fi = np.ceil(c_pos[0]+box_size[0]).astype(int)
            w_si = np.floor(c_pos[1]-box_size[1]).astype(int)
            w_fi = np.ceil(c_pos[1]+box_size[1]).astype(int)

            tmp = fdata[:, :, h_si:h_fi, w_si:w_fi]
            
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(np.log(np.mean(tmp, axis=(0, 1))), cmap="viridis")
            ax.axis("off")
            plt.show()
            
            tmp = tmp.reshape(fdata_shape[0], fdata_shape[1], -1)
            return tmp

        else:
            tmp = fdata.reshape(fdata_shape[0], fdata_shape[1], -1)
            return tmp

        
    elif flat_option == "radial":
        if len(crop_dist) != 3:
            print("Warning! 'crop_dist' must be a list containing 3 elements")
            
        tmp = circle_flatten(fdata, crop_dist, c_pos)
        return tmp
        
    else:
        print("Warning! Wrong option ('flat_option')")
        return
    
def circle_flatten(f_stack, radial_range, c_pos):
    k_indx = []
    k_indy = []
    
    for r in range(radial_range[0], radial_range[1], radial_range[2]):
        tmp_k, tmp_a = indices_at_r(f_stack.shape[2:], r, c_pos)
        k_indx.extend(tmp_k[0].tolist())
        k_indy.extend(tmp_k[1].tolist())
    
    k_indx = np.asarray(k_indx)
    k_indy = np.asarray(k_indy)
    flat_data = f_stack[:, :, k_indy, k_indx]
    
    return flat_data

def indices_at_r(shape, radius, center=None):
    y, x = np.indices(shape)
    if not center:
        center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
    r = np.hypot(y - center[0], x - center[1])
    r = np.around(r)
    
    ri = np.where(r == radius)
    
    angle_arr = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            angle_arr[i, j] = np.angle(complex(x[i, j]-center[1], y[i, j]-center[0]), deg=True)
            
    angle_arr = angle_arr + 180
    angle_arr = np.around(angle_arr)
    
    ai = np.argsort(angle_arr[ri])
    r_sort = (ri[1][ai], ri[0][ai])
    a_sort = np.sort(angle_arr[ri])
        
    return r_sort, a_sort

def reshape_coeff(coeffs, new_shape):
    """
    reshape a coefficient matrix to restore the original scanning shapes.
    """
    coeff_reshape = []
    for i in range(len(new_shape)):
        temp = coeffs[:int(new_shape[i, 0]*new_shape[i, 1]), :]
        coeffs = np.delete(coeffs, range(int(new_shape[i, 0]*new_shape[i, 1])), axis=0)
        temp = np.reshape(temp, (new_shape[i, 0], new_shape[i, 1], -1))
        #print(temp.shape)
        coeff_reshape.append(temp)
        
    return coeff_reshape

def label_arrangement(label_arr, new_shape):
    """
    reshape a clustering result obtained by performing OPTICS
    """
    label_sort = np.unique(label_arr)
    #print(label_sort)
    num_label = len(label_sort)
    hist, edge = np.histogram(label_arr, bins=num_label)
    #print(hist)
    label_reshape = reshape_coeff(label_arr.reshape(-1, 1), new_shape)
    
    for i in range(len(label_reshape)):
        label_reshape[i] = np.squeeze(label_reshape[i])
        
    selected = []
    for i in range(num_label):
        temp = []
        for j in range(len(label_reshape)):
            img_temp = np.zeros_like(label_reshape[j])
            img_temp[np.where(label_reshape[j] == label_sort[i])] = 1.0
            temp.append(img_temp)
        selected.append(temp)    
        
    return label_reshape, selected, hist