import streamlit as st
import os
import matplotlib.pyplot as plt
import numpy as np
import contextlib
import sys
from drca import drca
import tifffile

class StreamlitConsole:
    def __init__(self, container, step_key):
        self.container = container
        self.step_key = step_key
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        self.container.code(self.buffer, language="text")
        st.session_state.outputs[self.step_key]['log'] = self.buffer

    def flush(self): pass

def run_step(step_key, log_container, success_msg, func, *args, **kwargs):
    st.session_state.outputs[step_key] = {'log': '', 'figs': []}
    
    original_show = plt.show
    plt.show = lambda: None
    plt.close('all') 
    
    try:
        with contextlib.redirect_stdout(StreamlitConsole(log_container, step_key)):
            func(*args, **kwargs)
            
        fignums = plt.get_fignums()
        figs = [plt.figure(n) for n in fignums]
        st.session_state.outputs[step_key]['figs'] = figs
        st.success(success_msg)
    except Exception as e:
        st.error(f"Execution Error: {str(e)}")
    finally:
        plt.show = original_show

def render_step_results(step_key, log_container, fig_container):
    if st.session_state.outputs[step_key]['log']:
        log_container.code(st.session_state.outputs[step_key]['log'], language="text")
    if st.session_state.outputs[step_key]['figs']:
        with fig_container:
            for fig in st.session_state.outputs[step_key]['figs']:
                st.pyplot(fig)

st.set_page_config(page_title="DRCA", layout="wide")
st.title("DRCA Interface")

if 'model' not in st.session_state:
    st.session_state.model = None

if 'outputs' not in st.session_state:
    st.session_state.outputs = {
        key: {'log': '', 'figs': []} for key in [
            'init', 'bin', 'center', 'input', 'ini_dr', 'aug_dr', 'prep_clust', 'run_clust', 'vis'
        ]
    }

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Initialization", "2. Linear DR", "3. Nonlinear DR", "4. Clustering", "5. Visualization"
])

with tab1:
    st.header("Step 1: Load Data")
    file_paths_input = st.text_area("Enter absolute file paths (one path per line):", height=80, help="Paste the full system paths to your .dm3, .dm4, or .tif files here.")
    col1, col2 = st.columns(2)
    with col1:
        dat_dim = st.number_input("Data Dimension", value=3, min_value=2, max_value=4, help="Dimensions of hyperspectral data (e.g., 3 for EELS, 4 for 4D-STEM).")
        dat_unit = st.text_input("Data Unit", value="eV", help="Unit of the spectral axis.")
        cr_range_input = st.text_input("Crop Range (start, end, step)", value="1.0, 3.0, 0.01", help="Leave blank for None. Otherwise, provide comma-separated values. Float for DM files, integer for index range.")
        dat_scale = st.number_input("Data Scale", value=1.0, help="Specifies the actual step size if integer crop ranges are used.")
    with col2:
        rescale = st.checkbox("Rescale Data", value=False, help="If True, each hyperspectral data array will be divided by its maximum value during loading.")
        dm_file = st.checkbox("DM File Format (or HSPY)", value=True, help="Check this if you are using DM3/DM4 files requiring hyperspy.")
        verbose = st.checkbox("Verbose Output", value=True, help="Print progress and shapes to the console.")
        
    init_log = st.empty(); init_fig = st.container()
    if st.button("1. Initialize Model"):
        file_paths = [path.strip() for path in file_paths_input.split('\n') if path.strip()]
        if not file_paths: st.error("Please enter at least one file path.")
        else:
            def init_wrapper():
                cr = [float(x.strip()) for x in cr_range_input.split(",")] if cr_range_input else None
                st.session_state.model = drca(adr=file_paths, dat_dim=dat_dim, dat_unit=dat_unit, cr_range=cr, dat_scale=dat_scale, rescale=rescale, DM_file=dm_file, verbose=verbose)
            run_step('init', init_log, "Model Initialized.", init_wrapper)
    render_step_results('init', init_log, init_fig)

    st.write("---")
    st.header("Step 2: Binning (Optional)")
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        bin_y = st.number_input("Bin Y", value=1, min_value=1, help="Binning size in the height direction.")
        bin_x = st.number_input("Bin X", value=1, min_value=1, help="Binning size in the width direction.")
    with b_col2:
        str_y = st.number_input("Stride Y", value=1, min_value=1, help="Stride in the height direction.")
        str_x = st.number_input("Stride X", value=1, min_value=1, help="Stride in the width direction.")
    with b_col3:
        offset = st.number_input("Offset", value=0, help="Offset for the spectral depth dimension.")
        rescale_0to1_bin = st.checkbox("Rescale 0 to 1 (Binning)", value=True, help="Rescale each binned data pixel from 0 to 1.")

    bin_log = st.empty(); bin_fig = st.container()
    if st.button("2. Execute Binning"):
        if st.session_state.model: run_step('bin', bin_log, "Binning complete.", st.session_state.model.binning, bin_y, bin_x, str_y, str_x, offset=offset, rescale_0to1=rescale_0to1_bin)
        else: st.warning("Initialize Model first.")
    render_step_results('bin', bin_log, bin_fig)

    st.write("---")
    st.header("Step 3: Find Center (4D-STEM Only)")
    if dat_dim == 4:
        fc_col1, fc_col2 = st.columns(2)
        with fc_col1:
            cbox_edge = st.number_input("Center Box Edge", value=7, help="The edge length of the center box for finding the center position.")
            center_remove = st.number_input("Center Remove Range", value=0, help="If greater than zero, the center box specified by this radius will be removed from each diffraction pattern.")
        with fc_col2:
            fc_result_visual = st.checkbox("Show Center Visuals", value=True, help="Display the computed centers over the diffraction patterns.")
            fc_log_scale = st.checkbox("Log Scale Center Maps", value=True, help="Converts the intensities of each diffraction pattern into log-scale for visualization.")
            
        fc_log = st.empty(); fc_fig = st.container()
        if st.button("3. Find Center"):
            if st.session_state.model: run_step('center', fc_log, "Centers located.", st.session_state.model.find_center, cbox_edge=cbox_edge, center_remove=center_remove, result_visual=fc_result_visual, log_scale=fc_log_scale)
            else: st.warning("Initialize Model first.")
        render_step_results('center', fc_log, fc_fig)
    else:
        st.info("Find Center options are currently hidden. Change Data Dimension to 4 in Step 1 to access these controls.")
    
    st.write("---")
    st.header("Step 4: Prepare Input Dataset")
    mi_col1, mi_col2 = st.columns(2)
    with mi_col1:
        min_val = st.number_input("Minimum Value", value=0.0, help="Lower clipping bound for the data matrix to ensure non-negativity.")
        w_size = st.number_input("Window Size", value=0, help="Crop distance for box flattening.")
        radial_range_input = st.text_input("Radial Range (start, end, step)", value="", help="Leave blank for None. Provide comma-separated values for radial flattening.")
    with mi_col2:
        max_normalize = st.checkbox("Max Normalize", value=True, help="Divide the matrix by its maximum values.")
        rescale_0to1_input = st.checkbox("Rescale 0 to 1 (Input)", value=False, help="Rescale final flattened inputs from 0 to 1.")
        mi_log_scale = st.checkbox("Log Scale (Input)", value=False, help="Apply logarithmic scaling to the input matrix.")
        radial_flat = st.checkbox("Radial Flat", value=True, help="Use radial flattening instead of box flattening (for 4D-STEM).")
        final_dim = st.number_input("Final Dimension", value=1, min_value=1, max_value=2, help="Determines the shape of the flattened dataset (1 or 2).")

    in_log = st.empty(); in_fig = st.container()
    if st.button("4. Prepare Input Matrix"):
        if st.session_state.model:
            rr = [int(x.strip()) for x in radial_range_input.split(",")] if radial_range_input else None
            run_step('input', in_log, "Dataset flattened.", st.session_state.model.make_input, min_val=min_val, max_normalize=max_normalize, rescale_0to1=rescale_0to1_input, log_scale=mi_log_scale, radial_flat=radial_flat, w_size=w_size, radial_range=rr, final_dim=final_dim)
        else: st.warning("Initialize Model first.")
    render_step_results('input', in_log, in_fig)

with tab2:
    st.header("Initial Dimensionality Reduction")
    dr_col1, dr_col2 = st.columns(2)
    with dr_col1:
        dr_method = st.selectbox("Method", ["nmf", "pca", "cae"], help="First decomposition method (NMF is recommended for positive physical signals).")
        dr_num_comp = st.number_input("Number of Components (Initial)", value=5, min_value=2, help="Target dimensions for the first decomposition (e.g., number of expected phases).")
    with dr_col2:
        dr_intensity = st.selectbox("Intensity Range", ["absolute", "relative"], help="Determines the color mapping range for visual results.")
        dr_visual = st.checkbox("Show DR Visual Result", value=True, help="Plot the extracted loading vectors and their spatial maps.")
        
    idr_log = st.empty(); idr_fig = st.container()
    if st.button("Run Initial DR"):
        if st.session_state.model: run_step('ini_dr', idr_log, "DR Complete.", st.session_state.model.ini_DR, method=dr_method, num_comp=dr_num_comp, result_visual=dr_visual, intensity_range=dr_intensity)
        else: st.warning("Complete Tab 1 first.")
    render_step_results('ini_dr', idr_log, idr_fig)

with tab3:
    st.header("Augmented Dimensionality Reduction")
    aug_col1, aug_col2 = st.columns(2)
    with aug_col1:
        aug_method = st.selectbox("Method ", ["tsne", "umap"], help="Nonlinear dimensionality reduction algorithm to map data into islands.")
        aug_num_comp = st.number_input("Number of Components (Augmented)", value=2, min_value=2, help="Target spatial dimensions for mapping (usually 2 or 3).")
    with aug_col2:
        perplex_input = st.text_input("Perplexities", value="30, 40, 50", help="Comma-separated list of perplexity values to test during t-SNE. Balances local vs global structure.")
        
    adr_log = st.empty(); adr_fig = st.container()
    if st.button("Run Augmented DR"):
        if st.session_state.model:
            pl = [int(x.strip()) for x in perplex_input.split(",")]
            run_step('aug_dr', adr_log, "Augmented DR Complete.", st.session_state.model.aug_DR, num_comp=aug_num_comp, method=aug_method, perplex=pl)
        else: st.warning("Complete Tab 1 first.")
    render_step_results('aug_dr', adr_log, adr_fig)

with tab4:
    st.header("Density-Based Clustering")
    st.subheader("1. Prepare Clustering Space")
    pc_col1, pc_col2 = st.columns(2)
    with pc_col1: sel_ind = st.number_input("Select Embedding Index", value=1, min_value=1, help="Choose which DR space (e.g., from the perplexity list, 1-indexed) to use based on the previous tab's outputs.")
    with pc_col2: quick_visual = st.checkbox("Show Quick Visual", value=True, help="Plots the prepared spatial map before clustering.")
        
    pc_log = st.empty(); pc_fig = st.container()
    if st.button("Prepare Clustering"):
        if st.session_state.model: run_step('prep_clust', pc_log, "Space Extracted.", st.session_state.model.prepare_clustering, sel_ind=sel_ind, quick_visual=quick_visual)
        else: st.warning("Complete Tab 1 first.")
    render_step_results('prep_clust', pc_log, pc_fig)
            
    st.write("---")
    st.subheader("2. Interactive Clustering")
    cluster_method = st.selectbox("Clustering Algorithm", ["optics"], help="Algorithm to identify dense neighborhoods.")
    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1: msample = st.number_input("Min. Samples (Neighborhood)", value=0.05, step=0.01, format="%.3f", help="Minimum number of samples in a neighborhood for OPTICS.")
    with c_col2: steep = st.number_input("Steepness (xi)", value=0.001, step=0.001, format="%.4f", help="Minimum steepness on the reachability plot to declare a cluster boundary.")
    with c_col3: msize = st.number_input("Min. Samples (Cluster)", value=0.05, step=0.01, format="%.3f", help="Minimum number of samples required to form a valid cluster. Prevents noise from being labeled as a phase.")
    
    max_images = st.session_state.model.num_img if st.session_state.model else 1
    img_sel = st.number_input("Image Selection for Spatial Map", min_value=1, max_value=max_images, value=1, help="Select which original image slice to project the spatial distribution onto.")

    rc_log = st.empty(); rc_fig = st.container()
    if st.button("Run Clustering"):
        if st.session_state.model: run_step('run_clust', rc_log, "Clusters mapped. You can now proceed to Visualization.", st.session_state.model.preview_clustering, method=cluster_method, msample=msample, steep=steep, msize=msize, img_sel=img_sel)
        else: st.warning("Complete Tab 1 first.")
    render_step_results('run_clust', rc_log, rc_fig)

with tab5:
    st.header("Visualization & Export")
    vis_col1, vis_col2, vis_col3 = st.columns(3)
    with vis_col1: tf_map = st.checkbox("Render TF Map", value=False, help="Produces a True/False spatial mapping for every individual cluster.")
    with vis_col2: normalize_opt = st.selectbox("Normalize Type", ["max", "min", "none"], help="Divides each representative extracted spectrum by its extreme value to allow easy comparison.")
    with vis_col3: log_scale_vis = st.checkbox("Log Scale Plots", value=False, help="Plots the final output imagery on a logarithmic scale.")
        
    v_log = st.empty(); v_fig = st.container()
    if st.button("Render Final Results"):
        if st.session_state.model: run_step('vis', v_log, "Rendering complete. Ready to export.", st.session_state.model.clustering_result, tf_map=tf_map, normalize=normalize_opt, log_scale=log_scale_vis)
        else: st.warning("Complete Tab 1 first.")
    render_step_results('vis', v_log, v_fig)

    st.write("---")
    st.subheader("Export Results to TIFF")
    save_col1, save_col2 = st.columns(2)
    with save_col1: save_dir = st.text_input("Save Directory", value="./results", help="Directory path where the TIFF files will be saved.")
    with save_col2: save_prefix = st.text_input("File Prefix", value="drca_out", help="Prefix for the saved file names.")

    if st.button("Save TIFF Files"):
        if st.session_state.model is None or not hasattr(st.session_state.model, 'lines'):
            st.error("Please click 'Render Final Results' above before saving so the representative spectra can be extracted.")
        else:
            try:
                os.makedirs(save_dir, exist_ok=True)
                # Export Label Reshapes (Spatial Distributions)
                for idx, img_arr in enumerate(st.session_state.model.label_reshape):
                    tif_path = os.path.join(save_dir, f"{save_prefix}_labels_img{idx+1}.tif")
                    tifffile.imwrite(tif_path, img_arr.astype(np.float32))
                
                # Export Representative Spectra (Lines)
                spectra_path = os.path.join(save_dir, f"{save_prefix}_spectra.tif")
                tifffile.imwrite(spectra_path, st.session_state.model.lines.astype(np.float32))
                
                st.success(f"Files successfully written to: {os.path.abspath(save_dir)}")
            except Exception as e:
                st.error(f"Failed to save TIFFs: {str(e)}")