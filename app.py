import streamlit as st

# Define the pages
pages = {
    "Main Menu": "main_menu",
    "Blobs and Moons Visualization": "visualization",
    "Model Evaluation": "model_eval"
}

# Create a sidebar radio button for navigation
selected_page = st.sidebar.radio("Navigation", list(pages.keys()))

if selected_page == "Main Menu":
    st.title("SVM Application Demo")
    st.markdown("""
    **Overview:**  
    This app demonstrates a custom Support Vector Machine (SVM) implementation using the SMO algorithm.  
    - Use the **Blobs and Moons Visualization** to see how different kernels separate data.  
    - Use **Model Evaluation** to select a kernel and dataset, view performance metrics, and see the confusion matrix.  
    """)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from commented_final import SVM, plot_decision_regions  # your module and plotting function

# Page title
st.title("Visualization of Blobs and Moons")

# Sidebar selections for dataset and kernel
dataset_option = st.selectbox("Choose a dataset", ["Blobs", "Moons"])
kernel_option = st.selectbox("Choose a kernel", ["linear", "rbf"])

# Initialize session state variable if not already set
if 'vis_running' not in st.session_state:
    st.session_state.vis_running = False

# Create the Start button; disable it if processing is running
if st.button("Start Visualization", disabled=st.session_state.vis_running):
    # Set running flag to True so the button becomes disabled on the next rerun
    st.session_state.vis_running = True

# Check the flag and, if set, run the visualization code
if st.session_state.vis_running:
    with st.spinner("Generating visualization..."):
        # Generate the selected dataset
        if dataset_option == "Blobs":
            X, y = make_blobs(n_samples=30, centers=2, cluster_std=1.0, random_state=42)
            y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1 for SVM
        else:
            X, y = make_moons(n_samples=30, noise=0.2, random_state=42)
            y = np.where(y == 0, -1, 1)
    
        # Train the SVM model using the selected kernel
        model = SVM(kernel=kernel_option, C=1.0, tol=1e-3, max_passes=20, max_iter=100, gamma=0.5)
        model.fit(X, y)
    
        # Create a Matplotlib figure using your plotting function.
        fig, ax = plt.subplots()
        # Ensure your plot_decision_regions function is adapted so it does NOT call plt.show()
        plot_decision_regions(model, X, y, title=f"{kernel_option.capitalize()} Kernel on {dataset_option}")
    
        # Display the figure in Streamlit
        st.pyplot(fig)
    
    # Notify the user the process is complete
    st.success("Visualization complete!")
    # Reset the running flag to re-enable the button in the next run.
    st.session_state.vis_running = False



