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
            X, y = make_blobs(n_samples=300, centers=2, cluster_std=1.0, random_state=42)
            y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1 for SVM
        else:
            X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
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

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from commented_final import SVM, MultiClassSVM, compute_metrics, compute_metrics_multi, plot_confusion_matrix

st.title("Model Evaluation")

# Sidebar selections for kernel and dataset
kernel_choice = st.selectbox("Select Kernel", ["linear", "poly", "rbf", "sigmoid"])
dataset_choice = st.selectbox("Select Dataset", ["HTRU2", "Wheat Seeds"])

# Initialize session state variable for model evaluation page
if 'eval_running' not in st.session_state:
    st.session_state.eval_running = False

# Create the Start button and disable if evaluation is running
if st.button("Start Model Evaluation", disabled=st.session_state.eval_running):
    st.session_state.eval_running = True

# Check if the evaluation process should run
if st.session_state.eval_running:
    with st.spinner("Running model evaluation..."):
        if dataset_choice == "HTRU2":
            try:
                data = pd.read_csv("HTRU_2.csv", header=None)
                X = data.iloc[:, 0:8].values
                y = data.iloc[:, 8].values
                y = np.where(y == 0, -1, 1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                model = SVM(kernel=kernel_choice, C=1, tol=1e-3, max_passes=20, max_iter=100, gamma=0.5, degree=3)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy, precision, recall, f1, cm = compute_metrics(y_test, y_pred)
                
                st.subheader("Metrics")
                st.write(f"Accuracy: {accuracy:.3f}")
                st.write(f"Precision: {precision:.3f}")
                st.write(f"Recall: {recall:.3f}")
                st.write(f"F1 Score: {f1:.3f}")
                
                # Plot the confusion matrix
                fig, ax = plt.subplots()
                plot_confusion_matrix(cm, classes=['-1', '1'], title=f"{kernel_choice.capitalize()} Kernel – HTRU2 Confusion Matrix")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error with HTRU2 dataset: {e}")
    
        else:  # Wheat Seeds dataset
            try:
                data = pd.read_csv("wheat_seeds.csv", header=0)
                X = data.iloc[:, 0:7].values
                y = data.iloc[:, 7].values
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                model = MultiClassSVM(kernel=kernel_choice, C=1, tol=1e-3, max_passes=20, max_iter=100, gamma=0.5, degree=3)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy, precision, recall, f1, cm, classes = compute_metrics_multi(y_test, y_pred)
                
                st.subheader("Metrics")
                st.write(f"Accuracy: {accuracy:.3f}")
                st.write(f"Macro Precision: {precision:.3f}")
                st.write(f"Macro Recall: {recall:.3f}")
                st.write(f"Macro F1: {f1:.3f}")
                
                fig, ax = plt.subplots()
                plot_confusion_matrix(cm, classes=[str(c) for c in classes], 
                                      title=f"{kernel_choice.capitalize()} Kernel – Wheat Seeds Confusion Matrix")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error with Wheat Seeds dataset: {e}")
    
    st.success("Model evaluation complete!")
    st.session_state.eval_running = False

