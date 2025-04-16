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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from commented_final import SVM, plot_decision_regions  # Assuming your code is in commented_final.py

if selected_page == "Blobs and Moons Visualization":
    st.title("Visualization of Blobs and Moons")

    # Option for user to select dataset
    dataset_option = st.selectbox("Choose a dataset", ["Blobs", "Moons"])
    kernel_option = st.selectbox("Choose a kernel", ["linear", "rbf"])

    if dataset_option == "Blobs":
        X, y = make_blobs(n_samples=300, centers=2, cluster_std=1.0, random_state=42)
        # Convert labels to -1 and 1
        y = np.where(y==0, -1, 1)
    else:  # Moons dataset
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
        y = np.where(y==0, -1, 1)

    # Train a simple SVM using the selected kernel
    model = SVM(kernel=kernel_option, C=1.0, tol=1e-3, max_passes=20, max_iter=100, gamma=0.5)
    model.fit(X, y)
    
    # Create a decision boundary plot
    fig, ax = plt.subplots()
    plot_decision_regions(model, X, y, title=f"{kernel_option.capitalize()} Kernel on {dataset_option}")
    
    # Display the plot on the Streamlit app
    st.pyplot(fig)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from commented_final import SVM, MultiClassSVM, compute_metrics, compute_metrics_multi, plot_confusion_matrix

if selected_page == "Model Evaluation":
    st.title("Model Evaluation")
    
    # Kernel and dataset selectors
    kernel_choice = st.selectbox("Select Kernel", ["linear", "poly", "rbf", "sigmoid"])
    dataset_choice = st.selectbox("Select Dataset", ["HTRU2", "Wheat Seeds"])

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
            
            # Plot confusion matrix
            fig, ax = plt.subplots()
            plot_confusion_matrix(cm, classes=['-1', '1'], title=f"{kernel_choice.capitalize()} Kernel – HTRU2 Confusion Matrix")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error loading or processing HTRU2 dataset: {e}")

    else:  # Wheat Seeds dataset
        try:
            data = pd.read_csv("wheat_seeds.csv", header=0)
            X = data.iloc[:, 0:7].values
            y = data.iloc[:, 7].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Using MultiClassSVM for multi‑class classification
            model = MultiClassSVM(kernel=kernel_choice, C=1, tol=1e-3, max_passes=20, max_iter=100, gamma=0.5, degree=3)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Compute multi‑class metrics
            accuracy, precision, recall, f1, cm, classes = compute_metrics_multi(y_test, y_pred)
            
            st.subheader("Metrics")
            st.write(f"Accuracy: {accuracy:.3f}")
            st.write(f"Macro Precision: {precision:.3f}")
            st.write(f"Macro Recall: {recall:.3f}")
            st.write(f"Macro F1 Score: {f1:.3f}")
            
            # Plot confusion matrix for multi‑class
            fig, ax = plt.subplots()
            plot_confusion_matrix(cm, classes=[str(c) for c in classes],
                                  title=f"{kernel_choice.capitalize()} Kernel – Wheat Seeds Confusion Matrix")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error loading or processing Wheat Seeds dataset: {e}")
