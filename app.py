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
from commented_final import SVM  # your custom SVM implementation

# --- Custom Visualization Function ---
def visualize_decision_regions(model, X, y, title="Decision Regions"):
    """
    Generates a decision region plot in 2D.
    Assumes X has 2 features.
    """
    # Define bounds of the plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Flatten grid and predict decision function values for each point
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # For each grid point, use the model's decision_function:
    Z = np.array([model.decision_function(point) for point in grid_points])
    Z = Z.reshape(xx.shape)
    
    # Create a figure and plot contours and data points
    plt.figure()
    # Fill contours: areas where the decision function is <0 or >0
    plt.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], alpha=0.3, cmap="coolwarm")
    # Draw the decision boundary (where decision_function == 0)
    plt.contour(xx, yy, Z, levels=[0], colors="k", linewidths=2)
    # Overlay the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=30)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    return plt.gcf()

# --- Streamlit App: Blobs and Moons Visualization ---
st.title("Visualization of Blobs and Moons")

# Select dataset and kernel from sidebar
dataset_option = st.selectbox("Choose a dataset", ["Blobs", "Moons"])
kernel_option = st.selectbox("Choose a kernel", ["linear", "rbf"])

# Initialize session state flag if not already set
if "vis_running" not in st.session_state:
    st.session_state.vis_running = False

# Create a Start button that is disabled if processing is running
if st.button("Start Visualization", disabled=st.session_state.vis_running):
    st.session_state.vis_running = True

# If the visualization process is running, generate the plot
if st.session_state.vis_running:
    with st.spinner("Generating visualization..."):
        # Generate the selected dataset
        if dataset_option == "Blobs":
            X, y = make_blobs(n_samples=300, centers=2, cluster_std=1.0, random_state=42)
            # Convert labels: ensure values are -1 and 1 for SVM compatibility
            y = np.where(y == 0, -1, 1)
        else:  # Moons dataset
            X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
            y = np.where(y == 0, -1, 1)
        
        # Train the SVM model with the chosen kernel
        model = SVM(kernel=kernel_option, C=1.0, tol=1e-3, max_passes=20, max_iter=100, gamma=0.5)
        model.fit(X, y)
        
        # Create a Matplotlib figure using the custom visualization function
        fig = visualize_decision_regions(model, X, y, 
                title=f"{kernel_option.capitalize()} Kernel on {dataset_option}")
        st.pyplot(fig)
    st.success("Visualization complete!")
    # Reset the flag so that the button is clickable again.
    st.session_state.vis_running = False

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from commented_final import SVM, MultiClassSVM  # Import your SVM classes only

st.title("Model Evaluation")

# Sidebar selections for kernel and dataset
kernel_choice = st.selectbox("Select Kernel", ["linear", "poly", "rbf", "sigmoid"])
dataset_choice = st.selectbox("Select Dataset", ["HTRU2", "Wheat Seeds"])

# ----- Custom Helper Functions -----
# Custom confusion matrix and metrics functions for binary classification
def my_confusion_matrix_binary(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    return np.array([[TP, FN],
                     [FP, TN]])

def my_compute_metrics(y_true, y_pred):
    cm = my_confusion_matrix_binary(y_true, y_pred)
    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]
    accuracy = (TP + TN) / len(y_true)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1, cm

# Custom confusion matrix and metrics functions for multi-class classification
def my_confusion_matrix_multi(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for i, true_val in enumerate(classes):
        for j, pred_val in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_val) & (y_pred == pred_val))
    return cm, classes

def my_compute_metrics_multi(y_true, y_pred):
    cm, classes = my_confusion_matrix_multi(y_true, y_pred)
    accuracy = np.trace(cm) / len(y_true)
    precisions, recalls, f1s = [], [], []
    for i in range(len(classes)):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return accuracy, np.mean(precisions), np.mean(recalls), np.mean(f1s), cm, classes

# Helper functions to perform a simple PCA reduction for visualization
def my_compute_pca(X, n_components=2):
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    return mean, eigenvectors[:, :n_components]

def my_apply_pca(X, mean, components):
    return np.dot(X - mean, components)

# Custom decision region plot function inspired by your code
def my_plot_decision_regions(model, X, y, title="Decision Regions"):
    # Reduce data to 2D via PCA
    mean, components = my_compute_pca(X, n_components=2)
    X_pca = my_apply_pca(X, mean, components)
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Map the grid back to original space via the inverse PCA transform
    X_approx = mean + np.dot(grid, components.T)
    # Determine predictions on the grid. For binary classification, use decision_function.
    if len(np.unique(y)) == 2:
        Z = np.array([model.decision_function(x) for x in X_approx])
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], alpha=0.2, cmap="bwr")
        plt.contour(xx, yy, Z, levels=[0], colors="k", linewidths=2)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="bwr", edgecolors="k", s=30)
    else:
        # For multi-class, simply use predictions.
        Z = model.predict(X_approx)
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="rainbow")
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="rainbow", edgecolors="k", s=30)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    return plt.gcf()

# ----- End of Helper Functions -----

# Initialize session state variable for model evaluation page
if 'eval_running' not in st.session_state:
    st.session_state.eval_running = False

# Create the Start button and disable if evaluation is running
if st.button("Start Model Evaluation", disabled=st.session_state.eval_running):
    st.session_state.eval_running = True

if st.session_state.eval_running:
    with st.spinner("Running model evaluation..."):
        if dataset_choice == "HTRU2":
            try:
                # Load dataset
                data = pd.read_csv("HTRU_2.csv", header=None)
                X_htru = data.iloc[:, 0:8].values
                y_htru = data.iloc[:, 8].values
                # Convert classes: 0 -> -1, 1 -> 1
                y_htru = np.where(y_htru == 0, -1, 1)
                # Select a balanced subset of 250 samples per class
                pos_indices = np.where(y_htru == 1)[0]
                neg_indices = np.where(y_htru == -1)[0]
                n_sample_per_class = 100
                if len(pos_indices) < n_sample_per_class or len(neg_indices) < n_sample_per_class:
                    st.error("Dataset does not have enough samples for a balanced selection.")
                else:
                    selected_pos = np.random.choice(pos_indices, n_sample_per_class, replace=False)
                    selected_neg = np.random.choice(neg_indices, n_sample_per_class, replace=False)
                    selected_indices = np.concatenate([selected_pos, selected_neg])
                    X_htru = X_htru[selected_indices]
                    y_htru = y_htru[selected_indices]
                    X_train, X_test, y_train, y_test = train_test_split(X_htru, y_htru, test_size=0.2, random_state=42)
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    
                    # Train binary SVM
                    model = SVM(kernel=kernel_choice, C=1, tol=1e-3, max_passes=20, max_iter=100, gamma=0.5, degree=3)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    accuracy, precision, recall, f1, cm = my_compute_metrics(y_test, y_pred)
                    st.subheader("Metrics (HTRU2)")
                    st.write(f"Accuracy: {accuracy:.3f}")
                    st.write(f"Precision: {precision:.3f}")
                    st.write(f"Recall: {recall:.3f}")
                    st.write(f"F1 Score: {f1:.3f}")
                    
                    # Plot the custom confusion matrix
                    fig1, ax1 = plt.subplots()
                    ax1.matshow(cm, cmap="Blues", alpha=0.7)
                    for (i, j), val in np.ndenumerate(cm):
                        ax1.text(j, i, f"{val}", ha='center', va='center')
                    ax1.set_xticklabels([''] + ['Pred -1', 'Pred 1'])
                    ax1.set_yticklabels([''] + ['True -1', 'True 1'])
                    ax1.set_title(f"{kernel_choice.capitalize()} Kernel – HTRU2 Confusion Matrix")
                    st.pyplot(fig1)
                    
                    # Plot decision regions
                    fig2 = my_plot_decision_regions(model, X_test, y_test,
                                                    title=f"{kernel_choice.capitalize()} Kernel – HTRU2 Decision Regions")
                    st.pyplot(fig2)
                    
            except Exception as e:
                st.error(f"Error with HTRU2 dataset: {e}")
    
        else:  # Wheat Seeds dataset (multi-class)
            try:
                data = pd.read_csv("wheat_seeds.csv", header=0)
                X = data.iloc[:, 0:7].values
                y = data.iloc[:, 7].values
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                # Train multi-class SVM using One-vs-Rest
                model = MultiClassSVM(kernel=kernel_choice, C=1, tol=1e-3,
                                      max_passes=20, max_iter=100, gamma=0.5, degree=3)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy, precision, recall, f1, cm, classes = my_compute_metrics_multi(y_test, y_pred)
                st.subheader("Metrics (Wheat Seeds)")
                st.write(f"Accuracy: {accuracy:.3f}")
                st.write(f"Macro Precision: {precision:.3f}")
                st.write(f"Macro Recall: {recall:.3f}")
                st.write(f"Macro F1 Score: {f1:.3f}")
                
                # Plot the confusion matrix for multi-class
                fig1, ax1 = plt.subplots()
                ax1.matshow(cm, cmap="Blues", alpha=0.7)
                for (i, j), val in np.ndenumerate(cm):
                    ax1.text(j, i, f"{val}", ha='center', va='center')
                ax1.set_xticks(np.arange(len(classes)))
                ax1.set_yticks(np.arange(len(classes)))
                ax1.set_xticklabels(classes)
                ax1.set_yticklabels(classes)
                ax1.set_xlabel("Predicted Class")
                ax1.set_ylabel("True Class")
                ax1.set_title(f"{kernel_choice.capitalize()} Kernel – Wheat Seeds Confusion Matrix")
                st.pyplot(fig1)
                
                # Plot decision regions (for multi-class)
                fig2 = my_plot_decision_regions(model, X_test, y_test,
                                                title=f"{kernel_choice.capitalize()} Kernel – Wheat Seeds Decision Regions")
                st.pyplot(fig2)
                
            except Exception as e:
                st.error(f"Error with Wheat Seeds dataset: {e}")
    
    st.success("Model evaluation complete!")
    st.session_state.eval_running = False



