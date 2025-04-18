import streamlit as st

import streamlit as st
from streamlit_lottie import st_lottie
import json
import base64

def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

lottie_bg = load_lottie("0ByN8qzzTL.json")

# 2) Render the Lottie animation (size doesn't matter here)

# 3) CSS override to full‑screen the injected iframe
st.markdown("""
<style>
.lottie-background-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1;
        opacity: 0.2;
        pointer-events: none;
        overflow: hidden;
    }

/* target the Lottie iframe and force it full-screen behind everything */
div[data-testid="stAnimation"] iframe {
  position: absolute !important;
  top: 0 !important;
  left: 0 !important;
  width: 100vw !important;
  height: 100vh !important;
  z-index: -1 !important;
  pointer-events: none !important;
  opacity: 0.2 !important;
}
</style>
""", unsafe_allow_html=True)

with st.container():
    st_lottie(lottie_bg, speed=1, loop=True, quality="low", key="bg_anim")

# Define the pages
pages = {
    "Hlavné Menu": "main_menu",
    "Vizualizácia fungovania kernelov": "visualization",
    "Testovanie modelov": "model_eval",
    "Informácie o datasetoch": "info"
}

# Simulate a topbar with columns
from streamlit_option_menu import option_menu

# Horizontal menu at the top
selected = option_menu(
    menu_title=None,
    options=["Domov", "Vizualizácia", "Evaluácia", "Datasety"],
    icons=["house", "bar-chart", "clipboard-data", "file-earmark-text"],
    orientation="horizontal",
)

#st.title(f"{selected}")

# Create a sidebar radio button for navigation
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #2C3036;
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #8FBCBB;
    }
    </style>
""", unsafe_allow_html=True)

#selected_page = st.sidebar.radio("Navigácia", list(pages.keys()))
#st.sidebar.write("Autor: Daniel Zemančík")

from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url: str):
    return requests.get(url).json()

if selected == "Domov":
    lottie_anim = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_x62chJ.json")
    

    st.title("SVM Aplikácia")
    st.markdown(""" 
    **Prehľad:**  
    Táto aplikácia demonštruje vlastnú implementáciu Support Vector Machine (SVM) využívajúcu SMO (Sequential Minimal Optimization) pre optimalizáciu modelu. Cieľom je poskytnúť interaktívny nástroj na experimentovanie s rôznymi kernelovými funkciami, ktorý pomáha pochopiť, ako SVM funguje pri rozdeľovaní dát do jednotlivých tried.
    - **Vizualizácia - Bloby a Moons (polmesiace):**  
      V tejto sekcii sa vytvárajú syntetické datasety – blobs reprezentujúce viaceré klastre a moons s prekrývajúcimi sa oblasťami. Použitie rôznych kernelov, ukáže, ako SVM dokáže oddeliť dáta aj v prípade nelineárnych vzorov. Vizualizácia obsahuje interaktívne zobrazenie rozhodovacích hraníc, ktoré ilustrujú, ako sa jednotlivé triedy navzájom delia.
    - **Testovanie Modelov:**  
      Táto časť aplikácie umožňuje používateľovi vybrať kernel a datasety (HTRU2 alebo Wheat Seeds) a následne trénuje model na vybraných dátach. Po tréningu sa zobrazia kľúčové metriky, ako sú správnosť, presnosť, návratnosť a F1 skóre, spolu s konfúznou maticou a vizualizáciou rozhodovacích oblastí. Tieto informácie pomáhajú lepšie pochopiť správanie a výkonnosť modelu v praxi.
    Aplikácia je ideálna pre študentov, výskumníkov a každého, kto sa zaujíma o strojové učenie, pretože poskytuje prehľad o teoretických i praktických aspektoch SVM a umožňuje intuitívne skúmať, ako rôzne parametre ovplyvňujú klasifikačné výsledky.  
    """)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from commented_final import SVM, MultiClassSVM  # your custom SVM implementation

# --- Custom Visualization Function ---
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

def my_plot_decision_regions(centers, model, X, y, title="Decision Regions"):
    
    # Reduce data to 2D via PCA
    mean, components = my_compute_pca(X, n_components=2)
    X_pca = my_apply_pca(X, mean, components)
    
    # Determine plot boundaries in the PCA space
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Map grid back to original space using the inverse PCA transform
    grid = np.c_[xx.ravel(), yy.ravel()]
    X_approx = mean + np.dot(grid, components.T)
    
    num_classes = centers
    plt.figure()
    
    if num_classes == 2:
        # ---- Binary classification branch ----
        Z = np.array([model.decision_function(x) for x in X_approx])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], alpha=0.2, cmap="bwr")
        plt.contour(xx, yy, Z, levels=[0], colors="k", linewidths=2)
    else:
        # ---- Multi-class branch ----
        # Fill each region with the predicted label using a discrete colormap.
        Z_pred = model.predict(X_approx)
        Z_pred = Z_pred.reshape(xx.shape)
        plt.contourf(xx, yy, Z_pred, alpha=0.3, cmap="rainbow")
        
        # Now, attempt to draw pairwise decision boundaries.
        # This block expects model.decision_function to return an array of shape (n_points, n_classes)
        try:
            decision_values = model.decision_function(X_approx)  # shape (n_points, n_classes)
            decision_values = np.array(decision_values)
            for i in range(num_classes):
                for j in range(i + 1, num_classes):
                    # Compute the difference: decision value for class i minus that for class j
                    F = decision_values[:, i] - decision_values[:, j]
                    F = F.reshape(xx.shape)
                    plt.contour(xx, yy, F, levels=[0], colors="k", linestyles="--", linewidths=1)
        except Exception as e:
            # If model.decision_function is unavailable for multi-class,
            # a fallback is to compute boundaries from discrete predictions.
            unique_vals = np.sort(np.unique(Z_pred))
            if len(unique_vals) > 1:
                boundaries = [(unique_vals[k] + unique_vals[k + 1]) / 2 for k in range(len(unique_vals) - 1)]
                plt.contour(xx, yy, Z_pred, levels=boundaries, colors="k", linestyles="--")
    
    # Overlay the (PCA-transformed) data points using the same discrete colormap.
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="rainbow", edgecolor="k", s=30)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    return plt.gcf()
        
# --- Streamlit App: Blobs and Moons Visualization ---
if selected == "Vizualizácia":
    st.title("Vizualizácia fungovania kernelov")
    st.markdown("""
    Táto sekcia demonštruje generovanie syntetických datasetov, konkrétne blobs a moons, ktoré slúžia na vizualizáciu rozhodovacích hraníc SVM modelu.  
    - **Blobs:** Použitie datasetu s viacerými klastrami, kde každý klaster reprezentuje inú triedu.  
    - **Moons:** Dataset tvorený dvoma prekrývajúcimi sa polmesiacmi, ktorý je vhodný pre vizualizáciu nelineárne oddelených tried.  
    - Pri vybranom datasete sa trénuje SVM model s vybraným kernelom (lineárny alebo RBF) a následne sa vykreslia rozhodovacie oblasti, ktoré ukazujú, ako model rozdeľuje jednotlivé triedy na základe trénovacích dát.
    """)
    
    # Select dataset and kernel from sidebar
    dataset_option = st.selectbox("Choose a dataset", ["Blobs", "Moons"])
    kernel_option = st.selectbox("Choose a kernel", ["linear", "rbf"])
    centers_option = 2
    
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
                X, y = make_blobs(n_samples=30, n_features=centers_option, centers=centers_option, cluster_std=1.0, random_state=42)
                # Convert labels: ensure values are -1 and 1 for SVM compatibility
                y = np.where(y == 0, -1, 1)
            else:  # Moons dataset
                X, y = make_moons(n_samples=30, noise=0.2, random_state=42)
                y = np.where(y == 0, -1, 1)
            
            # Train the SVM model with the chosen kernel
            if (centers_option == 2 and dataset_option == "Blobs") or dataset_option == "Moons":
                model = SVM(kernel=kernel_option, C=1.0, tol=1e-3, max_passes=20, max_iter=100, gamma=0.5)
                model.fit(X, y)
                fig = my_plot_decision_regions(centers_option, model, X, y, 
                    title=f"{kernel_option.capitalize()} Kernel on {dataset_option}")
                st.pyplot(fig)
            else:
                model = MultiClassSVM(kernel=kernel_option, C=1.0, tol=1e-3, max_passes=20, max_iter=100, gamma=0.5)
                model.fit(X, y)
                fig = my_plot_decision_regions(centers_option, model, X, y, 
                    title=f"{kernel_option.capitalize()} Kernel on {dataset_option}")
                st.pyplot(fig)
            
            # Create a Matplotlib figure using the custom visualization function
            
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

if selected == "Evaluácia":
    st.title("Testovanie modelov")
    st.markdown("""
    V tejto časti sa testuje a vyhodnocuje SVM (alebo MultiClassSVM) model použitím reálnych datasetov, ako je HTRU2 alebo Wheat Seeds. Tieto datasety boli taktiež použité pri vypracovaní hlavného zadania. 
    - **Spracovanie dát:** Dataset je načítaný, normalizovaný a rozdelený na trénovaciu a testovaciu množinu.  
    - **Tréning modelu:** Na základe vybraného kernelu sa trénuje model na trénovacích dátach.  
    - **Hodnotenie:** Po trénovaní sa model testuje na testovacej množine, pričom sa počítajú metriky ako accuracy, precision, recall a F1 skóre.  
    - **Vizualizácia výsledkov:** Sú zobrazené konfúzna matica a rozhodovacie oblasti, čo poskytuje prehľad o správaní a výkonnosti modelu.
    """)
    
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

if selected == "Datasety":
    st.header("Informácie o datasetoch")
    st.markdown("""
    Dataset, ktorý používame, pochádza zo štúdie [HTRU2](https://archive.ics.uci.edu/ml/datasets/HTRU2), ktorá analyzuje signály pulsarov získavané z rádiových prenosov. Tento dataset obsahuje rôzne atribúty, napríklad štatistické veličiny signálu, a cieľová premenná je binárna – určuje, či ide o pulsar, alebo nie. Dáta zo štúdie HTRU2 boli využité na detekciu pulsarov a poskytujú zaujímavý základ pre experimentovanie s klasifikáciou pomocou SVM, najmä pre demonštráciu oddelenia dvoch tried pomocou rôznych kernelových funkcií.
    """)

    st.markdown("""
    Pre multi-klasifikáciu môžete použiť aj dataset [Wheat Seeds](https://archive.ics.uci.edu/ml/datasets/seeds), ktorý obsahuje údaje o troch rôznych typoch semien pšenice. Tento dataset zahŕňa geometrické a štrukturálne vlastnosti semien, ktoré umožňujú rozlíšiť medzi jednotlivými odrodami. Vďaka viackategóriovému charakteru je ideálny pre demonštráciu, ako modely zvládajú úlohy s viacerými triedami a vizualizáciu rozhodovacích hraníc medzi tromi zaujímavými skupinami.
    """)
    
    st.markdown("""
    Oba datasety ponúkajú atraktívny základ pre experimenty v oblasti strojového učenia, umožňujú porovnanie výkonnosti algoritmov na binárnych aj viackategóriových úlohách a prispievajú k lepšiemu pochopeniu fungovania SVM modelov.
    """)
