import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # Použitý pre 3D vizualizáciu
from sklearn.neighbors import KNeighborsClassifier
import pickle  # Na uloženie natrénovaných modelov

# Nastavenie semien pre replikovateľnosť náhodných operácií
np.random.seed(42)    # Pre numpy generátory náhodných čísel
random.seed(42)       # Pre knižnicu random

###########################################################
# Implementácia binárneho SVM pomocou SMO algoritmu (pre interné použitie)
###########################################################
class SVM:
    """
    Implementácia SVM (stroj podporných vektorov) pomocou SMO algoritmu.

    Podporované kernely:
      - 'linear': Lineárny kernel
      - 'poly': Polynomiálny kernel
      - 'rbf': RBF (Gaussovský) kernel
      - 'sigmoid': Sigmoidový kernel
      
    Parametre:
      kernel: typ kernela
      C: parameter mäkkých okrajov (penalizácia chybných klasifikácií)
      tol: tolerancia pre kontrolu KKT podmienok
      max_passes: počet prechodov bez zmeny α pred ukončením optimalizácie
      max_iter: maximálny počet iterácií (bez ohľadu na počet prechodov)
      degree: stupeň (pre polynomiálny kernel)
      gamma: parameter pre poly, rbf a sigmoid kernel
      coef0: voľný parameter pre poly a sigmoid kernel
    """
    def __init__(self, kernel='linear', C=1.0, tol=1e-3, max_passes=5, max_iter=1000,
                 degree=3, gamma=1.0, coef0=1.0):
        self.kernel_name = kernel    # Uloženie názvu zvoleného kernela
        self.C = C                  # Regularizačný parameter
        self.tol = tol              # Tolerancia pre porušenie KKT podmienok
        self.max_passes = max_passes# Počet iterácií bez zmeny α pred zastavením
        self.max_iter = max_iter    # Maximálny počet iterácií
        self.degree = degree        # Parameter pre polynomiálny kernel
        self.gamma = gamma          # Gamma parameter (ovplyvňuje šírku kernela)
        self.coef0 = coef0          # Konštantný parameter pre poly a sigmoid kernel
        
        # Výber kernelovej funkcie podľa zvoleného parametra
        if kernel == 'linear':
            self.kernel = self.linear_kernel
        elif kernel == 'poly':
            self.kernel = self.poly_kernel
        elif kernel == 'rbf':
            self.kernel = self.rbf_kernel
        elif kernel == 'sigmoid':
            self.kernel = self.sigmoid_kernel
        else:
            raise ValueError("Nepodporovaný kernel. Použite 'linear', 'poly', 'rbf' alebo 'sigmoid'.")

        # Inicializácia premenných, ktoré budú obsahovať natrénované hodnoty
        self.alphas = None  # Lagrangeove multiplikátory pre každý tréningový príklad
        self.b = 0          # Bias (posun hyperroviny)
        self.X = None       # Tréningová množina príkladov
        self.y = None       # Cieľové hodnoty

    def linear_kernel(self, x1, x2):
        # Vypočíta skalárny súčin dvoch vektorov
        return np.dot(x1, x2)

    def poly_kernel(self, x1, x2):
        # Vypočíta polynomiálny kernel: (γ * <x1, x2> + coef0)^degree
        return (self.gamma * np.dot(x1, x2) + self.coef0) ** self.degree

    def rbf_kernel(self, x1, x2):
        # Vypočíta RBF kernel (Gaussovský kernel)
        diff = x1 - x2
        return np.exp(-self.gamma * np.dot(diff, diff))
    
    def sigmoid_kernel(self, x1, x2):
        # Vypočíta sigmoidový kernel: tanh(γ * <x1, x2> + coef0)
        return np.tanh(self.gamma * np.dot(x1, x2) + self.coef0)

    def fit(self, X, y):
        """
        Trénuje SVM model pomocou SMO algoritmu.
        Ukladá trénovacie dáta do seba a inicializuje multiplikátory.
        Potom iteratívne aktualizuje multiplikátory podľa SMO algoritmu.
        """
        self.X = X
        self.y = y
        m, _ = X.shape
        self.alphas = np.zeros(m)  # Inicializácia multiplikátorov na nulu
        self.b = 0                 # Inicializácia biasu
        passes = 0                 # Počet prechodov bez zmeny multiplikátorov
        iters = 0                  # Celkový počet iterácií
        
        # Predpočet kernelovej matice – výpočet vo všetkých bodoch (O(n^2))
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = self.kernel(X[i], X[j])
        
        # Hlavný cyklus SMO algoritmu:
        while passes < self.max_passes and iters < self.max_iter:
            num_changed_alphas = 0  # Počet multiplikátorov, ktoré sa v tejto iterácii zmenili
            for i in range(m):
                # Výpočet chyby pre i-tý príklad (E_i = f(x_i) - y_i)
                E_i = self.decision_function(X[i]) - y[i]
                # Kontrola KKT podmienok; ak sú porušené, pokračujeme s aktualizáciou multiplikátorov
                if (y[i] * E_i < -self.tol and self.alphas[i] < self.C) or (y[i] * E_i > self.tol and self.alphas[i] > 0):
                    # Vyberieme index j náhodne, pričom i != j
                    j_candidates = [j for j in range(m) if j != i]
                    j = random.choice(j_candidates)
                    E_j = self.decision_function(X[j]) - y[j]
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    # Výpočet hraníc L a H pre nové hodnoty α_j podľa podmienok
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    if L == H:
                        continue
                    
                    # Výpočet eta – parameter pre aktualizáciu
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                        
                    # Aktualizácia α_j a jeho obmedzenie na interval [L, H]
                    self.alphas[j] = self.alphas[j] - (y[j]*(E_i - E_j)) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    # Ak sa zmena nie je dostatočná, pokračujeme s ďalším príkladom
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Aktualizácia α_i tak, aby bola zachovaná podmienka (α_i + α_j = const)
                    self.alphas[i] = self.alphas[i] + y[i]*y[j]*(alpha_j_old - self.alphas[j])
                    
                    # Výpočet aktualizovaných hodnot pre bias b pomocou dvoch alternatívnych aktualizácií
                    b1 = self.b - E_i - y[i]*(self.alphas[i]-alpha_i_old)*K[i, i] - y[j]*(self.alphas[j]-alpha_j_old)*K[i, j]
                    b2 = self.b - E_j - y[i]*(self.alphas[i]-alpha_i_old)*K[i, j] - y[j]*(self.alphas[j]-alpha_j_old)*K[j, j]
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1  # Zvýšenie počtu upravených multiplikátorov
            # Ak žiadne multiplikátory sa nezmenili, zväčšíme počet prechodov
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
            iters += 1  # Zvýšenie počtu iterácií
        return self

    def decision_function(self, x):
        """
        Vypočíta hodnotu rozhodovacej funkcie pre daný vstup x.
        Suma cez všetky trénovacie príklady s vážením podľa multiplikátorov a cieľových hodnôt.
        """
        result = 0
        for i in range(len(self.X)):
            result += self.alphas[i] * self.y[i] * self.kernel(self.X[i], x)
        return result + self.b

    def predict(self, X):
        """
        Predikuje triedy pre zadanú množinu príkladov.
        Ak rozhodovacia funkcia vráti hodnotu >= 0, priradí triedu 1, inak -1.
        """
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(1 if self.decision_function(X[i]) >= 0 else -1)
        return np.array(y_pred)

###########################################################
# MultiClassSVM – One-vs-Rest wrapper pre multi-klasifikačné úlohy
###########################################################
class MultiClassSVM:
    """
    Rozširuje binárne SVM na multi-klasifikačnú úlohu pomocou stratégie One-vs-Rest.
    Pre každú triedu sa vytvorí samostatný SVM klasifikátor, ktorý odlišuje túto triedu od ostatných.
    """
    def __init__(self, kernel='linear', C=1.0, tol=1e-3, max_passes=5, max_iter=1000,
                 degree=3, gamma=1.0, coef0=1.0):
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.classifiers = {}  # Slovník, kde pre každú triedu bude uložený príslušný SVM model
        self.classes = None    # Unikátne triedy získané z tréningových dát

    def fit(self, X, y):
        """
        Trénuje multi-klasifikačný model pomocou stratégie One-vs-Rest.
        Pre každú triedu vytvorí binárny SVM, kde príklady danej triedy majú hodnotu 1 a ostatné -1.
        """
        self.classes = np.unique(y)
        self.classifiers = {}
        for cls in self.classes:
            y_binary = np.where(y == cls, 1, -1)
            clf = SVM(kernel=self.kernel, C=self.C, tol=self.tol,
                      max_passes=self.max_passes, max_iter=self.max_iter,
                      degree=self.degree, gamma=self.gamma, coef0=self.coef0)
            clf.fit(X, y_binary)
            self.classifiers[cls] = clf
        return self

    def decision_functions(self, X):
        """
        Vráti maticu rozhodovacích hodnôt pre každý príklad a každú triedu.
        Každý stĺpec predstavuje rozhodovaciu funkciu pre jednu triedu.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        outputs = np.zeros((n_samples, n_classes))
        for idx, cls in enumerate(self.classes):
            clf = self.classifiers[cls]
            for i in range(n_samples):
                outputs[i, idx] = clf.decision_function(X[i])
        return outputs

    def predict(self, X):
        """
        Predikuje triedu pre každý príklad na základe maximálnej hodnoty rozhodovacej funkcie.
        """
        outputs = self.decision_functions(X)
        indices = np.argmax(outputs, axis=1)
        return self.classes[indices]

###########################################################
# Metriky a konfúzna matica
###########################################################
def confusion_matrix_custom(y_true, y_pred):
    # Vypočíta konfúznu maticu pre binárnu klasifikáciu (2x2)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    return np.array([[TP, FN],
                     [FP, TN]])

def compute_metrics(y_true, y_pred):
    # Vypočíta základné metriky: Accuracy, Precision, Recall, F1 a vráti aj konfúznu maticu.
    cm = confusion_matrix_custom(y_true, y_pred)
    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]
    accuracy = (TP + TN) / len(y_true)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1, cm

def confusion_matrix_multi(y_true, y_pred):
    # Vytvorí konfúznu maticu pre viactriednu klasifikáciu.
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for i, true_val in enumerate(classes):
        for j, pred_val in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_val) & (y_pred == pred_val))
    return cm, classes

def compute_metrics_multi(y_true, y_pred):
    # Vypočíta metriky pre viactriednu klasifikáciu vrátane Accuracy, Macro Precision, Recall a F1.
    cm, classes = confusion_matrix_multi(y_true, y_pred)
    accuracy = np.sum(np.diag(cm)) / len(y_true)
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

###########################################################
# Redukcia dimenzionality a vizualizácie
###########################################################
def compute_pca_custom(X, n_components=2):
    # Vykoná vlastnú PCA: centrovanie dát, výpočet kovariančnej matice,
    # výpočet vlastných hodnôt a vlastných vektorov a výber hlavných komponent.
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    return mean, eigenvectors[:, :n_components]

def apply_pca_custom(X, mean, components):
    # Projekcia dát na zvolené hlavné komponenty
    return np.dot(X - mean, components)

def plot_decision_regions(model, X, y, title='Rozhodovacia hranica'):
    # Vizualizácia rozhodovacích hraníc pomocou 2D PCA redukcie
    mean, components = compute_pca_custom(X, n_components=2)
    X_pca = apply_pca_custom(X, mean, components)
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # Aproximácia bodov späť do pôvodného priestoru
    X_approx = mean + np.dot(grid_points, components.T)
    Z = []
    for x in X_approx:
        Z.append(model.decision_function(x))
    Z = np.array(Z).reshape(xx.shape)
    
    # plt.figure()
    # plt.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], alpha=0.2, cmap=plt.cm.bwr)
    # plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.bwr, edgecolors='k')
    # plt.title(title)
    # plt.xlabel('Hlavná zložka 1')
    # plt.ylabel('Hlavná zložka 2')
    # plt.tight_layout()

def plot_decision_regions_multiclass(model, X, y, title='Rozhodovacia hranica'):
    # Vizualizácia rozhodovacích hraníc pre viactriednu klasifikáciu pomocou PCA
    mean, components = compute_pca_custom(X, n_components=2)
    X_pca = apply_pca_custom(X, mean, components)
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    X_approx = mean + np.dot(grid_points, components.T)
    Z = model.predict(X_approx)
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.rainbow)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.rainbow, edgecolors='k')
    plt.title(title)
    plt.xlabel('Hlavná zložka 1')
    plt.ylabel('Hlavná zložka 2')
    plt.tight_layout()

def plot_3d_pca(X, y, title="3D PCA"):
    # Redukcia dát na 3 rozmery pomocou PCA a vizualizácia v 3D priestore
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label in np.unique(y):
        ax.scatter(X_pca[y == label, 0], X_pca[y == label, 1], X_pca[y == label, 2], label=str(label))
    ax.set_title(title)
    ax.legend()
    plt.show()

def plot_tsne(X, y, title="t-SNE Visualizácia"):
    # Vizualizácia dát pomocou t-SNE redukcie na 2 rozmery
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    plt.figure()
    for label in np.unique(y):
        plt.scatter(X_tsne[y==label, 0], X_tsne[y==label, 1], label=str(label))
    plt.title(title)
    plt.legend()
    plt.show()

def plot_tsne_svm(X_train, X_test, y_test, best_model, title="t-SNE vizualizácia SVM predikcií"):
    """
    Vykoná t-SNE transformáciu zjednotených trénovacích a testovacích dát a zobrazí scatter plot,
    kde sú trénovacie body farebne kódované podľa predikcií SVM a testovacie body sú označené krížikmi.
    """
    # Spojenie tréningových a testovacích dát
    X_comb = np.vstack([X_train, X_test])
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_comb)
    
    n_train = X_train.shape[0]
    X_tsne_train = X_tsne[:n_train]
    X_tsne_test = X_tsne[n_train:]
    
    # Získanie predikcií pre tréningové dáta z modelu SVM
    y_train_pred = best_model.predict(X_train)
    
    plt.figure()
    # Vykreslenie tréningových bodov (farebné podľa predikcie)
    scatter_train = plt.scatter(X_tsne_train[:, 0], X_tsne_train[:, 1], c=y_train_pred, cmap=plt.cm.rainbow,
                                label="Trénovacie predikcie", alpha=0.7)
    # Vykreslenie testovacích bodov (označené krížikmi podľa skutočných tried)
    scatter_test = plt.scatter(X_tsne_test[:, 0], X_tsne_test[:, 1], marker='x', s=100, c=y_test, cmap=plt.cm.rainbow,
                               label="Testovacie skutočné triedy")
    plt.title(title)
    plt.xlabel("t-SNE zložka 1")
    plt.ylabel("t-SNE zložka 2")
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, classes, title='Konfúzna matica'):
    """
    Vykreslí konfúznu maticu pre dané triedy.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Skutočná trieda')
    plt.xlabel('Predikovaná trieda')
    plt.tight_layout()

###########################################################
# Hlavná časť – experimenty na dvoch datasetoch
###########################################################
def main():
    # Experimenty na HTRU2 datasete (binárna úloha)
    print("Experimenty na HTRU2 datasete (binárna úloha):")
    try:
        data = pd.read_csv("HTRU_2.csv", header=None)
    except Exception as e:
        print("Chyba pri načítaní HTRU_2.csv:", e)
        return
    X_htru = data.iloc[:, 0:8].values
    y_htru = data.iloc[:, 8].values
    # Transformácia tried: 0 -> -1 a 1 -> 1 pre binárne SVM
    y_htru = np.where(y_htru == 0, -1, 1)
    pos_indices = np.where(y_htru == 1)[0]
    neg_indices = np.where(y_htru == -1)[0]
    n_sample_per_class = 250
    if len(pos_indices) < n_sample_per_class or len(neg_indices) < n_sample_per_class:
        print("Dataset nemá dosť vzoriek pre vyvážený výber.")
        return
    selected_pos = np.random.choice(pos_indices, n_sample_per_class, replace=False)
    selected_neg = np.random.choice(neg_indices, n_sample_per_class, replace=False)
    selected_indices = np.concatenate([selected_pos, selected_neg])
    X_htru = X_htru[selected_indices]
    y_htru = y_htru[selected_indices]
    X_train_htru, X_test_htru, y_train_htru, y_test_htru = train_test_split(X_htru, y_htru, test_size=0.2, random_state=42)
    scaler_htru = StandardScaler()
    X_train_htru = scaler_htru.fit_transform(X_train_htru)
    X_test_htru = scaler_htru.transform(X_test_htru)
    
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    # Pre každý kernel trénujeme model a vyhodnocujeme metriku na HTRU2 datasete
    for kernel in kernels:
        print(f"\nKernel: {kernel} pre HTRU2 dataset")
        svm_model = SVM(kernel=kernel, C=1, tol=1e-3, max_passes=20, max_iter=100,
                        degree=3, gamma=0.5, coef0=1.0)
        svm_model.fit(X_train_htru, y_train_htru)
        y_pred_htru = svm_model.predict(X_test_htru)
        accuracy, precision, recall, f1, cm = compute_metrics(y_test_htru, y_pred_htru)
        print(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        plot_confusion_matrix(cm, classes=['-1','1'], title=f"{kernel} – HTRU2 Konfúzna matica")
        plot_decision_regions(svm_model, X_test_htru, y_test_htru, title=f"{kernel} – Rozhodovacia hranica (HTRU2)")
        plt.show()
        
    # Experimenty na Wheat Seeds datasete (multiklasa)
    print("\nExperimenty na Wheat Seeds datasete (multiklasa):")
    try:
        data_wheat = pd.read_csv("wheat_seeds.csv", header=0)
    except Exception as e:
        print("Chyba pri načítaní wheat_seeds.csv:", e)
        return
    # Predpokladáme 7 atribútov a 8. stĺpec obsahuje triedu
    X_wheat = data_wheat.iloc[:, 0:7].values
    y_wheat = data_wheat.iloc[:, 7].values
    X_train_wheat, X_test_wheat, y_train_wheat, y_test_wheat = train_test_split(X_wheat, y_wheat, test_size=0.3, random_state=42)
    scaler_wheat = StandardScaler()
    X_train_wheat = scaler_wheat.fit_transform(X_train_wheat)
    X_test_wheat = scaler_wheat.transform(X_test_wheat)
    
    # Grid search na získanie najlepších parametrov pre každý kernel na Wheat Seeds datasete
    best_models = {}  # Slovník s kľúčom ako kernel a hodnotou ako tuple (najlepšia konfigurácia, natrénovaný model)
    for kernel in kernels:
        best_score = -1
        best_config = None
        best_model = None
        if kernel == 'linear':
            for max_iter in [100]:
                model = MultiClassSVM(kernel='linear', C=1, tol=1e-3, max_passes=20, max_iter=max_iter)
                model.fit(X_train_wheat, y_train_wheat)
                y_pred = model.predict(X_test_wheat)
                acc = np.mean(y_pred == y_test_wheat)
                if acc > best_score:
                    best_score = acc
                    best_config = {'kernel': kernel, 'max_iter': max_iter}
                    best_model = model
        elif kernel == 'poly':
            for degree in [2, 3, 4, 5, 6]:
                for gamma in [0.1, 0.5, 1.0, 1.5]:
                    for max_iter in [100]:
                        model = MultiClassSVM(kernel='poly', C=1, tol=1e-3, max_passes=20, max_iter=max_iter,
                                               degree=degree, gamma=gamma, coef0=1.0)
                        model.fit(X_train_wheat, y_train_wheat)
                        y_pred = model.predict(X_test_wheat)
                        acc = np.mean(y_pred == y_test_wheat)
                        if acc > best_score:
                            best_score = acc
                            best_config = {'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': 1.0, 'max_iter': max_iter}
                            best_model = model
        elif kernel == 'rbf':
            for gamma in [0.1, 0.5, 1.0]:
                for max_iter in [100]:
                    model = MultiClassSVM(kernel='rbf', C=1, tol=1e-4, max_passes=20, max_iter=max_iter,
                                           gamma=gamma)
                    model.fit(X_train_wheat, y_train_wheat)
                    y_pred = model.predict(X_test_wheat)
                    acc = np.mean(y_pred == y_test_wheat)
                    if acc > best_score:
                        best_score = acc
                        best_config = {'kernel': kernel, 'gamma': gamma, 'max_iter': max_iter}
                        best_model = model
        elif kernel == 'sigmoid':
            for gamma in [0.1, 0.5, 1.0]:
                for max_iter in [100]:
                    model = MultiClassSVM(kernel='sigmoid', C=1, tol=1e-3, max_passes=20, max_iter=max_iter,
                                           gamma=gamma, coef0=1.0)
                    model.fit(X_train_wheat, y_train_wheat)
                    y_pred = model.predict(X_test_wheat)
                    acc = np.mean(y_pred == y_test_wheat)
                    if acc > best_score:
                        best_score = acc
                        best_config = {'kernel': kernel, 'gamma': gamma, 'coef0': 1.0, 'max_iter': max_iter}
                        best_model = model
        best_models[kernel] = (best_config, best_model)
    
    # Pre každý kernel zobrazíme výsledky najlepšej konfigurácie
    for kernel in kernels:
        config, model = best_models[kernel]
        print(f"\nNajlepšia konfigurácia pre kernel {kernel}: {config}")
        y_pred = model.predict(X_test_wheat)
        accuracy, precision, recall, f1, cm, classes = compute_metrics_multi(y_test_wheat, y_pred)
        print(f"Accuracy: {accuracy:.3f}, Macro Precision: {precision:.3f}, Macro Recall: {recall:.3f}, Macro F1: {f1:.3f}")
        class_labels = [str(c) for c in classes]
        plot_confusion_matrix(cm, classes=class_labels, title=f"{kernel} – Wheat Seeds Konfúzna matica")
        plot_decision_regions_multiclass(model, X_test_wheat, y_test_wheat, title=f"{kernel} – Rozhodovacia hranica (Wheat Seeds)")
        plt.show()
        # t-SNE vizualizácia s rozhodovacími oblasťami
        plot_tsne_svm(X_train_wheat, X_test_wheat, y_test_wheat, model, title=f"{kernel} – t-SNE s rozhodovacími oblasťami")
    
    # Dodatočná 3D PCA vizualizácia pre Wheat Seeds testovacie dáta
    plot_3d_pca(X_test_wheat, y_test_wheat, title="Wheat Seeds – 3D PCA Vizualizácia")
    
    # Uloženie natrénovaných modelov do súboru pre neskoršie použitie
    with open('trained_models.pkl', 'wb') as f:
        pickle.dump(best_models, f)
    print("Natrénované modely boli uložené do súboru 'trained_models.pkl'.")

if __name__ == '__main__':
    main()

