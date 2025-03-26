import os
import cv2
import mahotas
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

DATASET_PATH = "datasets/exames"
CATEGORIES = ["normal", "tumor"]
MODEL_PATH = "models/random_forest_tumor.pkl"
FEATURES_VERSION = "v3"
MIN_IMAGE_SIZE = 512
RANDOM_STATE = 42


def apply_preprocessing(image):
    #Aplica pré-processamento avançado na imagem
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    image = cv2.bilateralFilter(image, 9, 75, 75)

    return image


def extract_features(image_path):
    #Extrai características avançadas da imagem
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        height, width = image.shape
        if height < MIN_IMAGE_SIZE or width < MIN_IMAGE_SIZE:
            scale = max(MIN_IMAGE_SIZE / height, MIN_IMAGE_SIZE / width)
            new_size = (int(width * scale), int(height * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)

        image = cv2.resize(image, (256, 256))

        image = apply_preprocessing(image)

        haralick = mahotas.features.haralick(image).mean(axis=0)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        features = [haralick, hist]

        if FEATURES_VERSION in ["v2", "v3"]:
            glcm_features = compute_glcm_features(image)
            features.append(glcm_features)

        if FEATURES_VERSION == "v3":
            lbp_features = compute_lbp_features(image)
            edge_features = compute_edge_features(image)
            features.extend([lbp_features, edge_features])

        return np.hstack(features)
    except Exception as e:
        print(f"Erro ao processar {image_path}: {str(e)}")
        return None


def compute_glcm_features(image):
    #Calcula características de matriz de co-ocorrência de níveis de cinza (GLCM)
    distances = [1, 3, 5]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True)

    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = []
    for prop in props:
        features.append(graycoprops(glcm, prop).ravel())

    return np.hstack(features)


def compute_lbp_features(image, radius=3, n_points=24):
    #Calcula características de padrão binário local (LBP)
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalização
    return hist


def compute_edge_features(image):
    """Calcula características baseadas em bordas"""
    edges = cv2.Canny(image, 100, 200)
    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_features = []
    if len(contours) > 0:
        areas = [cv2.contourArea(cnt) for cnt in contours]
        contour_features = [
            np.mean(areas),
            np.max(areas),
            len(contours)
        ]
    else:
        contour_features = [0, 0, 0]

    return np.array([edge_density] + contour_features)


def load_dataset():
    X, y = [], []

    for label, category in enumerate(CATEGORIES):
        category_path = os.path.join(DATASET_PATH, category)
        print(f"\nProcessando {category}...")

        files = [f for f in os.listdir(category_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

        for filename in tqdm(files, desc=f"Processando {category}"):
            image_path = os.path.join(category_path, filename)
            features = extract_features(image_path)

            if features is not None:
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)


def plot_feature_importances(model, feature_names=None):
    #Plota a importância das características do modelo
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Importância das Características")
    plt.bar(range(len(importances)), importances[indices], align='center')

    if feature_names:
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    else:
        plt.xticks(range(len(importances)), indices, rotation=90)

    plt.tight_layout()
    plt.savefig("feature_importances.png")
    plt.close()


def train_and_evaluate():
    #Pipeline completo de treinamento
    print("\nCarregando dataset...")
    X, y = load_dataset()

    print("\nEstatísticas do dataset:")
    print(f"- Total de amostras: {len(X)}")
    print(f"- Distribuição de classes: Normal={np.sum(y == 0)}, Tumor={np.sum(y == 1)}")
    print(f"- Dimensões das features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    print("\nConfigurando busca de hiperparâmetros...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    }

    print("\nTreinando modelo com GridSearchCV...")
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='roc_auc'
    )
    grid_search.fit(X_train, y_train)

    print("\nMelhores parâmetros encontrados:")
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    print("\nAvaliando modelo no conjunto de teste:")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.2f}")

    print("\nImportância das características:")
    plot_feature_importances(best_model)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print(f"\nModelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    train_and_evaluate()