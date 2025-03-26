import logging
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import mahotas
import joblib
from skimage.measure import label, regionprops
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TumorDetectionPipeline:
    def __init__(self):
        #Inicializa o pipeline de detecção de tumores cerebrais com configurações padrão.
        current_dir = Path(__file__).parent

        self.MODEL_DIR = str(current_dir / "models")
        self.MODEL_PATH = str(current_dir / "models" / "random_forest_tumor.pkl")
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        self.MIN_IMAGE_SIZE = 512
        self.TUMOR_PROB_THRESHOLD = 0.65

        self.THRESHOLD_VALUE = 115
        self.MORPH_KERNEL_SIZE = (30, 30)
        self.NUM_MORPH_ITERATIONS = 3
        self.MIN_CEREBRAL_REGION = 5000

        self.model = self._load_model()
        self.executor = ThreadPoolExecutor(max_workers = 8)
        self.lock = threading.Lock()

    def _load_model(self):
        #Carrega o modelo de classificação de tumores cerebrais com fallback para modelos sem predict_proba.
        try:
            model = joblib.load(self.MODEL_PATH)

            if not hasattr(model, 'predict_proba'):
                class ProbabilityWrapper:
                    def __init__(self, model):
                        self.model = model

                    def predict_proba(self, X):
                        preds = self.model.predict(X)
                        return np.vstack([1 - preds, preds]).T

                    def predict(self, X):
                        return self.model.predict(X)

                model = ProbabilityWrapper(model)

            logger.info(f"Modelo de detecção cerebral carregado com sucesso de {self.MODEL_PATH}")
            return model
        except Exception as e:
            logger.error(f"Falha ao carregar modelo de classificação cerebral: {str(e)}")
            raise

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        #Aplica equalização adaptativa de histograma (CLAHE) para melhorar contraste em imagens cerebrais.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def _compute_lbp_features(self, image: np.ndarray) -> np.ndarray:
        #Calcula características de textura local usando Padrões Binários Locais (LBP).
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist

    def _compute_edge_features(self, image: np.ndarray) -> np.ndarray:
        #Extrai características de borda relevantes para tumores cerebrais usando Canny.
        edges = cv2.Canny(image, 100, 200)
        edge_density = np.mean(edges > 0)

        contours_result = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]

        if contours:
            areas = [cv2.contourArea(cnt) for cnt in contours]
            contour_features = [np.mean(areas), np.max(areas), len(contours)]
        else:
            contour_features = [0, 0, 0]

        return np.array([edge_density] + contour_features)

    def _extract_features_roi(self, roi: np.ndarray) -> np.ndarray:
        #Extrai um conjunto completo de características de uma região suspeita no cérebro.
        try:
            resized = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            equalized = self._apply_clahe(resized)
            filtered = cv2.bilateralFilter(equalized, 9, 75, 75)

            haralick = mahotas.features.haralick(filtered).mean(axis=0)

            hist = cv2.calcHist([filtered], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            features = [haralick, hist]

            distances = [1, 3, 5]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(filtered, distances=distances, angles=angles, levels=256, symmetric=True)

            glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(graycoprops, glcm, prop) for prop in glcm_props]
                for future in futures:
                    try:
                        features.append(future.result().ravel())
                    except Exception as e:
                        logger.error(f"Erro ao calcular propriedade GLCM: {str(e)}")
                        features.append(np.zeros(len(distances) * len(angles)))

            features.append(self._compute_lbp_features(filtered))
            features.append(self._compute_edge_features(filtered))

            return np.hstack(features)
        except Exception as e:
            logger.error(f"Erro na extração de características da ROI cerebral: {str(e)}")
            raise

    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        #Carrega e pré-processa uma imagem de ressonância cerebral para análise.
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagem cerebral não encontrada: {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
        if image is None:
            raise ValueError("Falha ao carregar imagem cerebral")

        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = image.shape
        if h < self.MIN_IMAGE_SIZE or w < self.MIN_IMAGE_SIZE:
            scale = max(self.MIN_IMAGE_SIZE / w, self.MIN_IMAGE_SIZE / h)
            new_size = (int(w * scale), int(h * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
            logger.info(f"Imagem cerebral redimensionada para {new_size} para análise")

        return self._apply_clahe(image)

    def segment_image(self, gray: np.ndarray) -> np.ndarray:
        #Segmenta a região cerebral usando técnicas morfológicas avançadas.
        try:
            _, thresh = cv2.threshold(gray, self.THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.MORPH_KERNEL_SIZE)

            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=self.NUM_MORPH_ITERATIONS)

            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=self.NUM_MORPH_ITERATIONS)

            cleaned = self._remove_small_regions(closed, min_size=self.MIN_CEREBRAL_REGION)

            return cleaned
        except Exception as e:
            logger.error(f"Erro na segmentação cerebral: {str(e)}")
            raise

    def segment_and_isolate(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        #Segmenta e isola a região cerebral, retornando máscara e imagem isolada.
        binary_segmented = self.segment_image(gray)

        contours, _ = cv2.findContours(binary_segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

        isolated = cv2.bitwise_and(gray, gray, mask=mask)

        return mask, isolated

    def _remove_small_regions(self, bin_img: np.ndarray, min_size: int) -> np.ndarray:
        #Remove regiões conectadas menores que min_size (artefatos) usando análise de componentes.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)

        output = np.zeros_like(bin_img)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                output[labels == i] = 255

        return output

    def dilate_and_fill(self, binary: np.ndarray) -> np.ndarray:
        #Aplica operações morfológicas para dilatar e preencher regiões cerebrais.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        dilated = cv2.dilate(binary, kernel, iterations=2)

        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)

        return closed

    def extract_region_features(self, filled: np.ndarray, gray: np.ndarray) -> list:
        #Extrai características de todas as regiões cerebrais segmentadas.
        features_list = []

        labeled_img = label(filled)

        if labeled_img.shape != gray.shape:
            labeled_img = cv2.resize(labeled_img.astype(np.uint8), (gray.shape[1], gray.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

        if len(gray.shape) == 3 and gray.shape[2] > 1:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        if labeled_img.shape != gray.shape:
            raise ValueError(f"Tamanhos incompatíveis: {labeled_img.shape} vs {gray.shape}")

        regions = regionprops(labeled_img, intensity_image=gray)

        futures = []

        for region in regions:
            if region.area < 100:
                continue
            futures.append(self.executor.submit(self._process_single_region, region, gray))

        for future in as_completed(futures):
            try:
                if (result := future.result()):
                    features_list.append(result)
            except Exception as e:
                logger.error(f"Erro ao processar região cerebral: {str(e)}")

        return features_list

    def _process_single_region(self, region, gray_image):
        #Processa uma única região cerebral suspeita extraindo características.
        minr, minc, maxr, maxc = region.bbox
        roi = gray_image[minr:maxr, minc:maxc]

        if roi.size == 0:
            return None

        try:
            features = self._extract_features_roi(roi)
            return {
                'bbox': region.bbox,
                'features': features,
                'area': region.area,
                'centroid': region.centroid
            }
        except Exception as e:
            logger.error(f"Falha ao extrair ROI cerebral: {str(e)}")
            return None

    def classify_regions(self, features_list: list) -> list:
        #Classifica todas as regiões cerebrais como benignas ou malignas.
        classified_regions = []
        futures = []

        for region in features_list:
            futures.append(self.executor.submit(self._classify_single_region, region))

        for future in as_completed(futures):
            try:
                classified_regions.append(future.result())
            except Exception as e:
                logger.error(f"Erro na classificação de tumor cerebral: {str(e)}")

        return classified_regions

    def _classify_single_region(self, region):
        #Classifica uma única região cerebral usando o modelo treinado.
        feat = region['features'].reshape(1, -1)
        proba = self.model.predict_proba(feat)[0][1]

        classification = 'malignant' if proba >= self.TUMOR_PROB_THRESHOLD else 'benign'
        region.update({
            'probability': proba,
            'classification': classification
        })

        return region  # Região com metadados de classificação

    def _save_image_async(self, path, image):
        #Salva imagens de forma assíncrona para melhor desempenho.
        try:
            cv2.imwrite(path, image)
        except Exception as e:
            logger.error(f"Erro ao salvar imagem {path}: {str(e)}")

    def process_pipeline(self, image_path: str) -> dict:
        'Executa o pipeline completo de detecção de tumores cerebrais.'
        try:
            filename = os.path.basename(image_path)
            base_dir = os.path.dirname(os.path.dirname(image_path))

            processed_dir = os.path.join(base_dir, 'Processada')
            preestatigos_dir = os.path.join(base_dir, 'préestatigos')
            os.makedirs(processed_dir, exist_ok=True)
            os.makedirs(preestatigos_dir, exist_ok=True)

            # estagio 1. Carregamento e Pré-processamento
            equalized = self.load_and_preprocess(image_path)
            he_filename = f"he_{filename}"
            he_path = os.path.join(processed_dir, he_filename)

            # estagio 2. Segmentação Cerebral
            binary_segmented = self.segment_image(equalized)

            mask = np.zeros_like(equalized)
            contours, _ = cv2.findContours(binary_segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
            isolated_brain = cv2.bitwise_and(equalized, equalized, mask=mask)


            segmented_vis = cv2.cvtColor(isolated_brain, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(segmented_vis, contours, -1, (0, 255, 0), 1)

            # estagio 3 Filtragem para Redução de Ruído
            filtered_global = cv2.bilateralFilter(isolated_brain, 9, 75, 75)

            # estagio 4. Operações Morfológicas para Refinamento
            filled = self.dilate_and_fill(binary_segmented)


            filled_vis = cv2.cvtColor(isolated_brain, cv2.COLOR_GRAY2BGR)
            filled_contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(filled_vis, filled_contours, -1, (255, 0, 0), 1)

            # estagio 5. Extração e Classificação de Características
            features_list = self.extract_region_features(filled, isolated_brain)
            if not features_list:
                logger.info("Nenhuma região cerebral relevante encontrada.")
                return {
                    'status': 'success',
                    'message': 'Nenhuma região cerebral relevante detectada.',
                    'processed_images': {
                        'original': filename,
                        'he': he_filename,
                        'seg': f"seg_{filename}",
                        'isolated': f"isolated_{filename}",
                        'filter': f"filter_{filename}",
                        'filled': f"filled_{filename}"
                    }
                }

            classified_regions = self.classify_regions(features_list)
            logger.info(f"Total de regiões cerebrais classificadas: {len(classified_regions)}")

            # estagio 6. Visualização Final com Resultados
            result_vis = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

            overlay = result_vis.copy()
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=cv2.FILLED)
            cv2.addWeighted(overlay, 0.3, result_vis, 0.7, 0, result_vis)

            for region in classified_regions:
                minr, minc, maxr, maxc = region['bbox']

                cv2.rectangle(result_vis, (minc, minr), (maxc, maxr), (0, 0, 255), 2)

            save_tasks = [
                (he_path, equalized),
                (os.path.join(processed_dir, f"seg_{filename}"), segmented_vis),
                (os.path.join(processed_dir, f"isolated_{filename}"), isolated_brain),
                (os.path.join(processed_dir, f"binary_seg_{Path(filename).stem}.png"), binary_segmented),
                (os.path.join(processed_dir, f"filter_{filename}"), filtered_global),
                (os.path.join(processed_dir, f"filled_{filename}"), filled_vis),
                (os.path.join(processed_dir, f"binary_filled_{Path(filename).stem}.png"), filled),
                (os.path.join(processed_dir, f"processed_{filename}"), result_vis)
            ]

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._save_image_async, path, img) for path, img in save_tasks]
                for future in as_completed(futures):
                    future.result()

            # estagio 7. Cópia para Diretório de Pré-Estatigos
            try:
                shutil.copy2(image_path, os.path.join(preestatigos_dir, filename))
            except Exception as e:
                logger.error(f"Erro ao copiar para préestatigos: {str(e)}")

            return {
                'status': 'success',
                'original_image': filename,
                'histogram_equalization_image': he_filename,
                'segmentation_image': f"seg_{filename}",
                'isolated_brain_image': f"isolated_{filename}",
                'filtering_image': f"filter_{filename}",
                'dilation_image': f"filled_{filename}",
                'processed_image': f"processed_{filename}",
                'regions': classified_regions,
                'analysis': {
                    'total_regions': len(classified_regions),
                    'malignant_count': sum(1 for r in classified_regions if r['classification'] == 'malignant'),
                    'benign_count': sum(1 for r in classified_regions if r['classification'] == 'benign')
                }
            }

        except Exception as e:
            logger.error(f"Falha no processamento de imagem cerebral: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'processed_images': {}
            }