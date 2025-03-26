from flask import Blueprint, request, jsonify, send_from_directory, current_app
from werkzeug.utils import secure_filename
import os
import time
import logging
import shutil
from datetime import datetime, timedelta
from .image_processing import TumorDetectionPipeline

main = Blueprint('api', __name__, url_prefix='/api')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_old_files(hours=1):
    #Remove arquivos com mais de X horas em todas as pastas
    for subfolder in ['Original', 'Processada', 'préestatigos']:
        target_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], subfolder)
        if not os.path.exists(target_dir):
            continue

        cutoff = (datetime.now() - timedelta(hours=hours)).timestamp()

        for filename in os.listdir(target_dir):
            filepath = os.path.join(target_dir, filename)
            if os.path.getmtime(filepath) < cutoff:
                try:
                    os.remove(filepath)
                    logger.info(f"Removido arquivo antigo: {subfolder}/{filename}")
                except Exception as e:
                    logger.error(f"Erro ao remover {filepath}: {str(e)}")


def has_disk_space(min_space=50):
    #Verifica se há pelo menos min_space MB livres
    try:
        import psutil
        disk_usage = psutil.disk_usage(current_app.config['UPLOAD_FOLDER'])
        free = disk_usage.free / (1024 * 1024)
        return free > min_space
    except Exception as e:
        logger.error(f"Erro ao verificar espaço em disco: {str(e)}")
        return True


def save_to_preestatigos(source_path, target_filename):
    #Salva uma cópia do arquivo na pasta préestatigos
    try:
        preestatigos_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'préestatigos')
        os.makedirs(preestatigos_dir, exist_ok=True)

        target_path = os.path.join(preestatigos_dir, target_filename)
        shutil.copy2(source_path, target_path)
        logger.info(f"Arquivo salvo em préestatigos: {target_path}")
        return target_path
    except Exception as e:
        logger.error(f"Erro ao salvar em préestatigos: {str(e)}")
        return None


@main.route('/analyze', methods=['POST'])
def analyze_image():
    clean_old_files()

    try:
        if 'file' not in request.files:
            logger.error("Nenhum arquivo na requisição")
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400

        file = request.files['file']

        if file.filename == '':
            logger.error("Nome de arquivo vazio")
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

        if not allowed_file(file.filename):
            logger.error(f"Tipo de arquivo não suportado: {file.filename}")
            return jsonify({'error': 'Tipo de arquivo não suportado'}), 400

        if file.content_length > MAX_FILE_SIZE:
            logger.error(f"Arquivo muito grande: {file.content_length} bytes")
            return jsonify({'error': 'Arquivo muito grande (máx 10MB)'}), 400

        if not has_disk_space():
            logger.error("Espaço em disco insuficiente")
            return jsonify({'error': 'Espaço em disco insuficiente'}), 500

        upload_dir = current_app.config['UPLOAD_FOLDER']
        original_dir = os.path.join(upload_dir, 'Original')
        processed_dir = os.path.join(upload_dir, 'Processada')
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        timestamp = str(int(time.time()))
        original_filename = secure_filename(file.filename)
        filename = f"{timestamp}_{original_filename}"
        original_path = os.path.join(original_dir, filename)
        file.save(original_path)

        preestatigos_original_path = save_to_preestatigos(original_path, filename)

        try:
            start_time = time.time()
            detector = TumorDetectionPipeline()
            result = detector.process_pipeline(original_path)
            processing_time = round(time.time() - start_time, 2)

            if result.get('status') != 'success':
                raise Exception(result.get('message', 'Erro no processamento'))

            processed_images = {
                'original_image_url': f"/api/uploads/Original/{filename}",
                'histogram_equalization_url': f"/api/uploads/Processada/{result.get('histogram_equalization_image')}",
                'segmentation_url': f"/api/uploads/Processada/{result.get('segmentation_image')}",
                'filtering_url': f"/api/uploads/Processada/{result.get('filtering_image')}",
                'dilation_url': f"/api/uploads/Processada/{result.get('dilation_image')}",
                'processed_image_url': f"/api/uploads/Processada/{result.get('processed_image')}",
                'preestatigos_url': f"/api/uploads/préestatigos/{filename}"
            }

            for key, image_url in processed_images.items():
                if key in ['original_image_url', 'preestatigos_url']:
                    continue
                source_path = os.path.join(upload_dir, image_url.replace("/api/uploads/", ""))
                target_filename = os.path.basename(image_url)
                save_to_preestatigos(source_path, target_filename)

            for key, image_url in processed_images.items():
                if key == 'preestatigos_url':
                    continue
                image_path = os.path.join(upload_dir, image_url.replace("/api/uploads/", ""))
                if not os.path.exists(image_path):
                    logger.warning(f"Imagem {key} não encontrada: {image_path}")
                    processed_images[key] = None

            logger.info(f"Processamento concluído em {processing_time}s")

            return jsonify({
                'status': 'success',
                'processing_time': processing_time,
                **processed_images,
                'analysis': result.get('analysis', {}),
                'metadata': {
                    'filename': original_filename,
                    'upload_time': datetime.now().isoformat(),
                    'size': os.path.getsize(original_path),
                    'preestatigos_path': f"/api/uploads/préestatigos/{filename}"
                }
            })

        except Exception as e:
            logger.error(f"Erro no processamento: {str(e)}", exc_info=True)
            if os.path.exists(original_path):
                os.remove(original_path)
            if preestatigos_original_path and os.path.exists(preestatigos_original_path):
                os.remove(preestatigos_original_path)
            return jsonify({
                'error': 'Erro no processamento da imagem',
                'details': str(e)
            }), 500

    except Exception as e:
        logger.error(f"Erro geral: {str(e)}", exc_info=True)
        return jsonify({'error': 'Erro interno no servidor'}), 500


@main.route('/uploads/<subfolder>/<filename>')
def serve_image(subfolder, filename):
    try:
        upload_dir = current_app.config['UPLOAD_FOLDER']
        file_path = os.path.join(upload_dir, subfolder, filename)

        if not os.path.exists(file_path):
            logger.error(f"Arquivo não encontrado: {file_path}")
            return jsonify({'error': 'Imagem não encontrada'}), 404

        return send_from_directory(os.path.join(upload_dir, subfolder), filename)

    except Exception as e:
        logger.error(f"Erro ao servir imagem: {str(e)}")
        return jsonify({'error': 'Erro ao acessar imagem'}), 500


@main.route('/status')
def service_status():
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'disk_space': has_disk_space(),
        'upload_dir': current_app.config['UPLOAD_FOLDER'],
        'directories': {
            'original': os.path.join(current_app.config['UPLOAD_FOLDER'], 'Original'),
            'processed': os.path.join(current_app.config['UPLOAD_FOLDER'], 'Processada'),
            'preestatigos': os.path.join(current_app.config['UPLOAD_FOLDER'], 'préestatigos')
        }
    })
