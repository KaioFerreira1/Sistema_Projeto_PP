from flask import Flask, send_from_directory
from flask_cors import CORS
import os

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploads'))
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

    CORS(app)

    # Garante que as pastas existam
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'Original'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'Processada'), exist_ok=True)

    from .routes import main
    app.register_blueprint(main)

    return app