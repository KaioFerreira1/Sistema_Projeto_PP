import React, { useState } from 'react';
import axios from 'axios';
import kaioferreira from '../assets/images/membros-equipe/img_kaio.jpg';
import alexfelix from '../assets/images/membros-equipe/img-alex.jpg';
import ivandersonamaral from '../assets/images/membros-equipe/img-ivanderson.png';
import luizcarlos from '../assets/images/membros-equipe/img-luiz.jpg';
import './ImageUploader.css';

const ImageUploader = () => {
  const [result, setResult] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [heImage, setHeImage] = useState(null);
  const [segImage, setSegImage] = useState(null);
  const [filterImage, setFilterImage] = useState(null);
  const [dilationImage, setDilationImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  // Dados do time de desenvolvedores
  const teamMembers = [
    {
      name: "Kaio Ferreira",
      role: "Full-Stack Developer",
      photo: kaioferreira,
      github: "https://github.com/KaioFerreira1",
      linkedin: "https://www.linkedin.com/in/kaioferreiradev/"
    },
    {
      name: "Alex Félix",
      role: "Frontend Developer",
      photo: alexfelix,
      github: "https://github.com/OAleex",
      linkedin: "https://www.linkedin.com/in/alex-f%C3%A9lix-319b5b196?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app "
    },
    {
      name: "Ivanderson Amaral",
      role: "Machine Learning",
      photo: ivandersonamaral,
      github: "https://github.com/IvandersonDev",
      linkedin: "https://www.linkedin.com/in/ivanderson-amaral-0645b2193/"
    },
    {
      name: "Luiz Carlos",
      role: "UI/UX Designer",
      photo: luizcarlos,
      github: "https://github.com/LuizDogui18",
      linkedin: "https://www.linkedin.com/in/luiz-carlos-38316a236/"
    }
  ];

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageUpload({ target: { files: e.dataTransfer.files } });
    }
  };

  const handleImageUpload = async (e) => {
    const file = e.target.files?.[0] || e.dataTransfer.files?.[0];
    if (!file) return;

    const validTypes = ['image/jpeg', 'image/png', 'image/tiff', 'image/bmp'];
    if (!validTypes.some(type => file.type.includes(type))) {
      setError('Por favor, envie apenas imagens (JPG, PNG, TIFF, BMP)');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('O arquivo é muito grande (máximo 10MB)');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setOriginalImage(null);
    setHeImage(null);
    setSegImage(null);
    setFilterImage(null);
    setDilationImage(null);
    setProcessedImage(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/api/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000
      });

      if (!response.data || response.data.status !== 'success') {
        throw new Error(response.data?.message || 'Resposta inválida do servidor');
      }

      const baseUrl = 'http://localhost:5000';
      const imageUrls = {
        original: `${baseUrl}${response.data.original_image_url || ''}`,
        he: `${baseUrl}${response.data.histogram_equalization_url || ''}`,
        seg: `${baseUrl}${response.data.segmentation_url || ''}`,
        filter: `${baseUrl}${response.data.filtering_url || ''}`,
        dilation: `${baseUrl}${response.data.dilation_url || ''}`,
        processed: `${baseUrl}${response.data.processed_image_url || ''}`
      };

      Object.entries(imageUrls).forEach(([key, url]) => {
        if (!url.includes('http://localhost:5000/api/uploads/')) {
          console.warn(`URL inválida para ${key}: ${url}`);
          imageUrls[key] = null;
        }
      });

      setResult(response.data);
      setOriginalImage(imageUrls.original);
      setHeImage(imageUrls.he);
      setSegImage(imageUrls.seg);
      setFilterImage(imageUrls.filter);
      setDilationImage(imageUrls.dilation);
      setProcessedImage(imageUrls.processed);

    } catch (err) {
      console.error('Erro no processamento:', err);
      setError(err.response?.data?.error ||
               err.response?.data?.message ||
               err.message ||
               'Erro ao processar a imagem');
    } finally {
      setLoading(false);
    }
  };

  const renderImageBox = (title, imgSrc, alt) => {
    if (!imgSrc) return null;

    return (
      <div className="image-box">
        <h4>{title}</h4>
        <img
          src={imgSrc}
          alt={alt}
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = '';
            e.target.style.display = 'none';
            console.warn(`Erro ao carregar imagem: ${imgSrc}`);
          }}
        />
      </div>
    );
  };

  const renderResults = () => {
    if (!result) return null;

    return (
      <div className="results-container">
        <h3>Resultados da Análise</h3>
        <div className="image-grid">
          {renderImageBox('Imagem Original', originalImage, 'Original')}
          {renderImageBox('Histograma Equalizado', heImage, 'Histograma Equalizado')}
          {renderImageBox('Segmentação', segImage, 'Segmentação')}
          {renderImageBox('Filtragem', filterImage, 'Filtragem')}
          {renderImageBox('Dilatação/Preenchimento', dilationImage, 'Dilatação/Preenchimento')}
          {renderImageBox('Imagem Final (Classificação)', processedImage, 'Processada')}
        </div>
      </div>
    );
  };

  const renderTeamSection = () => (
    <div className="team-section">
      <h3 className="team-title">Equipe de Desenvolvimento</h3>
      <div className="team-row">
        {teamMembers.map((member, index) => (
          <div key={index} className="team-member">
            <div className="member-photo-container">
              <img src={member.photo} alt={member.name} className="member-photo" />
            </div>
            <div className="member-info">
              <h4>{member.name}</h4>
              <p className="member-role">{member.role}</p>
              <div className="member-links">
                <a href={member.github} target="_blank" rel="noopener noreferrer">
                  <i className="fab fa-github"></i>
                </a>
                <a href={member.linkedin} target="_blank" rel="noopener noreferrer">
                  <i className="fab fa-linkedin"></i>
                </a>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="main-container">
      <div className="uploader-container">
        <div
          className={`upload-area ${dragActive ? 'active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            id="file-upload"
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            disabled={loading}
            style={{ display: 'none' }}
          />
          <label htmlFor="file-upload" className="upload-label">
            {loading ? (
              <div className="loading-spinner"></div>
            ) : (
              <>
                <svg className="upload-icon" viewBox="0 0 24 24">
                  <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
                </svg>
                <p>Arraste e solte sua imagem aqui ou clique para selecionar</p>
                <p className="small">Formatos suportados: JPG, PNG, TIFF, BMP (até 10MB)</p>
              </>
            )}
          </label>
        </div>

        {error && (
          <div className="error-message">
            <p>{error}</p>
            <button onClick={() => setError(null)}>Fechar</button>
          </div>
        )}

        {renderResults()}
      </div>

      {renderTeamSection()}
    </div>
  );
};

export default ImageUploader;