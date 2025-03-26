import React from 'react';
import ImageUploader from './components/ImageUploader';
import './App.css';

function App() {
  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Sistema de Detecção de Tumores</h1>
        <p>Analise imagens médicas utilizando processamento avançado</p>
      </header>
      <main>
        <ImageUploader />
      </main>
      <footer className="app-footer">
        <p>Sistema desenvolvido para diagnóstico médico assistido</p>
      </footer>
    </div>
  );
}

export default App;