import React, { useState, useEffect } from 'react';
import axios from 'axios';

function ImageUploader({ onResults, location, onImageSelect }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  useEffect(() => {
    if (onImageSelect) onImageSelect(!!selectedFile);
  }, [selectedFile, onImageSelect]);

  // Handles file selection for preview and reset
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setErrorMsg(null);
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnnotatedImage(null);
      if (onResults) onResults(null);
    } else {
      setSelectedFile(null);
      setPreviewUrl(null);
      if (onImageSelect) onImageSelect(false);
    }
  };

  // Handles form submit (image + location upload)
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile || !location?.lat || !location?.lng) {
      setErrorMsg('Please select an image and provide a valid location.');
      return;
    }

    setLoading(true);
    setErrorMsg(null);
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('latitude', location.lat);
    formData.append('longitude', location.lng);

    try {
      const backendUrl = process.env.REACT_APP_BACKEND_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${backendUrl}/api/infer`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setAnnotatedImage(response.data.annotated_image_url);
      if (onResults) onResults(response.data);
    } catch (error) {
      setErrorMsg(error?.response?.data?.error || 'Error uploading image');
      setAnnotatedImage(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '8px 0', marginBottom: '2em', background: '#fff', borderRadius: '7px' }}>
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '18px', alignItems: 'center', flexWrap: 'wrap' }}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button
          type="submit"
          disabled={loading || !selectedFile || !location?.lat || !location?.lng}
          style={{
            padding: '8px 20px',
            fontSize: '16px',
            backgroundColor: loading ? '#aaa' : '#28a745',
            color: 'white',
            border: 'none',
            borderRadius: '7px',
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? 'Processing...' : 'Upload & Classify'}
        </button>
      </form>
      
      {/* Error Message */}
      {errorMsg && (
        <div style={{ color: 'red', marginTop: '9px', fontWeight: 500 }}>
          {errorMsg}
        </div>
      )}

      {/* Image Preview */}
      {previewUrl && (
        <div style={{ marginTop: '18px' }}>
          <h4 style={{ marginBottom: '7px' }}>Preview</h4>
          <img src={previewUrl} alt="Preview" style={{ maxWidth: '320px', borderRadius: '8px', boxShadow: '0 1px 6px rgba(0,0,0,0.08)' }} />
        </div>
      )}

      {/* Annotated Result */}
      {annotatedImage && (
        <div style={{ marginTop: '22px' }}>
          <h4 style={{ marginBottom: '7px' }}>Annotated Result</h4>
          <img src={annotatedImage} alt="Annotated" style={{ maxWidth: '420px', borderRadius: '8px', boxShadow: '0 1px 8px rgba(0,0,0,0.12)' }} />
        </div>
      )}
    </div>
  );
}

export default ImageUploader;
