import React, { useState, useEffect } from 'react';
import axios from 'axios';

function ImageUploader({ onResults, location, onImageSelect }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Notify parent whether a valid image is selected
    if (onImageSelect) onImageSelect(!!selectedFile);
  }, [selectedFile, onImageSelect]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnnotatedImage(null);
      onResults(null);
    } else {
      setSelectedFile(null);
      setPreviewUrl(null);
      if (onImageSelect) onImageSelect(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile || !location || !location.lat || !location.lng) {
      alert('Please select an image and a valid location before uploading.');
      return;
    }

    setLoading(true);
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
      onResults(response.data);
    } catch (error) {
      console.error('Upload error:', error);
      alert('Error uploading image');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button type="submit" disabled={loading || !selectedFile || !location?.lat || !location?.lng}>
          {loading ? 'Processing...' : 'Upload and Classify'}
        </button>
      </form>

      {previewUrl && (
        <div>
          <h4>Preview</h4>
          <img src={previewUrl} alt="Preview" style={{ maxWidth: '300px' }} />
        </div>
      )}

      {annotatedImage && (
        <div>
          <h4>Annotated Result</h4>
          <img src={annotatedImage} alt="Annotated" style={{ maxWidth: '400px', marginTop: '12px' }} />
        </div>
      )}
    </div>
  );
}

export default ImageUploader;
