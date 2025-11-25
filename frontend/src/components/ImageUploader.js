import React, { useState, useEffect } from 'react';
import axios from 'axios';

function ImageUploader({ onResults, location, onImageSelect }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadConfirmed, setUploadConfirmed] = useState(false);

  useEffect(() => {
    if (onImageSelect) onImageSelect(!!selectedFile);
  }, [selectedFile, onImageSelect]);

  // Drag and drop handlers
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      processFile(files[0]);
    }
  };

  // Process file for both drag-drop and file input
  const processFile = (file) => {
    if (!file.type.startsWith('image/')) {
      setErrorMsg('Please upload an image file');
      setUploadConfirmed(false);
      return;
    }
    setErrorMsg(null);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setAnnotatedImage(null);
    setUploadConfirmed(true);
    if (onResults) onResults(null);
    
    // Auto-hide confirmation after 3 seconds
    setTimeout(() => setUploadConfirmed(false), 3000);
  };

  // Handles file selection for preview and reset
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    } else {
      setSelectedFile(null);
      setPreviewUrl(null);
      setUploadConfirmed(false);
      if (onImageSelect) onImageSelect(false);
    }
  };

  // Handles form submit (image + location upload)
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setErrorMsg('Please select an image to upload.');
      return;
    }

    setLoading(true);
    setErrorMsg(null);
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('latitude', location?.lat ?? 0);
    formData.append('longitude', location?.lng ?? 0);

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
    <div>
      {/* Drag and Drop Zone */}
      <div
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        style={{
          border: isDragging ? '3px dashed #667eea' : '3px dashed #d3d3d3',
          borderRadius: '20px',
          padding: '60px 40px',
          textAlign: 'center',
          background: isDragging 
            ? 'linear-gradient(135deg, rgba(102,126,234,0.08) 0%, rgba(118,75,162,0.08) 100%)'
            : 'linear-gradient(135deg, #f8f9ff 0%, #fef5ff 100%)',
          transition: 'all 0.3s ease',
          cursor: 'pointer',
          position: 'relative',
          boxShadow: isDragging ? '0 8px 30px rgba(102,126,234,0.15)' : 'inset 0 2px 4px rgba(0,0,0,0.04)'
        }}
      >
        {/* Upload Icon */}
        <div style={{
          fontSize: '4.5em',
          marginBottom: '20px',
          color: isDragging ? '#667eea' : '#b8b8b8',
          transition: 'all 0.3s ease',
          filter: isDragging ? 'drop-shadow(0 4px 12px rgba(102,126,234,0.3))' : 'none'
        }}>
          {isDragging ? '📥' : '🖼️'}
        </div>

        <h3 style={{
          color: '#1a202c',
          marginBottom: '12px',
          fontWeight: 700,
          fontSize: '1.4em',
          letterSpacing: '-0.025em'
        }}>
          {isDragging ? 'Drop your image here!' : 'Upload Waste Image'}
        </h3>
        
        <p style={{
          color: '#718096',
          marginBottom: '28px',
          fontSize: '1.05em'
        }}>
          Drag and drop or click to browse
        </p>

        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
          id="file-upload"
        />
        
        <label
          htmlFor="file-upload"
          style={{
            display: 'inline-block',
            padding: '14px 36px',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
            borderRadius: '12px',
            cursor: 'pointer',
            fontWeight: 700,
            fontSize: '1.05em',
            transition: 'all 0.3s ease',
            border: 'none',
            boxShadow: '0 4px 15px rgba(102,126,234,0.4)'
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = 'translateY(-2px)';
            e.currentTarget.style.boxShadow = '0 6px 20px rgba(102,126,234,0.6)';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 4px 15px rgba(102,126,234,0.4)';
          }}
        >
          Choose File
        </label>
      </div>

      {/* Upload Confirmation */}
      {uploadConfirmed && (
        <div style={{
          marginTop: '16px',
          padding: '14px 20px',
          background: '#f0fdf4',
          border: '1px solid #86efac',
          borderRadius: '8px',
          color: '#166534',
          fontSize: '0.95em',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          animation: 'fadeIn 0.3s ease'
        }}>
          <span style={{ fontSize: '1.2em' }}>✅</span>
          <span><strong>Image uploaded successfully!</strong> Ready to classify.</span>
        </div>
      )}

      {/* Upload Button */}
      {selectedFile && (
        <div style={{ marginTop: '28px', textAlign: 'center' }}>
          <button
            onClick={handleSubmit}
            disabled={loading}
            style={{
              padding: '15px 40px',
              fontSize: '1.1em',
              background: loading ? '#e2e8f0' : 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '12px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontWeight: 700,
              transition: 'all 0.3s ease',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '10px',
              boxShadow: loading ? 'none' : '0 4px 15px rgba(245,87,108,0.4)'
            }}
            onMouseOver={(e) => {
              if (!loading) {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 6px 20px rgba(245,87,108,0.6)';
              }
            }}
            onMouseOut={(e) => {
              if (!loading) {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 4px 15px rgba(245,87,108,0.4)';
              }
            }}
          >
            {loading ? (
              <>
                <div className="spinner" style={{ margin: 0, width: '18px', height: '18px', borderWidth: '2px' }}></div>
                Analyzing...
              </>
            ) : (
              <>
                <span>🚀</span>
                Classify Waste
              </>
            )}
          </button>
        </div>
      )}
      
      {/* Error Message */}
      {errorMsg && (
        <div style={{
          marginTop: '16px',
          padding: '14px 20px',
          background: '#fef2f2',
          borderRadius: '8px',
          color: '#991b1b',
          fontWeight: 500,
          fontSize: '0.95em',
          border: '1px solid #fecaca',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <span style={{ fontSize: '1.1em' }}>⚠️</span>
          {errorMsg}
        </div>
      )}

      {/* Loading Spinner */}
      {loading && (
        <div style={{
          marginTop: '32px',
          textAlign: 'center',
          padding: '48px 32px',
          background: '#ffffff',
          borderRadius: '12px',
          border: '1px solid #e2e8f0'
        }}>
          <div className="spinner" style={{ margin: '0 auto 20px', borderColor: '#5a67d8', borderRightColor: 'transparent' }}></div>
          <p style={{
            color: '#1a202c',
            fontWeight: 600,
            fontSize: '1.1em',
            marginBottom: '6px'
          }}>
            Analyzing your waste image...
          </p>
          <p style={{ color: '#718096', fontSize: '0.9em' }}>This may take a few moments</p>
        </div>
      )}

      {/* Image Previews Side by Side */}
      {(previewUrl || annotatedImage) && !loading && (
        <div style={{
          marginTop: '40px',
          display: 'grid',
          gridTemplateColumns: annotatedImage ? 'repeat(auto-fit, minmax(300px, 1fr))' : '1fr',
          gap: '28px'
        }}>
          {/* Original Preview */}
          {previewUrl && (
            <div className="fade-in" style={{
              background: '#ffffff',
              borderRadius: '16px',
              padding: '24px',
              boxShadow: '0 8px 30px rgba(0,0,0,0.1)'
            }}>
              <h4 style={{
                color: '#1a202c',
                marginBottom: '16px',
                fontSize: '1.1em',
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <span>🖼️</span> Original Image
              </h4>
              <div style={{
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                padding: '4px',
                borderRadius: '12px',
                boxShadow: '0 4px 15px rgba(102,126,234,0.3)'
              }}>
                <img
                  src={previewUrl}
                  alt="Preview"
                  style={{
                    width: '100%',
                    borderRadius: '8px',
                    display: 'block'
                  }}
                />
              </div>
            </div>
          )}

          {/* Annotated Result */}
          {annotatedImage && (
            <div className="fade-in" style={{
              background: '#ffffff',
              borderRadius: '16px',
              padding: '24px',
              boxShadow: '0 8px 30px rgba(0,0,0,0.1)'
            }}>
              <h4 style={{
                color: '#1a202c',
                marginBottom: '16px',
                fontSize: '1.1em',
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <span>✨</span> AI Classified Result
              </h4>
              <div style={{
                background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                padding: '4px',
                borderRadius: '12px',
                boxShadow: '0 4px 15px rgba(245,87,108,0.3)'
              }}>
                <img
                  src={annotatedImage}
                  alt="Annotated"
                  style={{
                    width: '100%',
                    borderRadius: '8px',
                    display: 'block'
                  }}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ImageUploader;
