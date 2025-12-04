import React, { useState } from "react";
import ImageUploader from "./components/ImageUploader";
import MapWithAutocomplete from "./components/MapWithAutocomplete";
import "./App.css";

function App() {
  const [results, setResults] = useState(null);
  const [location, setLocation] = useState(null);
  const [loadingCenters, setLoadingCenters] = useState(false);
  const [centersError, setCentersError] = useState(null);
 

  // Modern color and layout styles
  const containerStyle = {
    padding: "40px 20px",
    maxWidth: "1200px",
    margin: "0 auto",
    minHeight: "100vh",
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
  };

  const sectionTitleStyle = {
    color: "#1a202c",
    marginBottom: "20px",
    fontSize: "1.5em",
    fontWeight: 700,
    letterSpacing: "-0.025em",
    paddingBottom: "12px",
    borderBottom: "3px solid",
    borderImage: "linear-gradient(90deg, #667eea, #764ba2) 1"
  };

  const listItemStyle = {
    marginBottom: "12px",
    padding: "18px 24px",
    background: "#ffffff",
    borderRadius: "12px",
    fontSize: "1em",
    border: "none",
    boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
    transition: "all 0.3s ease",
    cursor: "pointer"
  };

  // Main submit action (if you want to perform more with map location)
  const handleLocationChange = (loc) => setLocation(loc);

  // Function to find recycling centers independently
  const handleFindCenters = async () => {
    if (!location) {
      setCentersError('Please select a location first');
      return;
    }

    setLoadingCenters(true);
    setCentersError(null);

    try {
      const backendUrl = process.env.REACT_APP_BACKEND_API_URL || 'http://localhost:8000';
      const response = await fetch(`${backendUrl}/api/find-centers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          latitude: location.lat,
          longitude: location.lng
        })
      });

      if (!response.ok) {
        throw new Error('Failed to fetch recycling centers');
      }

      const data = await response.json();
      setResults(prev => ({
        ...prev,
        nearest_centers: data.nearest_centers
      }));
    } catch (error) {
      setCentersError('Error finding recycling centers. Please try again.');
      console.error(error);
    } finally {
      setLoadingCenters(false);
    }
  };

  // Waste composition as styled table
  const CompositionTable = ({ composition }) => {
  const maxPercent = Math.max(...Object.values(composition));
  return (
    <div style={{ marginTop: "20px" }}>
      {Object.entries(composition).map(([category, percent], idx) => (
        <div
          key={`${category}-${idx}`}
          style={{
            marginBottom: "20px",
            animation: `slideIn 0.5s ease ${idx * 0.1}s backwards`
          }}
        >
            <div style={{
              display: "flex",
              justifyContent: "space-between",
              marginBottom: "10px",
              alignItems: "center"
            }}>
              <span style={{
                fontWeight: 700,
                fontSize: "1.05em",
                color: "#1a202c",
                textTransform: "capitalize"
              }}>{category}</span>
              <span style={{
                fontWeight: 700,
                fontSize: "1.15em",
                color: "#667eea"
              }}>{percent}%</span>
            </div>
            <div style={{
              height: "20px",
              background: "#f0f4ff",
              borderRadius: "10px",
              overflow: "hidden",
              boxShadow: "inset 0 2px 4px rgba(0,0,0,0.06)"
            }}>
              <div style={{
                height: "100%",
                width: `${percent}%`,
                background: percent === maxPercent 
                  ? "linear-gradient(90deg, #f093fb 0%, #f5576c 100%)"
                  : "linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
                borderRadius: "10px",
                transition: "width 1s ease",
                boxShadow: "0 2px 8px rgba(102, 126, 234, 0.4)"
              }} />
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div style={containerStyle}>
      <div style={{
        background: "#ffffff",
        borderRadius: "20px",
        padding: "40px",
        boxShadow: "0 20px 60px rgba(0,0,0,0.15)",
        marginBottom: "30px"
      }}>
        <div style={{
          textAlign: "center",
          marginBottom: "40px"
        }}>
          <div style={{
            fontSize: "4em",
            marginBottom: "16px",
            filter: "drop-shadow(0 4px 8px rgba(102,126,234,0.3))"
          }}>♻️</div>
          <h1 style={{
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
            fontWeight: 800,
            fontSize: "2.75em",
            margin: "0 0 12px 0",
            letterSpacing: "-0.025em"
          }}>Ecoscan</h1>
          <p style={{
            color: "#718096",
            fontSize: "1.15em",
            margin: 0,
            fontWeight: 500
          }}>Smart Waste Classification & Recycling Center Locator</p>
        </div>

        {/* Image Upload component handles file, location, results */}
        <ImageUploader location={location} onResults={setResults} />

        {/* Composition percentages as a colored table */}
        {results?.composition && Object.keys(results.composition).length > 0 && (
          <div style={{ marginTop: "40px" }} className="fade-in">
            <h3 style={sectionTitleStyle}>📊 Waste Composition (%)</h3>
            <CompositionTable composition={results.composition} />
          </div>
        )}

        {/* Raw category counts */}
        {results?.classifications && results.classifications.length > 0 && (
          <div style={{ marginTop: "40px" }} className="fade-in">
            <h3 style={sectionTitleStyle}>🎯 Classification Details</h3>
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
              gap: "16px"
            }}>
              {results.classifications.map((cat, idx) => (
                <div
                  key={`${cat.name}-${idx}`}
                  style={{
                    ...listItemStyle,
                    animation: `slideIn 0.5s ease ${idx * 0.1}s backwards`
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.transform = "translateY(-4px)";
                    e.currentTarget.style.boxShadow = "0 8px 20px rgba(102,126,234,0.2)";
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.transform = "translateY(0)";
                    e.currentTarget.style.boxShadow = "0 2px 8px rgba(0,0,0,0.08)";
                  }}
                >
                  <div style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center"
                  }}>
                    <span style={{ 
                      color: "#1a202c",
                      fontWeight: 600,
                      fontSize: "1.05em",
                      textTransform: "capitalize"
                    }}>{cat.name}</span>
                    <span style={{
                      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                      color: "white",
                      padding: "6px 16px",
                      borderRadius: "20px",
                      fontWeight: 700,
                      fontSize: "1em",
                      boxShadow: "0 4px 12px rgba(102,126,234,0.3)"
                    }}>{cat.count}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>

      {/* Map selector and nearest centers */}
      <div style={{
        background: "#ffffff",
        borderRadius: "20px",
        padding: "40px",
        boxShadow: "0 20px 60px rgba(0,0,0,0.15)",
        marginTop: "30px"
      }}>
        <h3 style={sectionTitleStyle}>🗺️ Select Location</h3>
        <p style={{
          color: "#718096",
          marginBottom: "24px",
          fontSize: "1.05em"
        }}>Choose your location to find nearby recycling centers</p>
        <MapWithAutocomplete
          onLocationChange={handleLocationChange}
          location={location}
          onConfirmLocation={handleLocationChange}
        />
        
        {/* Find Centers Button */}
        {location && (
          <div style={{ marginTop: '24px', textAlign: 'center' }}>
            <button
              onClick={handleFindCenters}
              disabled={loadingCenters}
              style={{
                padding: '14px 32px',
                fontSize: '1.05em',
                background: loadingCenters ? '#e2e8f0' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                border: 'none',
                borderRadius: '12px',
                cursor: loadingCenters ? 'not-allowed' : 'pointer',
                fontWeight: 700,
                transition: 'all 0.3s ease',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '10px',
                boxShadow: loadingCenters ? 'none' : '0 4px 15px rgba(102,126,234,0.4)'
              }}
              onMouseOver={(e) => {
                if (!loadingCenters) {
                  e.currentTarget.style.transform = 'translateY(-2px)';
                  e.currentTarget.style.boxShadow = '0 6px 20px rgba(102,126,234,0.6)';
                }
              }}
              onMouseOut={(e) => {
                if (!loadingCenters) {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 4px 15px rgba(102,126,234,0.4)';
                }
              }}
            >
              {loadingCenters ? (
                <>
                  <div className="spinner" style={{ margin: 0, width: '16px', height: '16px', borderWidth: '2px' }}></div>
                  Searching...
                </>
              ) : (
                <>
                  <span>📍</span>
                  Find Recycling Centers
                </>
              )}
            </button>
          </div>
        )}

        {/* Error Message */}
        {centersError && (
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
            {centersError}
          </div>
        )}
      </div>

      {location && results?.nearest_centers && results.nearest_centers.length > 0 && (
  <div style={{
    background: "#ffffff",
    borderRadius: "20px",
    padding: "40px",
    boxShadow: "0 20px 60px rgba(0,0,0,0.15)",
    marginTop: "30px"
  }} className="fade-in">
    <h3 style={sectionTitleStyle}>📍 Nearest Recycling Centers</h3>
    <div style={{ marginTop: "20px" }}>
      {results.nearest_centers.map((center, idx) => (
        <div 
          key={`${center.name}-${idx}`}
          style={{
            ...listItemStyle,
            marginBottom: "20px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            gap: "20px",
            animation: `slideIn 0.5s ease ${idx * 0.15}s backwards`
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = "translateX(6px)";
            e.currentTarget.style.boxShadow = "0 8px 20px rgba(66,153,225,0.2)";
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = "translateX(0)";
            e.currentTarget.style.boxShadow = "0 2px 8px rgba(0,0,0,0.08)";
          }}
        >
          <div style={{ flex: 1, minWidth: 0, paddingRight: "20px" }}>
            {/* Center Name */}
            <div style={{ 
              fontWeight: 700, 
              fontSize: "1.15em", 
              color: "#1a202c", 
              marginBottom: "10px",
              lineHeight: "1.3"
            }}>
              {center.name}
            </div>
            
            {/* 🆕 FULL NOMINATIM ADDRESS */}
            {center.full_address && (
              <div style={{
                fontSize: "0.92em",
                color: "#2d3748",
                background: "linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%)",
                padding: "12px 16px",
                borderRadius: "12px",
                borderLeft: "4px solid #4299e1",
                lineHeight: "1.5",
                boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
                maxWidth: "100%"
              }}>
                📍 {center.full_address}
              </div>
            )}
          </div>
          
          {/* Distance Badge */}
          <div style={{
            background: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
            color: "white",
            padding: "12px 24px",
            borderRadius: "30px",
            fontWeight: 700,
            fontSize: "1.1em",
            whiteSpace: "nowrap",
            boxShadow: "0 6px 16px rgba(245,87,108,0.3)",
            flexShrink: 0,
            minWidth: "100px",
            textAlign: "center",
            alignSelf: "center"
          }}>
            {typeof center.distance === 'number' ? center.distance.toFixed(2) : center.distance} km
          </div>
        </div>
      ))}
    </div>
  </div>
)}
    </div>
  );
}

export default App;
