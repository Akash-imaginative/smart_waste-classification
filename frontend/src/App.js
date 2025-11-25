import React, { useState } from "react";
import ImageUploader from "./components/ImageUploader";
import MapWithAutocomplete from "./components/MapWithAutocomplete";
import "./App.css";

function App() {
  const [results, setResults] = useState(null);
  const [location, setLocation] = useState(null);

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

  // Waste composition as styled table
  const CompositionTable = ({ composition }) => {
    const maxPercent = Math.max(...Object.values(composition));
    return (
      <div style={{ marginTop: "20px" }}>
        {Object.entries(composition).map(([category, percent], idx) => (
          <div
            key={category}
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
          }}>Waste Classification</h1>
          <p style={{
            color: "#718096",
            fontSize: "1.15em",
            margin: 0,
            fontWeight: 500
          }}>AI-Powered Waste Detection & Recycling Center Locator</p>
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
                  key={cat.name}
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
        />
      </div>

      {/* Nearest recycling centers listed */}
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
                key={center.name}
                style={{
                  ...listItemStyle,
                  marginBottom: "16px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  animation: `slideIn 0.5s ease ${idx * 0.15}s backwards`
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.transform = "translateX(6px)";
                  e.currentTarget.style.boxShadow = "0 8px 20px rgba(245,87,108,0.2)";
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.transform = "translateX(0)";
                  e.currentTarget.style.boxShadow = "0 2px 8px rgba(0,0,0,0.08)";
                }}
              >
                <div>
                  <div style={{ fontWeight: 700, fontSize: "1.1em", color: "#1a202c", marginBottom: "6px" }}>
                    {center.name}
                  </div>
                  <div style={{ color: "#718096", fontSize: "0.95em" }}>Click to view on map</div>
                </div>
                <div style={{
                  background: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
                  color: "white",
                  padding: "8px 18px",
                  borderRadius: "25px",
                  fontWeight: 700,
                  fontSize: "1em",
                  whiteSpace: "nowrap",
                  boxShadow: "0 4px 12px rgba(245,87,108,0.3)"
                }}>
                  {center.distance} km
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
