import React, { useState } from "react";
import ImageUploader from "./components/ImageUploader";
import MapWithAutocomplete from "./components/MapWithAutocomplete";

function App() {
  const [results, setResults] = useState(null);
  const [location, setLocation] = useState(null);

  const handleLocationChange = (loc) => {
    setLocation(loc);
  };

  const handleSubmit = () => {
    if (location) {
      alert(`Submitting location: lat ${location.lat}, lng ${location.lng}`);
      // Trigger backend or image upload with location info
    } else {
      alert("Please select a location first");
    }
  };

  const containerStyle = {
    padding: "20px",
    maxWidth: "900px",
    margin: "auto",
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    backgroundColor: "#f9f9f9",
    borderRadius: "10px",
    boxShadow: "0 0 15px rgba(0,0,0,0.1)",
  };

  const sectionTitleStyle = {
    color: "#333",
    marginBottom: "15px",
    borderBottom: "2px solid #ddd",
    paddingBottom: "5px",
  };

  const listStyle = {
    listStyleType: "none",
    padding: 0,
  };

  const listItemStyle = {
    marginBottom: "8px",
    padding: "10px",
    backgroundColor: "#fff",
    borderRadius: "6px",
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
  };

  const buttonStyle = {
    marginTop: "20px",
    padding: "12px 30px",
    fontSize: "16px",
    backgroundColor: "#007bff",
    color: "white",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
  };

  const buttonHoverStyle = {
    backgroundColor: "#0056b3",
  };

  return (
    <div style={containerStyle}>
      <h1 style={{ textAlign: "center", color: "#222" }}>Waste Classification</h1>

      <ImageUploader location={location} onResults={setResults} />

      {results && results.top_categories && (
        <div style={{ marginTop: "25px" }}>
          <h3 style={sectionTitleStyle}>Top Waste Categories</h3>
          <ul style={listStyle}>
            {results.top_categories.map((cat) => (
              <li key={cat.name} style={listItemStyle}>
                <strong>{cat.name}</strong>: {cat.percentage}%
              </li>
            ))}
          </ul>
        </div>
      )}

      <div style={{ marginTop: "30px" }}>
        <h3 style={sectionTitleStyle}>Select Location</h3>
        <MapWithAutocomplete onLocationChange={handleLocationChange} location={location} />
      </div>

      <button
        style={buttonStyle}
        onClick={handleSubmit}
        onMouseOver={(e) => (e.target.style.backgroundColor = buttonHoverStyle.backgroundColor)}
        onMouseOut={(e) => (e.target.style.backgroundColor = buttonStyle.backgroundColor)}
      >
        Submit
      </button>

      {location && results?.nearest_centers && (
        <div style={{ marginTop: "30px" }}>
          <h3 style={sectionTitleStyle}>Nearest Recycling Centers</h3>
          <ul style={listStyle}>
            {results.nearest_centers.map((center) => (
              <li key={center.name} style={listItemStyle}>
                {center.name} - {center.distance} km
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
