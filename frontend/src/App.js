import React, { useState } from "react";
import ImageUploader from "./components/ImageUploader";
import MapWithAutocomplete from "./components/MapWithAutocomplete";

function App() {
  const [results, setResults] = useState(null);
  const [location, setLocation] = useState(null);

  // Modern color and layout styles
  const containerStyle = {
    padding: "30px",
    maxWidth: "980px",
    margin: "36px auto",
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    background: "linear-gradient(90deg,#e8f7fc 0%,#fafdfe 84%)",
    borderRadius: "18px",
    boxShadow: "0 0 24px rgba(60,183,217,0.14)"
  };

  const sectionTitleStyle = {
    color: "#226ba0",
    marginBottom: "15px",
    borderBottom: "2px solid #8ee4f7",
    paddingBottom: "7px",
    letterSpacing: "0.5px",
    fontSize: "1.25em",
    fontWeight: 600
  };

  const listStyle = {
    listStyleType: "none",
    padding: 0,
    marginTop: "7px",
    marginBottom: 0
  };

  const listItemStyle = {
    marginBottom: "11px",
    padding: "15px",
    background: "linear-gradient(90deg,#baf1ef 0%,#d8fff6 100%)",
    borderRadius: "9px",
    fontSize: "1em",
    boxShadow: "0 1px 2px rgba(60,183,217,0.06)"
  };

  const buttonStyle = {
    marginTop: "32px",
    padding: "16px 42px",
    fontSize: "17px",
    background: "linear-gradient(90deg,#7fd8dd 0%,#67bee3 100%)",
    color: "#fff",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    boxShadow: "0 2px 8px rgba(60,183,217,0.13)",
    fontWeight: 600,
    transition: "background 0.18s"
  };

  const buttonHoverStyle = { background: "#45b3c7" };

  // Main submit action (if you want to perform more with map location)
  const handleLocationChange = (loc) => setLocation(loc);
  const handleSubmit = () => {
    if (location) {
      alert(`Location: lat ${location.lat}, lng ${location.lng}`);
    } else {
      alert("Please select a location first");
    }
  };

  // Waste composition as styled table
  const CompositionTable = ({ composition }) => (
    <table style={{
      width: "100%",
      borderCollapse: "collapse",
      background: "#e8f7fc",
      borderRadius: "8px",
      boxShadow: "0 1px 8px rgba(60,183,217,0.09)",
      marginTop: "4px"
    }}>
      <thead>
        <tr>
          <th style={{
            padding: "11px",
            background: "#32b6e0",
            color: "#fff",
            fontWeight: 600,
            fontSize: "1em",
            borderTopLeftRadius: "8px"
          }}>Category</th>
          <th style={{
            padding: "11px",
            background: "#32b6e0",
            color: "#fff",
            fontWeight: 600,
            fontSize: "1em",
            borderTopRightRadius: "8px"
          }}>Composition (%)</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(composition).map(([category, percent]) => (
          <tr key={category} style={{ background: "#d2f5ff" }}>
            <td style={{ padding: "12px", borderBottom: "1px solid #c7e5f0" }}>{category}</td>
            <td style={{ padding: "12px", borderBottom: "1px solid #c7e5f0" }}>{percent}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  );

  return (
    <div style={containerStyle}>
      <h1 style={{
        textAlign: "center",
        color: "#1881b2",
        letterSpacing: "1px",
        fontWeight: 700,
        fontSize: "2.1em",
        marginBottom: "26px"
      }}>Waste Classification</h1>

      {/* Image Upload component handles file, location, results */}
      <ImageUploader location={location} onResults={setResults} />

      {/* Annotated waste image preview */}
      {results?.annotated_image_url && (
        <div style={{ marginTop: "38px" }}>
          <h3 style={sectionTitleStyle}>Annotated Waste Image</h3>
          <img
            src={results.annotated_image_url}
            alt="Annotated upload"
            style={{
              maxWidth: "100%",
              borderRadius: "11px",
              boxShadow: "0 2px 18px rgba(24,129,178,0.13)"
            }}
          />
        </div>
      )}

      {/* Composition percentages as a colored table */}
      {results?.composition && Object.keys(results.composition).length > 0 && (
        <div style={{ marginTop: "28px" }}>
          <h3 style={sectionTitleStyle}>Waste Composition (%)</h3>
          <CompositionTable composition={results.composition} />
        </div>
      )}

      {/* Raw category counts */}
      {results?.classifications && results.classifications.length > 0 && (
        <div style={{ marginTop: "28px" }}>
          <h3 style={sectionTitleStyle}>Classified Waste Counts</h3>
          <ul style={listStyle}>
            {results.classifications.map((cat) => (
              <li key={cat.name} style={listItemStyle}>
                <strong style={{ color: "#1a8890" }}>{cat.name}</strong>: {cat.count}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Map selector and nearest centers */}
      <div style={{ marginTop: "35px" }}>
        <h3 style={sectionTitleStyle}>Select Location</h3>
        <MapWithAutocomplete
          onLocationChange={handleLocationChange}
          location={location}
        />
      </div>

      <button
        style={buttonStyle}
        onClick={handleSubmit}
        onMouseOver={e => (e.target.style.background = buttonHoverStyle.background)}
        onMouseOut={e => (e.target.style.background = buttonStyle.background)}
      >
        Submit
      </button>

      {/* Nearest recycling centers listed */}
      {location && results?.nearest_centers && (
        <div style={{ marginTop: "38px" }}>
          <h3 style={sectionTitleStyle}>Nearest Recycling Centers</h3>
          <ul style={listStyle}>
            {results.nearest_centers.map((center) => (
              <li key={center.name} style={listItemStyle}>
                {center.name} <span style={{ color: "#2e7fa4" }}>({center.distance} km)</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
