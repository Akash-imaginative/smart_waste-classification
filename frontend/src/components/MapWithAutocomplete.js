import React, { useState, useEffect } from "react";
import { MapContainer, TileLayer, Marker, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

function RecenterMap({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center) map.setView(center, map.getZoom(), { animate: true });
  }, [center, map]);
  return null;
}

function MapWithAutocomplete({ onLocationChange, location }) {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  // Map marker position is controlled by location state
  const markerPosition = location || { lat: 40.7128, lng: -74.006 };

  useEffect(() => {
    if (query.length < 3) {
      setSuggestions([]);
      setIsSearching(false);
      return;
    }

    const controller = new AbortController();
    setIsSearching(true);

    async function fetchSuggestions() {
      const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
        query
      )}&addressdetails=1&limit=5`;
      try {
        const res = await fetch(url, { signal: controller.signal });
        const data = await res.json();
        setSuggestions(data);
      } catch (e) {
        if (e.name !== "AbortError") console.error(e);
      } finally {
        setIsSearching(false);
      }
    }
    
    const timeoutId = setTimeout(fetchSuggestions, 300);
    return () => {
      clearTimeout(timeoutId);
      controller.abort();
    };
  }, [query]);

  const handleSelect = (place) => {
    const newPos = { lat: parseFloat(place.lat), lng: parseFloat(place.lon) };
    setQuery(place.display_name);
    setSuggestions([]);
    onLocationChange(newPos);
  };

  const handleMarkerDrag = (e) => {
    const latlng = e.target.getLatLng();
    onLocationChange(latlng);
  };

  return (
    <div style={{ position: "relative" }}>
      {/* Search Input Container */}
      <div
        style={{
          position: "absolute",
          zIndex: 1000,
          top: 10,
          left: "50%",
          transform: "translateX(-50%)",
          width: "90%",
          maxWidth: 400,
        }}
      >
        <div style={{ position: "relative" }}>
          <input
            type="text"
            value={query}
            placeholder="🔍 Search for a location..."
            onChange={(e) => setQuery(e.target.value)}
            style={{
              width: "100%",
              padding: "14px 45px 14px 20px",
              fontSize: "15px",
              backgroundColor: "white",
              borderRadius: "12px",
              border: "2px solid #e0e0e0",
              outline: "none",
              boxShadow: "0 4px 20px rgba(0, 0, 0, 0.1)",
              transition: "all 0.3s ease",
              fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
            }}
            onFocus={(e) => {
              e.target.style.border = "2px solid #667eea";
              e.target.style.boxShadow = "0 6px 30px rgba(102, 126, 234, 0.2)";
            }}
            onBlur={(e) => {
              e.target.style.border = "2px solid #e0e0e0";
              e.target.style.boxShadow = "0 4px 20px rgba(0, 0, 0, 0.1)";
            }}
          />
          
          {/* Loading Spinner */}
          {isSearching && (
            <div
              className="spinner"
              style={{
                position: "absolute",
                right: 15,
                top: "50%",
                transform: "translateY(-50%)",
                width: 18,
                height: 18,
                borderWidth: 2,
              }}
            />
          )}
        </div>

        {/* Suggestions Dropdown */}
        {suggestions.length > 0 && (
          <ul
            style={{
              maxHeight: 240,
              overflowY: "auto",
              background: "white",
              border: "none",
              listStyle: "none",
              margin: "8px 0 0 0",
              padding: 0,
              borderRadius: "12px",
              boxShadow: "0 8px 30px rgba(0, 0, 0, 0.15)",
              animation: "fadeIn 0.2s ease",
            }}
          >
            {suggestions.map((place, index) => (
              <li
                key={place.place_id}
                onClick={() => handleSelect(place)}
                style={{
                  padding: "14px 18px",
                  cursor: "pointer",
                  borderBottom: index < suggestions.length - 1 ? "1px solid #f0f0f0" : "none",
                  transition: "all 0.2s ease",
                  fontSize: "14px",
                  color: "#333",
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                }}
                onMouseEnter={(e) => {
                  e.target.style.background = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)";
                  e.target.style.color = "white";
                  e.target.style.paddingLeft = "22px";
                }}
                onMouseLeave={(e) => {
                  e.target.style.background = "white";
                  e.target.style.color = "#333";
                  e.target.style.paddingLeft = "18px";
                }}
              >
                <span style={{ fontSize: "16px" }}>📍</span>
                <span style={{ flex: 1, lineHeight: 1.4 }}>{place.display_name}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Map Container */}
      <MapContainer
        center={markerPosition}
        zoom={13}
        scrollWheelZoom
        style={{
          height: 500,
          width: "100%",
          borderRadius: "16px",
          overflow: "hidden",
          boxShadow: "0 8px 30px rgba(0, 0, 0, 0.12)",
        }}
        key={`${markerPosition.lat}-${markerPosition.lng}`}
      >
        <RecenterMap center={markerPosition} />
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <Marker position={markerPosition} draggable eventHandlers={{ dragend: handleMarkerDrag }} />
      </MapContainer>

      {/* Instructions */}
      <div
        style={{
          marginTop: "16px",
          padding: "12px",
          background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
          borderRadius: "12px",
          fontSize: "13px",
          color: "#555",
          textAlign: "center",
          fontStyle: "italic",
        }}
      >
        💡 <strong>Tip:</strong> Search for a location or drag the marker to set your position
      </div>
    </div>
  );
}

export default MapWithAutocomplete;
