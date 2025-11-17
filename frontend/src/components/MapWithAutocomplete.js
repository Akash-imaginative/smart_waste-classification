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

  // Map marker position is controlled by location state
  const markerPosition = location || { lat: 40.7128, lng: -74.006 };

  useEffect(() => {
    if (query.length < 3) return setSuggestions([]);

    const controller = new AbortController();

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
      }
    }
    fetchSuggestions();
    return () => controller.abort();
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
      <input
        type="text"
        value={query}
        placeholder="Start typing an address..."
        onChange={(e) => setQuery(e.target.value)}
        style={{
          position: "absolute",
          zIndex: 1000,
          width: 300,
          padding: 8,
          margin: 10,
          backgroundColor: "white",
          borderRadius: 4,
          border: "1px solid #ccc",
          top: 0,
          left: "50%",
          transform: "translateX(-50%)",
        }}
      />
      {suggestions.length > 0 && (
        <ul
          style={{
            position: "absolute",
            zIndex: 1000,
            width: 300,
            maxHeight: 200,
            overflowY: "auto",
            background: "white",
            border: "1px solid #ccc",
            listStyle: "none",
            margin: 0,
            padding: 0,
            top: 40,
            left: "50%",
            transform: "translateX(-50%)",
          }}
        >
          {suggestions.map((place) => (
            <li
              key={place.place_id}
              onClick={() => handleSelect(place)}
              style={{ padding: 8, cursor: "pointer" }}
            >
              {place.display_name}
            </li>
          ))}
        </ul>
      )}

      <MapContainer
        center={markerPosition}
        zoom={13}
        scrollWheelZoom
        style={{ height: 500, width: "100%", marginTop: 60 }}
        key={`${markerPosition.lat}-${markerPosition.lng}`} // Forces rerender on location change
      >
        <RecenterMap center={markerPosition} />
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <Marker position={markerPosition} draggable eventHandlers={{ dragend: handleMarkerDrag }} />
      </MapContainer>
    </div>
  );
}

export default MapWithAutocomplete;
