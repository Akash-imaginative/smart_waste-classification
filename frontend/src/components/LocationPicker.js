import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

// Fix Leaflet icon bug with React+Webpack
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

function RecenterMap({ lat, lng }) {
  const map = useMap();
  useEffect(() => {
    if (lat && lng) {
      map.setView([lat, lng], map.getZoom(), { animate: true });
    }
  }, [lat, lng, map]);
  return null;
}

function LocationPicker({ location, onLocationChange }) {
  const [markerPos, setMarkerPos] = useState(location || { lat: 40.7128, lng: -74.006 });

  useEffect(() => {
    if (location) {
      setMarkerPos(location);
    }
  }, [location]);

  const handleDragEnd = (e) => {
    const latlng = e.target.getLatLng();
    setMarkerPos(latlng);
    onLocationChange(latlng);
  };

  return (
    <MapContainer center={markerPos} zoom={13} scrollWheelZoom style={{ height: '400px', width: '100%' }}>
      <RecenterMap lat={markerPos.lat} lng={markerPos.lng} />
      <TileLayer
        attribution='&copy; OpenStreetMap contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <Marker position={markerPos} draggable onDragend={handleDragEnd} />
    </MapContainer>
  );
}

export default LocationPicker;
