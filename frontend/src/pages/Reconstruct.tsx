import { useState, useRef, useEffect } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Button } from '../components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { Input, Textarea } from '../components/ui/Input';
import { cn } from '../lib/utils';
import { api, type GPSPoint, type ReconstructionRequest, type ReconstructedRoute } from '../lib/api';
import {
  MapPin,
  Upload,
  Download,
  Trash2,
  Play,
  Loader2,
  AlertCircle,
  CheckCircle,
  Settings,
  Eye,
  Layers,
} from 'lucide-react';

// Fix Leaflet marker icon issue
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

interface ReconstructPageState {
  // Input
  inputMethod: 'manual' | 'file' | 'json';
  gpsPoints: GPSPoint[];
  jsonInput: string;
  fileError: string | null;

  // Options
  candidateRadius: number;
  maxCandidates: number;
  transitionPenalty: number;
  includeDiagnostics: boolean;
  includeGeometry: boolean;

  // Output
  isLoading: boolean;
  error: string | null;
  route: ReconstructedRoute | null;
  diagnostics: any | null;
  requestId: string | null;

  // Map
  map: L.Map | null;
  gpsMarkers: L.CircleMarker[];
  routeLayer: L.Polyline | null;
  candidateLayers: L.LayerGroup | null;
  showGPSPoints: boolean;
  showCandidates: boolean;
}

const initialState: ReconstructPageState = {
  inputMethod: 'manual',
  gpsPoints: [],
  jsonInput: '',
  fileError: null,
  candidateRadius: 50,
  maxCandidates: 10,
  transitionPenalty: 1.0,
  includeDiagnostics: true,
  includeGeometry: true,
  isLoading: false,
  error: null,
  route: null,
  diagnostics: null,
  requestId: null,
  map: null,
  gpsMarkers: [],
  routeLayer: null,
  candidateLayers: null,
  showGPSPoints: true,
  showCandidates: false,
};

export function Reconstruct() {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<ReconstructPageState>(initialState);

  // Initialize map
  useEffect(() => {
    if (!mapContainerRef.current || state.map) return;

    const map = L.map(mapContainerRef.current, {
      center: [37.7749, -122.4194],
      zoom: 13,
      zoomControl: false,
      attributionControl: false,
    });

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 19,
    }).addTo(map);

    // Add zoom control to top-right
    L.control.zoom({ position: 'topright' }).addTo(map);

    setState(prev => ({ ...prev, map, candidateLayers: L.layerGroup().addTo(map) }));

    return () => {
      map.remove();
      setState(prev => ({ ...prev, map: null }));
    };
  }, []);

  // Update map when route changes
  useEffect(() => {
    if (!state.map || !state.route) return;

    const { route, map, gpsMarkers, routeLayer, candidateLayers } = state;

    // Clear existing layers
    gpsMarkers.forEach(m => map.removeLayer(m));
    if (routeLayer) map.removeLayer(routeLayer);
    candidateLayers?.clearLayers();

    const newMarkers: L.CircleMarker[] = [];
    const coords = route.coordinates;

    if (state.showGPSPoints && state.gpsPoints.length > 0) {
      // Add GPS point markers
      state.gpsPoints.forEach((point, index) => {
        const marker = L.circleMarker([point.latitude, point.longitude], {
          radius: 6,
          fillColor: '#00d4aa',
          color: '#0a0a0f',
          weight: 2,
          fillOpacity: 0.9,
        }).bindPopup(`
          <strong>GPS Point ${index + 1}</strong><br/>
          Lat: ${point.latitude.toFixed(6)}<br/>
          Lon: ${point.longitude.toFixed(6)}<br/>
          Time: ${new Date(point.timestamp).toLocaleTimeString()}<br/>
          ${point.speed ? `Speed: ${point.speed.toFixed(1)} m/s` : ''}
        `);
        newMarkers.push(marker);
        marker.addTo(map);
      });
    }

    // Add route polyline
    if (coords.length >= 2) {
      const latLngs = coords.map(([lon, lat]) => [lat, lon] as L.LatLngExpression);
      const newRouteLayer = L.polyline(latLngs, {
        color: '#00d4aa',
        weight: 4,
        opacity: 0.9,
        lineCap: 'round',
        lineJoin: 'round',
      }).addTo(map);

      // Fit bounds to route
      const bounds = L.latLngBounds(latLngs);
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 16 });

      setState(prev => ({
        ...prev,
        gpsMarkers: newMarkers,
        routeLayer: newRouteLayer,
      }));
    }
  }, [state.route, state.showGPSPoints, state.gpsPoints]);

  const handleFileUpload = async (file: File) => {
    setState(prev => ({ ...prev, fileError: null }));
    try {
      const points = await api.parseGPSFile(file);
      setState(prev => ({ ...prev, gpsPoints: points, inputMethod: 'file' }));
    } catch (err) {
      setState(prev => ({ ...prev, fileError: err instanceof Error ? err.message : 'Failed to parse file' }));
    }
  };

  const handleJSONInput = () => {
    try {
      const points = api.parseJSON(state.jsonInput);
      setState(prev => ({ ...prev, gpsPoints: points, fileError: null }));
    } catch (err) {
      setState(prev => ({ ...prev, fileError: err instanceof Error ? err.message : 'Invalid JSON' }));
    }
  };

  const addManualPoint = () => {
    setState(prev => ({
      ...prev,
      gpsPoints: [
        ...prev.gpsPoints,
        {
          latitude: 37.7749 + Math.random() * 0.01,
          longitude: -122.4194 + Math.random() * 0.01,
          timestamp: new Date(Date.now() + prev.gpsPoints.length * 10000).toISOString(),
        }
      ]
    }));
  };

  const removePoint = (index: number) => {
    setState(prev => ({
      ...prev,
      gpsPoints: prev.gpsPoints.filter((_, i) => i !== index)
    }));
  };

  const updatePoint = (index: number, field: keyof GPSPoint, value: string | number) => {
    setState(prev => ({
      ...prev,
      gpsPoints: prev.gpsPoints.map((p, i) =>
        i === index ? { ...p, [field]: typeof value === 'string' && field !== 'timestamp' ? parseFloat(value) : value } : p
      )
    }));
  };

  const handleReconstruct = async () => {
    if (state.gpsPoints.length < 2) {
      setState(prev => ({ ...prev, error: 'At least 2 GPS points required' }));
      return;
    }

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const request: ReconstructionRequest = {
        gps_points: state.gpsPoints,
        options: {
          candidate_radius_m: state.candidateRadius,
          max_candidates_per_point: state.maxCandidates,
          transition_penalty: state.transitionPenalty,
          include_diagnostics: state.includeDiagnostics,
          include_geometry: state.includeGeometry,
        },
      };

      const response = await api.reconstructRoute(request);

      setState(prev => ({
        ...prev,
        isLoading: false,
        route: response.route || null,
        diagnostics: response.diagnostics || null,
        requestId: response.request_id,
        error: response.success ? null : response.error || 'Reconstruction failed',
      }));
    } catch (err) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: err instanceof Error ? err.message : 'Reconstruction failed',
      }));
    }
  };

  const handleClear = () => {
    setState(prev => ({
      ...prev,
      gpsPoints: [],
      route: null,
      diagnostics: null,
      requestId: null,
      error: null,
      jsonInput: '',
      fileError: null,
    }));
    if (state.routeLayer) {
      state.map?.removeLayer(state.routeLayer);
    }
    state.gpsMarkers.forEach(m => state.map?.removeLayer(m));
    setState(prev => ({ ...prev, gpsMarkers: [], routeLayer: null }));
    state.candidateLayers?.clearLayers();
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(1)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] flex">
      {/* Sidebar */}
      <aside className="w-full lg:w-96 flex-shrink-0 border-r border-border-primary bg-bg-secondary/50 p-4 lg:p-6 overflow-y-auto">
        <div className="space-y-6">
          {/* Input Method Tabs */}
          <div className="flex gap-1 bg-bg-tertiary rounded-lg p-1" role="tablist">
            {[
              { id: 'manual', label: 'Manual', icon: MapPin },
              { id: 'file', label: 'File Upload', icon: Upload },
              { id: 'json', label: 'JSON Input', icon: Settings },
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = state.inputMethod === tab.id;
              return (
                <button
                  key={tab.id}
                  role="tab"
                  aria-selected={isActive}
                  onClick={() => setState(prev => ({ ...prev, inputMethod: tab.id as any, fileError: null }))}
                  className={cn(
                    'flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all',
                    isActive
                      ? 'bg-bg-card text-text-primary shadow-sm'
                      : 'text-text-secondary hover:text-text-primary'
                  )}
                >
                  <Icon className="w-4 h-4" aria-hidden="true" />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* Input Forms */}
          <Card variant="glass">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MapPin className="w-5 h-5 text-accent-primary" />
                GPS Points ({state.gpsPoints.length})
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {state.inputMethod === 'manual' && (
                <div className="space-y-3">
                  {state.gpsPoints.map((point, index) => (
                    <div key={index} className="flex items-center gap-2 p-3 bg-bg-tertiary rounded-lg">
                      <span className="text-xs text-text-muted w-6 text-center">{index + 1}</span>
                      <Input
                        type="number"
                        step="0.000001"
                        placeholder="Latitude"
                        value={point.latitude}
                        onChange={(e) => updatePoint(index, 'latitude', e.target.value)}
                        className="w-24"
                        aria-label={`Point ${index + 1} latitude`}
                      />
                      <Input
                        type="number"
                        step="0.000001"
                        placeholder="Longitude"
                        value={point.longitude}
                        onChange={(e) => updatePoint(index, 'longitude', e.target.value)}
                        className="w-24"
                        aria-label={`Point ${index + 1} longitude`}
                      />
                      <Input
                        type="datetime-local"
                        value={point.timestamp.replace('Z', '').replace('T', 'T').slice(0, 16)}
                        onChange={(e) => updatePoint(index, 'timestamp', new Date(e.target.value).toISOString())}
                        className="flex-1"
                        aria-label={`Point ${index + 1} timestamp`}
                      />
                      <button
                        onClick={() => removePoint(index)}
                        className="p-1.5 text-text-muted hover:text-error hover:bg-error/10 rounded transition-colors"
                        aria-label={`Remove point ${index + 1}`}
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                  {state.gpsPoints.length < 10 && (
                    <Button variant="secondary" size="sm" onClick={addManualPoint} className="w-full">
                      <MapPin className="w-4 h-4" />
                      Add Point
                    </Button>
                  )}
                </div>
              )}

              {state.inputMethod === 'file' && (
                <div className="space-y-3">
                  <div
                    className="border-2 border-dashed border-border-primary rounded-lg p-8 text-center hover:border-accent-primary/50 transition-colors"
                    onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add('border-accent-primary'); }}
                    onDragLeave={(e) => { e.currentTarget.classList.remove('border-accent-primary'); }}
                    onDrop={(e) => {
                      e.preventDefault();
                      e.currentTarget.classList.remove('border-accent-primary');
                      const file = e.dataTransfer.files[0];
                      if (file) handleFileUpload(file);
                    }}
                  >
                    <Upload className="w-10 h-10 mx-auto text-text-muted mb-3" />
                    <p className="text-text-secondary mb-1">Drag & drop GPS file here</p>
                    <p className="text-xs text-text-muted">CSV, GPX, or JSON</p>
                    <input
                      type="file"
                      accept=".csv,.gpx,.json"
                      className="hidden"
                      id="file-upload"
                      onChange={(e) => { const file = e.target.files?.[0]; if (file) handleFileUpload(file); }}
                    />
                    <label htmlFor="file-upload" className="mt-3 inline-flex items-center gap-2 text-sm text-accent-primary hover:underline cursor-pointer">
                      <Upload className="w-4 h-4" />
                      Browse files
                    </label>
                  </div>

                  {state.fileError && (
                    <div className="p-3 bg-error/10 border border-error/30 rounded-lg text-error text-sm flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 flex-shrink-0" />
                      {state.fileError}
                    </div>
                  )}

                  {state.gpsPoints.length > 0 && (
                    <div className="text-sm text-text-secondary">
                      Parsed {state.gpsPoints.length} points from file
                    </div>
                  )}
                </div>
              )}

              {state.inputMethod === 'json' && (
                <div className="space-y-3">
                  <Textarea
                    value={state.jsonInput}
                    onChange={(e) => setState(prev => ({ ...prev, jsonInput: e.target.value }))}
                    placeholder='{"points": [{"latitude": 37.7749, "longitude": -122.4194, "timestamp": "2024-01-15T10:00:00"}, ...]}'
                    rows={8}
                    className="font-mono text-xs"
                    aria-label="JSON input"
                  />
                  <div className="flex gap-2">
                    <Button variant="secondary" size="sm" onClick={handleJSONInput}>
                      Parse JSON
                    </Button>
                    <Button variant="ghost" size="sm" onClick={() => setState(prev => ({ ...prev, jsonInput: JSON.stringify({ points: state.gpsPoints }, null, 2) }))}>
                      Use Current Points
                    </Button>
                  </div>
                  {state.fileError && (
                    <p className="text-error text-sm">{state.fileError}</p>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Options */}
          <Card variant="glass">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5 text-accent-primary" />
                Reconstruction Options
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-2">
                  Candidate Radius: {state.candidateRadius}m
                </label>
                <input
                  type="range"
                  min="10"
                  max="500"
                  step="10"
                  value={state.candidateRadius}
                  onChange={(e) => setState(prev => ({ ...prev, candidateRadius: parseInt(e.target.value) }))}
                  className="w-full h-2 bg-bg-tertiary rounded-lg appearance-none accent-accent-primary"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-2">
                  Max Candidates: {state.maxCandidates}
                </label>
                <input
                  type="range"
                  min="1"
                  max="50"
                  step="1"
                  value={state.maxCandidates}
                  onChange={(e) => setState(prev => ({ ...prev, maxCandidates: parseInt(e.target.value) }))}
                  className="w-full h-2 bg-bg-tertiary rounded-lg appearance-none accent-accent-primary"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-text-secondary mb-2">
                  Transition Penalty: {state.transitionPenalty.toFixed(1)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="0.1"
                  value={state.transitionPenalty}
                  onChange={(e) => setState(prev => ({ ...prev, transitionPenalty: parseFloat(e.target.value) }))}
                  className="w-full h-2 bg-bg-tertiary rounded-lg appearance-none accent-accent-primary"
                />
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="include-diagnostics"
                  checked={state.includeDiagnostics}
                  onChange={(e) => setState(prev => ({ ...prev, includeDiagnostics: e.target.checked }))}
                  className="w-4 h-4 rounded border-border-primary bg-bg-tertiary text-accent-primary focus:ring-accent-primary"
                />
                <label htmlFor="include-diagnostics" className="text-sm text-text-secondary">
                  Include Diagnostics
                </label>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="include-geometry"
                  checked={state.includeGeometry}
                  onChange={(e) => setState(prev => ({ ...prev, includeGeometry: e.target.checked }))}
                  className="w-4 h-4 rounded border-border-primary bg-bg-tertiary text-accent-primary focus:ring-accent-primary"
                />
                <label htmlFor="include-geometry" className="text-sm text-text-secondary">
                  Include Geometry
                </label>
              </div>
            </CardContent>
          </Card>

          {/* Actions */}
          <div className="flex gap-3">
            <Button
              size="lg"
              onClick={handleReconstruct}
              disabled={state.isLoading || state.gpsPoints.length < 2}
              className="flex-1"
            >
              {state.isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Reconstruct Route
                </>
              )}
            </Button>
            <Button
              size="lg"
              variant="ghost"
              onClick={handleClear}
              disabled={state.gpsPoints.length === 0 && !state.route}
            >
              <Trash2 className="w-4 h-4" />
              Clear
            </Button>
          </div>

          {/* Results Summary */}
          {state.route && (
            <Card variant="glass">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-success" />
                  Reconstruction Complete
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="p-3 bg-bg-tertiary rounded-lg">
                    <div className="text-text-muted">Segments</div>
                    <div className="font-mono font-semibold text-text-primary">{state.route.segment_count}</div>
                  </div>
                  <div className="p-3 bg-bg-tertiary rounded-lg">
                    <div className="text-text-muted">Distance</div>
                    <div className="font-mono font-semibold text-text-primary">
                      {(state.route.total_distance_m / 1000).toFixed(2)} km
                    </div>
                  </div>
                  <div className="p-3 bg-bg-tertiary rounded-lg">
                    <div className="text-text-muted">Points</div>
                    <div className="font-mono font-semibold text-text-primary">{state.route.point_count}</div>
                  </div>
                  <div className="p-3 bg-bg-tertiary rounded-lg">
                    <div className="text-text-muted">Request ID</div>
                    <div className="font-mono text-xs text-text-secondary truncate">{state.requestId}</div>
                  </div>
                </div>

                {state.diagnostics && (
                  <div className="pt-3 border-t border-border-primary space-y-2">
                    <div className="text-xs font-medium text-text-muted uppercase tracking-wider">Timing</div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="flex justify-between">
                        <span className="text-text-secondary">Preprocessing</span>
                        <span className="font-mono text-text-primary">{formatDuration(state.diagnostics.preprocessing_time_ms)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-secondary">Candidates</span>
                        <span className="font-mono text-text-primary">{formatDuration(state.diagnostics.candidate_generation_time_ms)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-secondary">ML Model</span>
                        <span className="font-mono text-text-primary">{formatDuration(state.diagnostics.model_prediction_time_ms)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-secondary">Viterbi</span>
                        <span className="font-mono text-text-primary">{formatDuration(state.diagnostics.decoding_time_ms)}</span>
                      </div>
                      <div className="flex justify-between font-medium col-span-2">
                        <span className="text-text-primary">Total</span>
                        <span className="font-mono text-accent-primary">{formatDuration(state.diagnostics.total_time_ms)}</span>
                      </div>
                    </div>
                  </div>
                )}

                <div className="flex gap-2 pt-2">
                  <Button variant="secondary" size="sm" className="flex-1">
                    <Download className="w-4 h-4" />
                    Export GeoJSON
                  </Button>
                  <Button variant="ghost" size="sm">
                    <Eye className="w-4 h-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {state.error && (
            <div className="p-4 bg-error/10 border border-error/30 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-error flex-shrink-0 mt-0.5" />
              <div className="text-error text-sm">
                <p className="font-medium">Error</p>
                <p>{state.error}</p>
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* Map */}
      <main className="flex-1 relative min-h-[calc(100vh-4rem)]">
        <div ref={mapContainerRef} className="w-full h-full map-container" />

        {/* Map Controls */}
        <div className="absolute bottom-4 right-4 flex flex-col gap-2">
          <div className="flex items-center gap-2 bg-bg-card/90 backdrop-blur rounded-lg border border-border-primary p-2">
            <button
              onClick={() => setState(prev => ({ ...prev, showGPSPoints: !prev.showGPSPoints }))}
              className={cn(
                'p-2 rounded transition-colors',
                state.showGPSPoints ? 'bg-bg-tertiary text-text-primary' : 'text-text-muted hover:text-text-primary'
              )}
              aria-pressed={state.showGPSPoints}
              aria-label="Toggle GPS points"
            >
              <MapPin className="w-5 h-5" />
            </button>
            <button
              onClick={() => setState(prev => ({ ...prev, showCandidates: !prev.showCandidates }))}
              className={cn(
                'p-2 rounded transition-colors',
                state.showCandidates ? 'bg-bg-tertiary text-text-primary' : 'text-text-muted hover:text-text-primary'
              )}
              aria-pressed={state.showCandidates}
              aria-label="Toggle candidates"
            >
              <Layers className="w-5 h-5" />
            </button>
          </div>

          {state.isLoading && (
            <div className="bg-bg-card/90 backdrop-blur rounded-lg border border-border-primary p-4 flex items-center gap-3 animate-fade-in">
              <Loader2 className="w-6 h-6 text-accent-primary animate-spin" />
              <div>
                <p className="font-medium text-text-primary">Reconstructing route...</p>
                <p className="text-xs text-text-secondary">Running HMM/Viterbi pipeline</p>
              </div>
            </div>
          )}
        </div>

        {/* Empty state */}
        {!state.route && !state.isLoading && state.gpsPoints.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-center p-8 glass rounded-xl max-w-md mx-4 pointer-events-auto">
              <MapPin className="w-16 h-16 mx-auto text-text-muted/50 mb-4" />
              <h3 className="text-lg font-semibold text-text-primary mb-2">No GPS Points</h3>
              <p className="text-text-secondary text-sm mb-6">
                Add GPS points manually, upload a file, or paste JSON to reconstruct a route.
              </p>
              <Button variant="secondary" asChild>
                <span>Upload a file instead</span>
              </Button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}