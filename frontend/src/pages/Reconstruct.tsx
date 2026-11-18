import { useState, useRef, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { Textarea } from '../components/ui/Input';
import { Badge } from '../components/ui/Badge';
import { cn } from '../lib/utils';
import { api, type GPSPoint, type ReconstructionRequest, type ReconstructedRoute } from '../lib/api';
import {
  Upload,
  Download,
  Trash2,
  Play,
  Loader2,
  AlertCircle,
  CheckCircle,
  Sparkles,
} from 'lucide-react';

// Fix Leaflet marker icons
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const sampleTrips: { name: string; desc: string; points: GPSPoint[] }[] = [
  {
    name: 'SF Downtown Loop',
    desc: '15 GPS observations in SF Financial District',
    points: Array.from({ length: 15 }, (_, i) => ({
      latitude: 37.7946,
      longitude: -122.3999 + i * 0.001,
      timestamp: new Date(Date.now() + i * 10000).toISOString(),
      speed: 8.5,
      heading: 90.0,
      accuracy: 5.0,
    })),
  },
  {
    name: 'Golden Gate Park Crossing',
    desc: '20 GPS observations along park corridor',
    points: Array.from({ length: 20 }, (_, i) => ({
      latitude: 37.7694,
      longitude: -122.4862 + i * 0.001,
      timestamp: new Date(Date.now() + i * 12000).toISOString(),
      speed: 11.2,
      heading: 90.0,
      accuracy: 4.0,
    })),
  },
  {
    name: 'Mission District Transit',
    desc: '18 GPS points along urban avenues',
    points: Array.from({ length: 18 }, (_, i) => ({
      latitude: 37.7599 + i * 0.0005,
      longitude: -122.4148 + i * 0.0008,
      timestamp: new Date(Date.now() + i * 8000).toISOString(),
      speed: 6.8,
      heading: 35.0,
      accuracy: 6.0,
    })),
  },
];

interface ReconstructState {
  inputMethod: 'sample' | 'manual' | 'file' | 'json';
  gpsPoints: GPSPoint[];
  jsonInput: string;
  fileError: string | null;

  candidateRadius: number;
  maxCandidates: number;
  transitionPenalty: number;
  includeDiagnostics: boolean;

  isLoading: boolean;
  error: string | null;
  route: ReconstructedRoute | null;
  diagnostics: any | null;
  requestId: string | null;

  showGPSPoints: boolean;
  showCandidates: boolean;
}

export function Reconstruct() {
  const location = useLocation();
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const markersRef = useRef<L.CircleMarker[]>([]);
  const polylineRef = useRef<L.Polyline | null>(null);

  const [state, setState] = useState<ReconstructState>({
    inputMethod: 'sample',
    gpsPoints: sampleTrips[0].points,
    jsonInput: '',
    fileError: null,
    candidateRadius: 100,
    maxCandidates: 10,
    transitionPenalty: 1.0,
    includeDiagnostics: true,
    isLoading: false,
    error: null,
    route: null,
    diagnostics: null,
    requestId: null,
    showGPSPoints: true,
    showCandidates: false,
  });

  // Check if points passed from navigation (e.g. Upload page)
  useEffect(() => {
    const passedPoints = (location.state as any)?.gpsPoints;
    if (passedPoints && Array.isArray(passedPoints) && passedPoints.length >= 2) {
      setState(prev => ({
        ...prev,
        gpsPoints: passedPoints,
        inputMethod: 'file',
        route: null,
        error: null,
      }));
    }
  }, [location.state]);

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [37.7749, -122.4194],
      zoom: 13,
      zoomControl: false,
      attributionControl: false,
    });

    const streets = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
      {
        attribution: '&copy; Esri &mdash; Street Map',
        maxZoom: 19,
      }
    );

    const osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 19,
    });

    const dark = L.tileLayer(
      'https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      {
        attribution: '&copy; Esri &mdash; Dark Canvas',
        maxZoom: 16,
      }
    );

    const satellite = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      {
        attribution: '&copy; Esri &mdash; Satellite',
        maxZoom: 19,
      }
    );

    // Add Street Map by default so roads, labels and buildings are completely visible
    streets.addTo(map);

    L.control.layers(
      {
        'Street Map': streets,
        'OpenStreetMap': osm,
        'Dark Canvas': dark,
        'Satellite Imagery': satellite,
      },
      undefined,
      { position: 'topright' }
    ).addTo(map);

    L.control.zoom({ position: 'topright' }).addTo(map);
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Update map markers when points change or route is reconstructed
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // Clear previous markers
    markersRef.current.forEach(m => m.remove());
    markersRef.current = [];

    // Clear polyline if no route
    if (!state.route && polylineRef.current) {
      polylineRef.current.remove();
      polylineRef.current = null;
    }

    if (state.gpsPoints.length > 0 && state.showGPSPoints) {
      const latLngs: L.LatLngExpression[] = [];

      state.gpsPoints.forEach((p, idx) => {
        const marker = L.circleMarker([p.latitude, p.longitude], {
          radius: 6,
          fillColor: '#00d4aa',
          color: '#090a10',
          weight: 2,
          fillOpacity: 0.95,
        }).bindPopup(`
          <div style="font-family: inherit;">
            <strong style="color: #00d4aa;">Observation #${idx + 1}</strong><br/>
            Lat: ${p.latitude.toFixed(6)}<br/>
            Lon: ${p.longitude.toFixed(6)}<br/>
            Time: ${new Date(p.timestamp).toLocaleTimeString()}
            ${p.speed ? `<br/>Speed: ${p.speed.toFixed(1)} m/s` : ''}
          </div>
        `);
        marker.addTo(map);
        markersRef.current.push(marker);
        latLngs.push([p.latitude, p.longitude]);
      });

      // Fit map bounds if no route drawn yet
      if (!state.route && latLngs.length > 0) {
        map.fitBounds(L.latLngBounds(latLngs), { padding: [50, 50], maxZoom: 16 });
      }
    }
  }, [state.gpsPoints, state.showGPSPoints, state.route]);

  // Update route polyline when route reconstruction succeeds
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !state.route) return;

    if (polylineRef.current) {
      polylineRef.current.remove();
    }

    const coords = state.route.coordinates;
    if (coords && coords.length >= 2) {
      const latLngs = coords.map(([lon, lat]: [number, number]) => [lat, lon] as L.LatLngExpression);

      const polyline = L.polyline(latLngs, {
        color: '#8b5cf6',
        weight: 5,
        opacity: 0.95,
        lineCap: 'round',
        lineJoin: 'round',
      }).addTo(map);

      polyline.bindPopup(`
        <div style="font-family: inherit;">
          <strong style="color: #8b5cf6;">Reconstructed Route</strong><br/>
          Arcs: ${state.route.arc_ids.length}<br/>
          Distance: ${(state.route.total_distance_m / 1000).toFixed(2)} km<br/>
          Matched Points: ${state.route.point_count}
        </div>
      `);

      polylineRef.current = polyline;
      map.fitBounds(L.latLngBounds(latLngs), { padding: [60, 60], maxZoom: 16 });
    }
  }, [state.route]);

  // Load a sample preset
  const handleSelectSample = (trip: typeof sampleTrips[0]) => {
    setState(prev => ({
      ...prev,
      inputMethod: 'sample',
      gpsPoints: trip.points,
      route: null,
      diagnostics: null,
      error: null,
    }));
  };

  // Run Route Reconstruction
  const handleReconstruct = async () => {
    if (state.gpsPoints.length < 2) {
      setState(prev => ({ ...prev, error: 'At least 2 GPS observations are required for map matching' }));
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
          include_geometry: true,
        },
      };

      const response = await api.reconstructRoute(request);

      setState(prev => ({
        ...prev,
        isLoading: false,
        route: response.route || null,
        diagnostics: response.diagnostics || null,
        requestId: response.request_id,
        error: response.success ? null : (response.error || 'Reconstruction failed'),
      }));
    } catch (err) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: err instanceof Error ? err.message : 'Reconstruction failed',
      }));
    }
  };

  // Export GeoJSON
  const handleExportGeoJSON = () => {
    if (!state.route) return;
    const geojson = {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          properties: {
            arc_ids: state.route.arc_ids,
            total_distance_m: state.route.total_distance_m,
            point_count: state.route.point_count,
            request_id: state.requestId,
          },
          geometry: {
            type: 'LineString',
            coordinates: state.route.coordinates,
          },
        },
      ],
    };

    const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/geo+json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pathsnap_route_${state.requestId || 'export'}.geojson`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleClear = () => {
    setState(prev => ({
      ...prev,
      gpsPoints: [],
      route: null,
      diagnostics: null,
      error: null,
      requestId: null,
      jsonInput: '',
    }));
    if (polylineRef.current) {
      polylineRef.current.remove();
      polylineRef.current = null;
    }
    markersRef.current.forEach(m => m.remove());
    markersRef.current = [];
  };

  return (
    <div className="w-full flex-1 flex flex-col lg:flex-row min-h-[calc(100vh-4rem)]">
      {/* Sidebar Controls */}
      <aside className="w-full lg:w-[420px] flex-shrink-0 bg-bg-secondary/70 border-r border-border-primary/80 p-5 lg:p-6 overflow-y-auto max-h-none lg:max-h-[calc(100vh-4rem)]">
        <div className="space-y-6">
          <div>
            <h1 className="text-2xl font-bold text-text-primary tracking-tight">Route Reconstruction</h1>
            <p className="text-xs text-text-secondary mt-1">
              Test candidate generation, Random Forest scoring, and HMM Viterbi decoding.
            </p>
          </div>

          {/* Preset Sample Trips Selector */}
          <div className="space-y-2">
            <span className="text-xs font-semibold text-text-muted uppercase tracking-wider block">
              Quick Test Presets
            </span>
            <div className="grid grid-cols-1 gap-2">
              {sampleTrips.map((trip) => {
                const isSelected =
                  state.inputMethod === 'sample' && state.gpsPoints.length === trip.points.length;
                return (
                  <button
                    key={trip.name}
                    onClick={() => handleSelectSample(trip)}
                    className={cn(
                      'text-left p-3 rounded-lg border transition-all flex items-center justify-between',
                      isSelected
                        ? 'bg-accent-primary/10 border-accent-primary text-text-primary'
                        : 'bg-bg-tertiary/60 border-border-primary/70 text-text-secondary hover:text-text-primary hover:border-border-secondary'
                    )}
                  >
                    <div>
                      <div className="font-semibold text-sm">{trip.name}</div>
                      <div className="text-xs text-text-muted">{trip.desc}</div>
                    </div>
                    <Badge variant={isSelected ? 'success' : 'default'} className="text-xs">
                      {trip.points.length} pts
                    </Badge>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Input Method Selector Tabs */}
          <div className="flex bg-bg-tertiary/80 rounded-lg p-1 border border-border-primary/50 text-xs font-medium">
            <button
              onClick={() => setState(prev => ({ ...prev, inputMethod: 'sample' }))}
              className={cn(
                'flex-1 py-1.5 rounded-md transition-all text-center',
                state.inputMethod === 'sample' ? 'bg-bg-card text-accent-primary font-bold shadow-sm' : 'text-text-secondary'
              )}
            >
              Presets
            </button>
            <button
              onClick={() => setState(prev => ({ ...prev, inputMethod: 'manual' }))}
              className={cn(
                'flex-1 py-1.5 rounded-md transition-all text-center',
                state.inputMethod === 'manual' ? 'bg-bg-card text-accent-primary font-bold shadow-sm' : 'text-text-secondary'
              )}
            >
              Manual ({state.gpsPoints.length})
            </button>
            <button
              onClick={() => setState(prev => ({ ...prev, inputMethod: 'file' }))}
              className={cn(
                'flex-1 py-1.5 rounded-md transition-all text-center',
                state.inputMethod === 'file' ? 'bg-bg-card text-accent-primary font-bold shadow-sm' : 'text-text-secondary'
              )}
            >
              File
            </button>
            <button
              onClick={() => setState(prev => ({ ...prev, inputMethod: 'json' }))}
              className={cn(
                'flex-1 py-1.5 rounded-md transition-all text-center',
                state.inputMethod === 'json' ? 'bg-bg-card text-accent-primary font-bold shadow-sm' : 'text-text-secondary'
              )}
            >
              JSON
            </button>
          </div>

          {/* File Upload Tab */}
          {state.inputMethod === 'file' && (
            <div className="p-4 bg-bg-tertiary/50 border border-dashed border-border-primary rounded-xl text-center">
              <Upload className="w-8 h-8 text-accent-primary mx-auto mb-2 opacity-80" />
              <p className="text-xs text-text-primary font-medium">Upload GPS File</p>
              <p className="text-xs text-text-muted mb-3">Supports CSV, GPX, or JSON track files</p>
              <label className="btn btn-secondary text-xs cursor-pointer py-1.5 px-3">
                Select File
                <input
                  type="file"
                  accept=".csv,.gpx,.json"
                  className="hidden"
                  onChange={async (e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      try {
                        const pts = await api.parseGPSFile(file);
                        setState(prev => ({ ...prev, gpsPoints: pts, error: null, route: null }));
                      } catch (err: any) {
                        setState(prev => ({ ...prev, error: err.message }));
                      }
                    }
                  }}
                />
              </label>
              {state.gpsPoints.length > 0 && (
                <div className="mt-3 text-xs text-accent-primary font-mono">
                  {state.gpsPoints.length} points loaded from file
                </div>
              )}
            </div>
          )}

          {/* JSON Tab */}
          {state.inputMethod === 'json' && (
            <div className="space-y-2">
              <Textarea
                rows={6}
                value={state.jsonInput}
                onChange={(e) => setState(prev => ({ ...prev, jsonInput: e.target.value }))}
                placeholder='[{"latitude": 37.7946, "longitude": -122.3999, "timestamp": "2024-01-15T10:00:00Z"}, ...]'
                className="font-mono text-xs"
              />
              <Button
                variant="secondary"
                size="sm"
                className="w-full text-xs"
                onClick={() => {
                  try {
                    const pts = api.parseJSON(state.jsonInput);
                    setState(prev => ({ ...prev, gpsPoints: pts, error: null, route: null }));
                  } catch (err: any) {
                    setState(prev => ({ ...prev, error: err.message }));
                  }
                }}
              >
                Apply JSON Points
              </Button>
            </div>
          )}

          {/* Parameters Accordion / Sliders */}
          <Card variant="glass" className="p-4 border-border-primary/80">
            <span className="text-xs font-semibold text-text-muted uppercase tracking-wider block mb-3">
              Decoding Parameters
            </span>
            <div className="space-y-4 text-xs">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-text-secondary">Search Radius</span>
                  <span className="font-mono font-bold text-accent-primary">{state.candidateRadius} m</span>
                </div>
                <input
                  type="range"
                  min="20"
                  max="500"
                  step="10"
                  value={state.candidateRadius}
                  onChange={(e) => setState(prev => ({ ...prev, candidateRadius: parseInt(e.target.value) }))}
                  className="w-full h-1.5 bg-bg-tertiary rounded-lg appearance-none accent-accent-primary cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-text-secondary">Max Candidates / Obs</span>
                  <span className="font-mono font-bold text-accent-primary">{state.maxCandidates}</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="30"
                  step="1"
                  value={state.maxCandidates}
                  onChange={(e) => setState(prev => ({ ...prev, maxCandidates: parseInt(e.target.value) }))}
                  className="w-full h-1.5 bg-bg-tertiary rounded-lg appearance-none accent-accent-primary cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-text-secondary">Transition Penalty</span>
                  <span className="font-mono font-bold text-accent-primary">{state.transitionPenalty.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="5"
                  step="0.1"
                  value={state.transitionPenalty}
                  onChange={(e) => setState(prev => ({ ...prev, transitionPenalty: parseFloat(e.target.value) }))}
                  className="w-full h-1.5 bg-bg-tertiary rounded-lg appearance-none accent-accent-primary cursor-pointer"
                />
              </div>
            </div>
          </Card>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <Button
              size="lg"
              onClick={handleReconstruct}
              disabled={state.isLoading || state.gpsPoints.length < 2}
              className="flex-1 shadow-md shadow-accent-primary/20 text-sm font-bold"
            >
              {state.isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Running Pipeline...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Run Reconstruction
                </>
              )}
            </Button>

            <Button
              size="lg"
              variant="ghost"
              onClick={handleClear}
              disabled={state.gpsPoints.length === 0 && !state.route}
              className="text-text-muted hover:text-error"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>

          {/* Error Notice */}
          {state.error && (
            <div className="p-3 bg-error/15 border border-error/30 rounded-xl text-error text-xs flex items-start gap-2">
              <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
              <span>{state.error}</span>
            </div>
          )}

          {/* Telemetry & Results Card */}
          {state.route && (
            <Card variant="glass" className="p-4 border-accent-primary/40 bg-accent-primary/5">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2 text-success font-semibold text-sm">
                  <CheckCircle className="w-4 h-4" />
                  <span>Matched Successfully</span>
                </div>
                <Badge variant="success" className="text-xs font-mono">
                  {state.requestId}
                </Badge>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                <div className="p-2.5 bg-bg-tertiary/80 rounded-lg">
                  <span className="text-text-muted block text-[10px] uppercase">Total Distance</span>
                  <span className="font-mono text-base font-bold text-text-primary">
                    {(state.route.total_distance_m / 1000).toFixed(2)} km
                  </span>
                </div>

                <div className="p-2.5 bg-bg-tertiary/80 rounded-lg">
                  <span className="text-text-muted block text-[10px] uppercase">Matched Arcs</span>
                  <span className="font-mono text-base font-bold text-text-primary">
                    {state.route.arc_ids.length}
                  </span>
                </div>

                <div className="p-2.5 bg-bg-tertiary/80 rounded-lg">
                  <span className="text-text-muted block text-[10px] uppercase">Obs Points</span>
                  <span className="font-mono text-base font-bold text-text-primary">
                    {state.route.point_count}
                  </span>
                </div>

                <div className="p-2.5 bg-bg-tertiary/80 rounded-lg">
                  <span className="text-text-muted block text-[10px] uppercase">Latency</span>
                  <span className="font-mono text-base font-bold text-accent-primary">
                    {state.diagnostics?.total_time_ms ? `${Math.round(state.diagnostics.total_time_ms)} ms` : '<300 ms'}
                  </span>
                </div>
              </div>

              {state.diagnostics && (
                <div className="pt-2 border-t border-border-primary/60 text-[11px] space-y-1 text-text-secondary">
                  <div className="flex justify-between">
                    <span>Preprocessing:</span>
                    <span className="font-mono text-text-primary">{state.diagnostics.preprocessing_time_ms.toFixed(1)} ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Candidate Search:</span>
                    <span className="font-mono text-text-primary">{state.diagnostics.candidate_generation_time_ms.toFixed(1)} ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span>ML Scoring (RF):</span>
                    <span className="font-mono text-text-primary">{state.diagnostics.model_prediction_time_ms.toFixed(1)} ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Viterbi Decoder:</span>
                    <span className="font-mono text-text-primary">{state.diagnostics.decoding_time_ms.toFixed(1)} ms</span>
                  </div>
                </div>
              )}

              <Button
                variant="secondary"
                size="sm"
                className="w-full mt-3 text-xs"
                onClick={handleExportGeoJSON}
              >
                <Download className="w-3.5 h-3.5 mr-1.5" />
                Export GeoJSON Route
              </Button>
            </Card>
          )}
        </div>
      </aside>

      {/* Main Map Viewer */}
      <main className="flex-1 relative min-h-[500px] lg:min-h-full flex flex-col">
        <div ref={mapContainerRef} className="w-full h-full flex-1" />

        {/* Floating Map Legend & Toggles */}
        <div className="absolute top-4 right-4 z-20 flex items-center gap-2 bg-bg-card/90 backdrop-blur border border-border-primary/80 rounded-xl px-3 py-2 text-xs shadow-lg">
          <label className="flex items-center gap-1.5 cursor-pointer text-text-secondary hover:text-text-primary">
            <input
              type="checkbox"
              checked={state.showGPSPoints}
              onChange={(e) => setState(prev => ({ ...prev, showGPSPoints: e.target.checked }))}
              className="accent-accent-primary w-3.5 h-3.5"
            />
            <span className="w-2.5 h-2.5 rounded-full bg-accent-primary inline-block" />
            <span>GPS Points</span>
          </label>

          {state.route && (
            <div className="flex items-center gap-1.5 ml-2 pl-2 border-l border-border-primary/80 text-text-secondary">
              <span className="w-4 h-1 bg-accent-tertiary rounded-full inline-block" />
              <span>Matched Road Route</span>
            </div>
          )}
        </div>

        {/* Initial Prompt Card if no route yet */}
        {!state.route && !state.isLoading && (
          <div className="absolute bottom-6 left-6 z-20 pointer-events-none hidden sm:block">
            <div className="glass px-4 py-3 rounded-xl border border-border-primary text-xs max-w-sm pointer-events-auto shadow-xl">
              <div className="flex items-center gap-2 text-accent-primary font-semibold mb-1">
                <Sparkles className="w-4 h-4" />
                <span>Ready to Reconstruct</span>
              </div>
              <p className="text-text-secondary leading-snug">
                Click <strong>"Run Reconstruction"</strong> on the left panel to execute candidate generation, ML probability scoring, and Viterbi sequence decoding on this trip.
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
