import { useState, useEffect, useRef, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Button } from '../components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Skeleton } from '../components/ui/Skeleton';
import { cn } from '../lib/utils';
import { api, type EvaluationTripSummary, type EvaluationTripDetail, type EvaluationResponse } from '../lib/api';
import {
  MapPin,
  Play,
  Loader2,
  CheckCircle,
  AlertCircle,
  XCircle,
  Target,
  BarChart3,
  RefreshCw,
} from 'lucide-react';

// Fix Leaflet marker icon issue
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

interface EvaluationPageState {
  trips: EvaluationTripSummary[];
  selectedTrip: EvaluationTripDetail | null;
  isLoadingTrips: boolean;
  isLoadingTrip: boolean;
  isReconstructing: boolean;
  error: string | null;
  result: EvaluationResponse | null;
  showReference: boolean;
  showPredicted: boolean;
  showGPSPoints: boolean;
}

const initialState: EvaluationPageState = {
  trips: [],
  selectedTrip: null,
  isLoadingTrips: true,
  isLoadingTrip: false,
  isReconstructing: false,
  error: null,
  result: null,
  showReference: true,
  showPredicted: true,
  showGPSPoints: true,
};

export function Evaluation() {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<EvaluationPageState>(initialState);
  const mapRef = useRef<L.Map | null>(null);
  const referenceLayerRef = useRef<L.Polyline | null>(null);
  const predictedLayerRef = useRef<L.Polyline | null>(null);
  const gpsMarkersRef = useRef<L.CircleMarker[]>([]);

  // Load evaluation trips on mount
  useEffect(() => {
    loadTrips();
  }, []);

  const loadTrips = async () => {
    setState(prev => ({ ...prev, isLoadingTrips: true, error: null }));
    try {
      const response = await api.listEvaluationTrips();
      setState(prev => ({
        ...prev,
        trips: response.trips,
        isLoadingTrips: false
      }));
    } catch (err) {
      setState(prev => ({
        ...prev,
        isLoadingTrips: false,
        error: err instanceof Error ? err.message : 'Failed to load evaluation trips'
      }));
    }
  };

  const loadTripDetail = async (tripId: string) => {
    setState(prev => ({ ...prev, isLoadingTrip: true, error: null, result: null }));
    try {
      const trip = await api.getEvaluationTrip(tripId);
      setState(prev => ({
        ...prev,
        selectedTrip: trip,
        isLoadingTrip: false
      }));
      // Initialize map with trip bounds
      setTimeout(() => fitMapToTrip(trip), 100);
    } catch (err) {
      setState(prev => ({
        ...prev,
        isLoadingTrip: false,
        error: err instanceof Error ? err.message : 'Failed to load trip details'
      }));
    }
  };

  const handleSelectTrip = (tripId: string) => {
    loadTripDetail(tripId);
  };

  const handleReconstruct = useCallback(async () => {
    if (!state.selectedTrip) return;

    setState(prev => ({ ...prev, isReconstructing: true, error: null }));

    try {
      const result = await api.reconstructEvaluationTrip(state.selectedTrip.trip_id);
      setState(prev => ({
        ...prev,
        isReconstructing: false,
        result
      }));
      // Update map with predicted route
      if (result.predicted_route) {
        updateMapWithResult(result);
      }
    } catch (err) {
      setState(prev => ({
        ...prev,
        isReconstructing: false,
        error: err instanceof Error ? err.message : 'Reconstruction failed'
      }));
    }
  }, [state.selectedTrip]);

  const handleReset = () => {
    setState(prev => ({
      ...prev,
      selectedTrip: null,
      result: null,
      error: null
    }));
    clearMapLayers();
  };

  // Initialize map
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

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

    L.control.zoom({ position: 'topright' }).addTo(map);

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  const fitMapToTrip = (trip: EvaluationTripDetail) => {
    const map = mapRef.current;
    if (!map) return;

    clearMapLayers();

    const coords = trip.reference_geometry;
    if (coords.length >= 2) {
      const latLngs = coords.map(([lon, lat]) => [lat, lon] as L.LatLngExpression);
      const bounds = L.latLngBounds(latLngs);
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 16 });
    }
  };

  const clearMapLayers = () => {
    const map = mapRef.current;
    if (!map) return;

    referenceLayerRef.current?.remove();
    predictedLayerRef.current?.remove();
    gpsMarkersRef.current.forEach(m => map.removeLayer(m));

    referenceLayerRef.current = null;
    predictedLayerRef.current = null;
    gpsMarkersRef.current = [];
  };

  const updateMapWithResult = (result: EvaluationResponse) => {
    const map = mapRef.current;
    if (!map || !state.selectedTrip) return;

    clearMapLayers();

    const trip = state.selectedTrip;

    // Add reference route
    if (state.showReference && result.reference_route) {
      const coords = result.reference_route.coordinates;
      if (coords.length >= 2) {
        const latLngs = coords.map(([lon, lat]) => [lat, lon] as L.LatLngExpression);
        referenceLayerRef.current = L.polyline(latLngs, {
          color: '#00d4aa',
          weight: 5,
          opacity: 0.8,
          lineCap: 'round',
          lineJoin: 'round',
          dashArray: '10, 5',
        }).bindPopup('<strong>Reference Route</strong>').addTo(map);
      }
    }

    // Add predicted route
    if (state.showPredicted && result.predicted_route) {
      const coords = result.predicted_route.coordinates;
      if (coords.length >= 2) {
        const latLngs = coords.map(([lon, lat]) => [lat, lon] as L.LatLngExpression);
        predictedLayerRef.current = L.polyline(latLngs, {
          color: '#7c5cff',
          weight: 4,
          opacity: 0.9,
          lineCap: 'round',
          lineJoin: 'round',
        }).bindPopup('<strong>Predicted Route</strong>').addTo(map);
      }
    }

    // Add GPS points
    if (state.showGPSPoints && trip.gps_points.length > 0) {
      trip.gps_points.forEach((point, index) => {
        const marker = L.circleMarker([point.latitude, point.longitude], {
          radius: 5,
          fillColor: '#f5a623',
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
        marker.addTo(map);
        gpsMarkersRef.current.push(marker);
      });
    }

    // Fit bounds to show everything
    const allLayers: L.Layer[] = [];
    if (referenceLayerRef.current) allLayers.push(referenceLayerRef.current);
    if (predictedLayerRef.current) allLayers.push(predictedLayerRef.current);
    allLayers.push(...gpsMarkersRef.current);

    if (allLayers.length > 0) {
      const group = L.featureGroup(allLayers);
      map.fitBounds(group.getBounds(), { padding: [50, 50], maxZoom: 16 });
    }
  };

  // Update map when visibility toggles change
  useEffect(() => {
    if (state.result) {
      updateMapWithResult(state.result);
    }
  }, [state.showReference, state.showPredicted, state.showGPSPoints, state.result]);

  const formatDistance = (meters: number) => {
    if (meters < 1000) return `${meters.toFixed(0)} m`;
    return `${(meters / 1000).toFixed(2)} km`;
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(1)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const getAccuracyColor = (score: number) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] flex">
      {/* Sidebar */}
      <aside className="w-full lg:w-96 flex-shrink-0 border-r border-border-primary bg-bg-secondary/50 p-4 lg:p-6 overflow-y-auto">
        <div className="space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-text-primary">Evaluation Trips</h2>
              <p className="text-sm text-text-secondary">Select a trip to evaluate</p>
            </div>
            <Button variant="ghost" size="sm" onClick={loadTrips} disabled={state.isLoadingTrips}>
              <RefreshCw className={cn('w-4 h-4', state.isLoadingTrips && 'animate-spin')} />
            </Button>
          </div>

          {/* Trips List */}
          {state.isLoadingTrips ? (
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-20 w-full rounded-lg" variant="rectangular" />
              ))}
            </div>
          ) : state.error && !state.selectedTrip ? (
            <div className="p-4 bg-error/10 border border-error/30 rounded-lg text-center">
              <AlertCircle className="w-8 h-8 text-error mx-auto mb-2" />
              <p className="text-error text-sm">{state.error}</p>
              <Button variant="secondary" size="sm" className="mt-2" onClick={loadTrips}>
                Retry
              </Button>
            </div>
          ) : (
            <div className="space-y-2 max-h-[calc(100vh-300px)] overflow-y-auto">
              {state.trips.map((trip) => {
                const isSelected = state.selectedTrip?.trip_id === trip.trip_id;
                return (
                  <button
                    key={trip.trip_id}
                    onClick={() => handleSelectTrip(trip.trip_id)}
                    className={cn(
                      'w-full p-4 rounded-xl text-left transition-all duration-200',
                      'border-2',
                      isSelected
                        ? 'border-accent-primary bg-accent-primary/5'
                        : 'border-border-primary hover:border-border-secondary hover:bg-bg-tertiary'
                    )}
                    aria-pressed={isSelected}
                  >
                    <div className="flex items-start gap-3">
                      <div className={cn(
                        'w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0',
                        isSelected ? 'bg-accent-primary/20 text-accent-primary' : 'bg-bg-tertiary text-text-secondary'
                      )}>
                        <MapPin className="w-5 h-5" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className={cn('font-medium truncate', isSelected ? 'text-text-primary' : 'text-text-secondary')}>
                          {trip.name}
                        </h3>
                        <p className="text-xs text-text-muted truncate mt-0.5">{trip.description}</p>
                        <div className="flex flex-wrap gap-1.5 mt-2">
                          <Badge variant="info" size="sm">{trip.point_count} points</Badge>
                          <Badge variant="default" size="sm">{trip.reference_segment_count} segments</Badge>
                          <Badge variant="default" size="sm">{formatDistance(trip.reference_distance_m)}</Badge>
                        </div>
                      </div>
                      {isSelected && (
                        <CheckCircle className="w-5 h-5 text-accent-primary flex-shrink-0" />
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          )}

          {/* Selected Trip Details */}
          {state.selectedTrip && !state.isLoadingTrip && (
            <Card variant="glass">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5 text-accent-primary" />
                  {state.selectedTrip.name}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-text-secondary text-sm">{state.selectedTrip.description}</p>

                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-bg-tertiary rounded-lg">
                    <div className="text-text-muted text-xs">GPS Points</div>
                    <div className="font-mono font-semibold text-text-primary">{state.selectedTrip.gps_points.length}</div>
                  </div>
                  <div className="p-3 bg-bg-tertiary rounded-lg">
                    <div className="text-text-muted text-xs">Ref. Segments</div>
                    <div className="font-mono font-semibold text-text-primary">{state.selectedTrip.reference_route.segment_count}</div>
                  </div>
                  <div className="p-3 bg-bg-tertiary rounded-lg">
                    <div className="text-text-muted text-xs">Ref. Distance</div>
                    <div className="font-mono font-semibold text-text-primary">{formatDistance(state.selectedTrip.reference_route.total_distance_m)}</div>
                  </div>
                  <div className="p-3 bg-bg-tertiary rounded-lg">
                    <div className="text-text-muted text-xs">Trip ID</div>
                    <div className="font-mono text-xs text-text-secondary truncate">{state.selectedTrip.trip_id}</div>
                  </div>
                </div>

                <div className="flex gap-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    className="flex-1"
                    onClick={handleReconstruct}
                    disabled={state.isReconstructing || !!state.result}
                  >
                    {state.isReconstructing ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Evaluating...
                      </>
                    ) : state.result ? (
                      <>
                        <CheckCircle className="w-4 h-4" />
                        Evaluated
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        Run Evaluation
                      </>
                    )}
                  </Button>
                  <Button variant="ghost" size="sm" onClick={handleReset}>
                    <XCircle className="w-4 h-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {state.isLoadingTrip && (
            <div className="space-y-3">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-20 w-full rounded-lg" variant="rectangular" />
              ))}
            </div>
          )}

          {/* Results */}
          {state.result && (
            <Card variant="glass">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-accent-primary" />
                  Evaluation Results
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Overall Score */}
                {state.result.accuracy_metrics && (
                  <div className="p-4 bg-bg-tertiary rounded-xl">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-text-secondary">Overall Accuracy</span>
                      <Badge variant={getAccuracyColor(state.result.accuracy_metrics.overall_score)}>
                        {(state.result.accuracy_metrics.overall_score * 100).toFixed(1)}%
                      </Badge>
                    </div>
                    <div className="h-2 bg-bg-primary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-accent-primary to-accent-tertiary rounded-full transition-all duration-500"
                        style={{ width: `${state.result.accuracy_metrics.overall_score * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Detailed Metrics */}
                {state.result.accuracy_metrics && (
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 bg-bg-tertiary rounded-lg">
                      <div className="text-text-muted text-xs">Segment Accuracy</div>
                      <div className="font-mono font-semibold text-text-primary">
                        {(state.result.accuracy_metrics!.segment_accuracy * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-3 bg-bg-tertiary rounded-lg">
                      <div className="text-text-muted text-xs">Distance Error</div>
                      <div className="font-mono font-semibold text-text-primary">
                        {state.result.accuracy_metrics!.distance_error_pct.toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-3 bg-bg-tertiary rounded-lg">
                      <div className="text-text-muted text-xs">Point Accuracy</div>
                      <div className="font-mono font-semibold text-text-primary">
                        {(state.result.accuracy_metrics!.point_accuracy * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-3 bg-bg-tertiary rounded-lg">
                      <div className="text-text-muted text-xs">Request ID</div>
                      <div className="font-mono text-xs text-text-secondary truncate">{state.result.request_id}</div>
                    </div>
                  </div>
                )}

                {/* Route Comparison */}
                {state.result.predicted_route && state.result.reference_route && (
                  <div className="pt-3 border-t border-border-primary space-y-3">
                    <div className="text-xs font-medium text-text-muted uppercase tracking-wider">Route Comparison</div>
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div className="p-2 bg-bg-tertiary rounded-lg">
                        <div className="flex items-center gap-1 text-text-muted">
                          <div className="w-3 h-3 rounded border border-dashed border-accent-primary/50" />
                          Reference
                        </div>
                        <div className="font-mono text-text-primary mt-1">
                          {state.result.reference_route.segment_count} segments · {formatDistance(state.result.reference_route.total_distance_m)}
                        </div>
                      </div>
                      <div className="p-2 bg-bg-tertiary rounded-lg">
                        <div className="flex items-center gap-1 text-text-muted">
                          <div className="w-3 h-3 rounded bg-accent-tertiary" />
                          Predicted
                        </div>
                        <div className="font-mono text-text-primary mt-1">
                          {state.result.predicted_route.segment_count} segments · {formatDistance(state.result.predicted_route.total_distance_m)}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Diagnostics */}
                {state.result.diagnostics && (
                  <div className="pt-3 border-t border-border-primary space-y-2">
                    <div className="text-xs font-medium text-text-muted uppercase tracking-wider">Timing</div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="flex justify-between">
                        <span className="text-text-secondary">Preprocessing</span>
                        <span className="font-mono text-text-primary">{formatDuration(state.result.diagnostics!.preprocessing_time_ms)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-secondary">Candidates</span>
                        <span className="font-mono text-text-primary">{formatDuration(state.result.diagnostics!.candidate_generation_time_ms)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-secondary">ML Model</span>
                        <span className="font-mono text-text-primary">{formatDuration(state.result.diagnostics!.model_prediction_time_ms)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-secondary">Viterbi</span>
                        <span className="font-mono text-text-primary">{formatDuration(state.result.diagnostics!.decoding_time_ms)}</span>
                      </div>
                      <div className="flex justify-between font-medium col-span-2">
                        <span className="text-text-primary">Total</span>
                        <span className="font-mono text-accent-primary">{formatDuration(state.result.diagnostics!.total_time_ms)}</span>
                      </div>
                    </div>
                  </div>
                )}
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

        {/* Map Legend */}
        <div className="absolute top-4 left-4 glass-strong p-3 rounded-lg border border-border-primary">
          <div className="text-xs font-medium text-text-primary mb-2">Map Layers</div>
          <div className="space-y-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={state.showReference}
                onChange={(e) => setState(prev => ({ ...prev, showReference: e.target.checked }))}
                className="w-4 h-4 rounded border-border-primary bg-bg-tertiary text-accent-primary focus:ring-accent-primary"
              />
              <span className="flex items-center gap-1 text-sm text-text-secondary">
                <span className="w-6 h-1 rounded bg-gradient-to-r from-accent-primary to-accent-tertiary" style={{ borderRadius: '9999px' }} />
                Reference Route
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={state.showPredicted}
                onChange={(e) => setState(prev => ({ ...prev, showPredicted: e.target.checked }))}
                className="w-4 h-4 rounded border-border-primary bg-bg-tertiary text-accent-primary focus:ring-accent-primary"
              />
              <span className="flex items-center gap-1 text-sm text-text-secondary">
                <span className="w-6 h-1 rounded bg-accent-tertiary" />
                Predicted Route
              </span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={state.showGPSPoints}
                onChange={(e) => setState(prev => ({ ...prev, showGPSPoints: e.target.checked }))}
                className="w-4 h-4 rounded border-border-primary bg-bg-tertiary text-accent-primary focus:ring-accent-primary"
              />
              <span className="flex items-center gap-1 text-sm text-text-secondary">
                <span className="w-3 h-3 rounded-full bg-warning border-2 border-bg-primary" />
                GPS Points
              </span>
            </label>
          </div>
        </div>

        {/* Loading Overlay */}
        {state.isReconstructing && (
          <div className="absolute inset-0 flex items-center justify-center bg-bg-primary/80 backdrop-blur-sm z-10">
            <div className="glass p-8 rounded-xl text-center max-w-md mx-4 animate-scale-in">
              <Loader2 className="w-12 h-12 text-accent-primary animate-spin mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-text-primary mb-2">Running Evaluation</h3>
              <p className="text-text-secondary">Processing trip through PATHSNAP pipeline...</p>
              <div className="mt-4 flex justify-center gap-4 text-xs text-text-muted">
                <span>Preprocessing</span>
                <span>Candidates</span>
                <span>ML Scoring</span>
                <span>Viterbi</span>
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!state.selectedTrip && !state.isLoadingTrips && !state.isLoadingTrip && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-center p-8 glass rounded-xl max-w-md mx-4 pointer-events-auto">
              <Target className="w-16 h-16 mx-auto text-text-muted/50 mb-4" />
              <h3 className="text-lg font-semibold text-text-primary mb-2">Select an Evaluation Trip</h3>
              <p className="text-text-secondary text-sm mb-6">
                Choose one of the 10 reserved evaluation trips from the sidebar to begin.
              </p>
              <div className="text-xs text-text-muted space-y-1">
                <p>1. SF Downtown Loop</p>
                <p>2. Golden Gate Park Crossing</p>
                <p>3. Mission District Streets</p>
                <p>4. Embarcadero Waterfront</p>
                <p>5. Lombard Street Zigzag</p>
                <p>6. Twin Peaks Summit</p>
                <p>7. Presidio to Marina</p>
                <p>8. SoMa Arterial</p>
                <p>9. Castro Neighborhood</p>
                <p>10. Ocean Beach Coastal</p>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}