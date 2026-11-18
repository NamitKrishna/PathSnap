/**
 * PATHSNAP API Client & Type Definitions
 */

export interface GPSPoint {
  latitude: number;
  longitude: number;
  timestamp: string;
  speed?: number;
  heading?: number;
  accuracy?: number;
}

export interface ReconstructionOptions {
  candidate_radius_m?: number;
  max_candidates_per_point?: number;
  transition_penalty?: number;
  include_diagnostics?: boolean;
  include_geometry?: boolean;
}

export interface ReconstructionRequest {
  gps_points: GPSPoint[];
  options?: ReconstructionOptions;
}

export interface CandidateSegment {
  arc_id: string;
  probability: number;
  distance_m?: number;
  heading_diff_deg?: number;
  road_class?: string;
  geometry?: any;
}

export interface ReconstructedRoute {
  arc_ids: string[];
  coordinates: [number, number][];
  total_distance_m: number;
  point_count: number;
  segment_count: number;
}

export interface ReconstructionDiagnostics {
  preprocessing_time_ms: number;
  candidate_generation_time_ms: number;
  model_prediction_time_ms: number;
  decoding_time_ms: number;
  total_time_ms: number;
  candidate_stats?: Record<string, any>;
  hmm_stats?: Record<string, any>;
  warnings?: string[];
}

export interface ReconstructionResponse {
  success: boolean;
  status: string;
  request_id: string;
  route?: ReconstructedRoute | null;
  diagnostics?: ReconstructionDiagnostics | null;
  error?: string | null;
  error_code?: string | null;
}

export interface FileUploadResponse {
  success: boolean;
  filename: string;
  format: string;
  point_count: number;
  bounds?: {
    min_lat: number;
    max_lat: number;
    min_lon: number;
    max_lon: number;
  };
  duration_ms: number;
  message?: string;
  error?: string;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  timestamp: string;
  components: Record<string, string>;
  uptime_seconds?: number;
}

export interface EvaluationTripSummary {
  trip_id: string;
  name: string;
  description: string;
  point_count: number;
  bounds: {
    min_lat: number;
    max_lat: number;
    min_lon: number;
    max_lon: number;
  };
  reference_distance_m: number;
  reference_segment_count: number;
}

export interface EvaluationTripDetail {
  trip_id: string;
  name: string;
  description: string;
  gps_points: GPSPoint[];
  reference_route: ReconstructedRoute;
  reference_geometry: [number, number][];
}

export interface EvaluationResponse {
  success: boolean;
  trip_id: string;
  request_id: string;
  predicted_route?: ReconstructedRoute | null;
  reference_route?: ReconstructedRoute | null;
  accuracy_metrics?: Record<string, number>;
  diagnostics?: ReconstructionDiagnostics | null;
  error?: string | null;
  error_code?: string | null;
}

export interface EvaluationListResponse {
  trips: EvaluationTripSummary[];
  total: number;
}

function getBaseUrl(): string {
  if (typeof window !== 'undefined') {
    const stored = localStorage.getItem('apiBase');
    if (stored && stored.trim() !== '') {
      return stored.replace(/\/+$/, '');
    }
  }
  const envBase = (import.meta as any).env?.VITE_API_BASE;
  if (envBase) {
    return envBase.replace(/\/+$/, '');
  }
  return '/api/v1';
}

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const base = getBaseUrl();
  const path = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  const url = `${base}${path}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    let errMessage = `Request failed: ${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      if (data && (data.error || data.detail || data.message || data.user_message)) {
        errMessage = data.error || data.user_message || (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) || data.message;
      }
    } catch {
      // Non-JSON response
    }
    throw new Error(errMessage);
  }

  return response.json();
}

export const api = {
  /**
   * Check backend health
   */
  async checkHealth(): Promise<HealthResponse> {
    return request<HealthResponse>('/health');
  },

  /**
   * Upload GPS file (CSV, GPX, JSON) for validation
   */
  async uploadGPSFile(file: File): Promise<FileUploadResponse> {
    const base = getBaseUrl();
    const url = `${base}/upload`;
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errMessage = `Upload failed: ${response.statusText}`;
      try {
        const data = await response.json();
        if (data && (data.error || data.user_message || data.detail || data.message)) {
          errMessage = data.error || data.user_message || (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) || data.message;
        }
      } catch {
        // Non-JSON response
      }
      throw new Error(errMessage);
    }

    return response.json();
  },

  /**
   * Reconstruct route from GPS observations
   */
  async reconstructRoute(req: ReconstructionRequest): Promise<ReconstructionResponse> {
    return request<ReconstructionResponse>('/reconstruct', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  /**
   * List all evaluation trips
   */
  async listEvaluationTrips(): Promise<EvaluationListResponse> {
    return request<EvaluationListResponse>('/evaluation/trips');
  },

  /**
   * Get single evaluation trip details
   */
  async getEvaluationTrip(tripId: string): Promise<EvaluationTripDetail> {
    return request<EvaluationTripDetail>(`/evaluation/trips/${encodeURIComponent(tripId)}`);
  },

  /**
   * Reconstruct and evaluate an evaluation trip
   */
  async reconstructEvaluationTrip(
    tripId: string,
    options?: ReconstructionOptions
  ): Promise<EvaluationResponse> {
    return request<EvaluationResponse>(`/evaluation/trips/${encodeURIComponent(tripId)}/reconstruct`, {
      method: 'POST',
      body: JSON.stringify({
        trip_id: tripId,
        options: options || null,
      }),
    });
  },

  /**
   * Parse JSON string containing GPS points
   */
  parseJSON(jsonStr: string): GPSPoint[] {
    const parsed = JSON.parse(jsonStr);
    const rawList = Array.isArray(parsed)
      ? parsed
      : Array.isArray(parsed.points)
      ? parsed.points
      : Array.isArray(parsed.gps_points)
      ? parsed.gps_points
      : null;

    if (!rawList || rawList.length === 0) {
      throw new Error('JSON must contain an array of GPS points or an object with a "points" property');
    }

    return rawList.map((item: any, idx: number) => {
      const lat = item.latitude ?? item.lat;
      const lon = item.longitude ?? item.lon ?? item.lng;
      const time = item.timestamp ?? item.time ?? new Date().toISOString();

      if (typeof lat !== 'number' || typeof lon !== 'number') {
        throw new Error(`Point at index ${idx} is missing valid latitude/longitude numbers`);
      }

      return {
        latitude: lat,
        longitude: lon,
        timestamp: typeof time === 'string' ? time : new Date(time).toISOString(),
        speed: typeof item.speed === 'number' ? item.speed : undefined,
        heading: typeof item.heading === 'number' ? item.heading : undefined,
        accuracy: typeof item.accuracy === 'number' ? item.accuracy : undefined,
      };
    });
  },

  /**
   * Parse a File object (CSV, GPX, JSON) into GPS points
   */
  async parseGPSFile(file: File): Promise<GPSPoint[]> {
    const text = await file.text();
    const name = file.name.toLowerCase();

    if (name.endsWith('.json') || text.trim().startsWith('{') || text.trim().startsWith('[')) {
      return this.parseJSON(text);
    }

    if (name.endsWith('.gpx') || text.includes('<gpx')) {
      const parser = new DOMParser();
      const xmlDoc = parser.parseFromString(text, 'text/xml');
      const trkpts = xmlDoc.getElementsByTagName('trkpt');

      if (trkpts.length === 0) {
        throw new Error('No <trkpt> points found in GPX file');
      }

      const points: GPSPoint[] = [];
      for (let i = 0; i < trkpts.length; i++) {
        const pt = trkpts[i];
        const lat = parseFloat(pt.getAttribute('lat') || '');
        const lon = parseFloat(pt.getAttribute('lon') || '');
        const timeEl = pt.getElementsByTagName('time')[0];
        const timestamp = timeEl ? timeEl.textContent || new Date().toISOString() : new Date().toISOString();
        const speedEl = pt.getElementsByTagName('speed')[0];
        const speed = speedEl && speedEl.textContent ? parseFloat(speedEl.textContent) : undefined;

        if (!isNaN(lat) && !isNaN(lon)) {
          points.push({
            latitude: lat,
            longitude: lon,
            timestamp,
            speed: isNaN(speed as number) ? undefined : speed,
          });
        }
      }

      if (points.length === 0) {
        throw new Error('No valid track points could be parsed from GPX file');
      }
      return points;
    }

    // Otherwise assume CSV
    const lines = text.trim().split(/\r?\n/).filter(line => line.trim() !== '');
    if (lines.length < 2) {
      throw new Error('CSV file must have a header line and at least one data row');
    }

    const header = lines[0].split(',').map(h => h.trim().toLowerCase());
    let latIdx = header.findIndex(h => h === 'latitude' || h === 'lat');
    let lonIdx = header.findIndex(h => h === 'longitude' || h === 'lon' || h === 'lng');
    let timeIdx = header.findIndex(h => h === 'timestamp' || h === 'time' || h === 'datetime');
    let speedIdx = header.findIndex(h => h === 'speed');

    if (latIdx === -1 || lonIdx === -1) {
      // Fallback: check index 0, 1
      latIdx = 0;
      lonIdx = 1;
      timeIdx = 2;
    }

    const points: GPSPoint[] = [];
    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(',').map(p => p.trim());
      if (parts.length <= Math.max(latIdx, lonIdx)) continue;

      const lat = parseFloat(parts[latIdx]);
      const lon = parseFloat(parts[lonIdx]);
      const timestamp = timeIdx !== -1 && parts[timeIdx] ? parts[timeIdx] : new Date().toISOString();
      const speed = speedIdx !== -1 && parts[speedIdx] ? parseFloat(parts[speedIdx]) : undefined;

      if (!isNaN(lat) && !isNaN(lon)) {
        points.push({
          latitude: lat,
          longitude: lon,
          timestamp,
          speed: isNaN(speed as number) ? undefined : speed,
        });
      }
    }

    if (points.length === 0) {
      throw new Error('No valid GPS data rows found in CSV');
    }

    return points;
  },
};
