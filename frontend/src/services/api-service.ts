export interface TranscriptionRequest {
  youtube_url: string;
  instruments: string[];
  tuning?: string;
  constraints?: {
    first_bar_time_signature?: string;
    pickup_bar_beats?: number | null;
    first_bar_tempo_bpm?: number | null;
    triplet_feel?: 'auto' | 'straight' | 'triplet';
    capo_fret?: number;
  };
}

export interface JobResponse {
  job_id: string;
  status: string;
  progress: number;
  message?: string;
  download_url?: string;
  title?: string;
}

export interface DraftWarning {
  code: string;
  severity: 'info' | 'warning' | 'error';
  message: string;
}

export interface DraftTrackSummary {
  name: string;
  source_stem?: string;
  statistics: {
    note_count: number;
    chord_slot_count: number;
    last_beat: number;
  };
  analysis?: Record<string, unknown>;
}

export interface DraftSummary {
  schema_version: string;
  metadata: {
    title?: string;
    artist?: string;
    tempo?: number;
    detected_tempo?: number;
    key?: string;
  };
  constraints: {
    time_signature?: string;
    time_signature_source?: string;
    pickup_bar_beats?: number | null;
    tempo_bpm?: number | null;
    tempo_source?: string;
    triplet_feel?: boolean;
    capo_fret?: number;
  };
  tuning: {
    name: string;
  };
  tracks: DraftTrackSummary[];
  quality: {
    warnings: DraftWarning[];
    next_actions: string[];
  };
  statistics: {
    track_count: number;
    note_count: number;
    last_beat: number;
  };
}

export interface DraftTrackCorrection {
  enabled?: boolean;
  min_velocity?: number | null;
  max_notes_per_slot?: number | null;
}

export interface DraftCorrectionRequest {
  tempo_bpm?: number | null;
  time_signature?: string | null;
  pickup_bar_beats?: number | null;
  tuning?: string | null;
  capo_fret?: number | null;
  triplet_feel?: 'auto' | 'straight' | 'triplet' | null;
  tracks?: Record<string, DraftTrackCorrection>;
}

export class ApiService {
  private baseUrl = import.meta.env.VITE_API_URL || '/api/v1';

  private async parseError(response: Response, fallback: string): Promise<Error> {
    try {
      const error = await response.json();
      const detail = Array.isArray(error.detail)
        ? error.detail.map((item: { msg?: string }) => item.msg).filter(Boolean).join(', ')
        : error.detail;
      return new Error(detail || fallback);
    } catch {
      return new Error(fallback);
    }
  }

  async createTranscription(request: TranscriptionRequest): Promise<JobResponse> {
    const response = await fetch(`${this.baseUrl}/transcribe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw await this.parseError(response, 'Failed to create transcription');
    }

    return response.json();
  }

  async getJobStatus(jobId: string): Promise<JobResponse> {
    const response = await fetch(`${this.baseUrl}/jobs/${encodeURIComponent(jobId)}`);
    if (!response.ok) {
      throw await this.parseError(response, 'Failed to get job status');
    }
    return response.json();
  }

  getDownloadUrl(jobId: string): string {
    return `${this.baseUrl}/download/${encodeURIComponent(jobId)}`;
  }

  getDraftUrl(jobId: string): string {
    return `${this.baseUrl}/draft/${encodeURIComponent(jobId)}`;
  }

  async getDraftSummary(jobId: string): Promise<DraftSummary> {
    const response = await fetch(`${this.baseUrl}/draft/${encodeURIComponent(jobId)}/summary`);
    if (!response.ok) {
      throw await this.parseError(response, 'Failed to get draft summary');
    }
    return response.json();
  }

  async regenerateDraft(jobId: string, request: DraftCorrectionRequest): Promise<DraftSummary> {
    const response = await fetch(`${this.baseUrl}/draft/${encodeURIComponent(jobId)}/regenerate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      throw await this.parseError(response, 'Failed to regenerate draft');
    }
    return response.json();
  }
}
