export interface TranscriptionRequest {
  youtube_url: string;
  instruments: string[];
  tuning?: string;
}

export interface JobResponse {
  job_id: string;
  status: string;
  progress: number;
  message?: string;
  download_url?: string;
  title?: string;
}

export class ApiService {
  private baseUrl = import.meta.env.VITE_API_URL || '/api/v1';

  async createTranscription(request: TranscriptionRequest): Promise<JobResponse> {
    const response = await fetch(`${this.baseUrl}/transcribe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      let detail = 'Failed to create transcription';
      try {
        const error = await response.json();
        detail = error.detail || detail;
      } catch {
        // ignore
      }
      throw new Error(detail);
    }

    return response.json();
  }

  async getJobStatus(jobId: string): Promise<JobResponse> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}`);
    if (!response.ok) {
      throw new Error('Failed to get job status');
    }
    return response.json();
  }

  getDownloadUrl(jobId: string): string {
    return `${this.baseUrl}/download/${jobId}`;
  }
}
