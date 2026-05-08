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
}
