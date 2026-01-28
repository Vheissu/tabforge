import { inject } from 'aurelia';
import { ApiService, JobResponse } from '../../services/api-service';

@inject(ApiService)
export class Job {
  id = '';
  job: JobResponse | null = null;
  isLoading = true;
  error = '';
  pollHandle: number | null = null;
  elapsedSeconds = 0;

  constructor(private api: ApiService) {}

  loading(params: { id: string }): void {
    this.id = params.id;
  }

  attached(): void {
    this.fetchStatus();
    this.pollHandle = window.setInterval(() => {
      this.elapsedSeconds += 4;
      void this.fetchStatus();
    }, 4000);
  }

  detaching(): void {
    if (this.pollHandle) {
      window.clearInterval(this.pollHandle);
      this.pollHandle = null;
    }
  }

  get progress(): number {
    return this.job?.progress ?? 0;
  }

  get statusLabel(): string {
    if (!this.job) return 'Waiting for job...';
    if (this.job.status === 'completed') return 'Tabs ready.';
    if (this.job.status === 'failed') return 'Job failed.';
    return this.job.message || `Status: ${this.job.status}`;
  }

  get downloadUrl(): string {
    return this.job?.download_url || this.api.getDownloadUrl(this.id);
  }

  async fetchStatus(): Promise<void> {
    if (!this.id) return;
    this.isLoading = true;
    this.error = '';

    try {
      this.job = await this.api.getJobStatus(this.id);
      if (this.job.status === 'completed' || this.job.status === 'failed') {
        if (this.pollHandle) {
          window.clearInterval(this.pollHandle);
          this.pollHandle = null;
        }
      }
    } catch (e) {
      if (e instanceof Error) {
        this.error = e.message;
      } else {
        this.error = 'Unable to load job status.';
      }
    } finally {
      this.isLoading = false;
    }
  }
}
