import { inject } from 'aurelia';
import { ApiService, DraftSummary, JobResponse } from '../../services/api-service';

@inject(ApiService)
export class Job {
  id = '';
  job: JobResponse | null = null;
  draftSummary: DraftSummary | null = null;
  isLoading = true;
  isLoadingDraft = false;
  error = '';
  draftError = '';
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

  get draftUrl(): string {
    return this.api.getDraftUrl(this.id);
  }

  get hasWarnings(): boolean {
    return Boolean(this.draftSummary?.quality.warnings.length);
  }

  get tempoLabel(): string {
    const summary = this.draftSummary;
    if (!summary) return 'Pending';
    const tempo = summary.metadata.tempo || summary.constraints.tempo_bpm;
    if (!tempo) return 'Unknown';
    return summary.constraints.tempo_source === 'user' ? `${tempo} BPM set` : `${tempo} BPM detected`;
  }

  get meterLabel(): string {
    const constraints = this.draftSummary?.constraints;
    if (!constraints) return 'Pending';
    const meter = constraints.time_signature || '4/4';
    return constraints.time_signature_source === 'user' ? `${meter} set` : `${meter} assumed`;
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
      if (this.job.status === 'completed' && !this.draftSummary) {
        await this.fetchDraftSummary();
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

  async fetchDraftSummary(): Promise<void> {
    if (!this.id || this.isLoadingDraft) return;
    this.isLoadingDraft = true;
    this.draftError = '';

    try {
      this.draftSummary = await this.api.getDraftSummary(this.id);
    } catch (e) {
      this.draftError = e instanceof Error ? e.message : 'Unable to load draft summary.';
    } finally {
      this.isLoadingDraft = false;
    }
  }
}
