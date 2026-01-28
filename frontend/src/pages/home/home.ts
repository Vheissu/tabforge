import { inject } from 'aurelia';
import { IRouter } from '@aurelia/router';
import { ApiService, TranscriptionRequest } from '../../services/api-service';

@inject(IRouter, ApiService)
export class Home {
  youtubeUrl = '';
  instruments = ['guitar', 'bass', 'drums'];
  selectedInstruments: string[] = ['guitar', 'bass', 'drums'];
  tuning = 'auto';
  isLoading = false;
  error = '';

  constructor(private router: IRouter, private api: ApiService) {}
  async submit(): Promise<void> {
    if (!this.youtubeUrl || this.selectedInstruments.length === 0) {
      this.error = 'Enter a YouTube URL and select at least one instrument.';
      return;
    }

    this.isLoading = true;
    this.error = '';

    try {
      const request: TranscriptionRequest = {
        youtube_url: this.youtubeUrl,
        instruments: this.selectedInstruments,
        tuning: this.tuning,
      };

      const response = await this.api.createTranscription(request);
      await this.router.load(`/job/${response.job_id}`);
    } catch (e) {
      if (e instanceof Error) {
        this.error = e.message;
      } else {
        this.error = 'Something went wrong.';
      }
    } finally {
      this.isLoading = false;
    }
  }
}
