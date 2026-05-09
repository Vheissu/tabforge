import { inject } from 'aurelia';
import { IRouter } from '@aurelia/router';
import { ApiService, TranscriptionRequest } from '../../services/api-service';

@inject(IRouter, ApiService)
export class Home {
  youtubeUrl = '';
  instruments = ['guitar', 'bass', 'drums'];
  selectedInstruments: string[] = ['guitar', 'bass', 'drums'];
  tuning = 'auto';
  showSettings = false;
  firstBarTimeSignature = 'auto';
  pickupBarBeats = '';
  firstBarTempoBpm = '';
  tripletFeel = 'auto';
  capoFret = 0;
  showReferenceTab = false;
  referenceTabText = '';
  referenceTabKey = '';
  referenceColumnsPerBeat = '4';
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
        constraints: {
          first_bar_time_signature: this.firstBarTimeSignature,
          pickup_bar_beats: this.pickupBarBeats === '' ? null : Number(this.pickupBarBeats),
          first_bar_tempo_bpm: this.firstBarTempoBpm === '' ? null : Number(this.firstBarTempoBpm),
          triplet_feel: this.tripletFeel as 'auto' | 'straight' | 'triplet',
          capo_fret: Number(this.capoFret) || 0,
        },
      };
      const referenceTab = this.referenceTabText.trim();
      if (referenceTab) {
        request.reference_tab = {
          ascii_tab: referenceTab,
          tempo_bpm: this.firstBarTempoBpm === '' ? null : Number(this.firstBarTempoBpm),
          key: this.referenceTabKey.trim() || null,
          tuning: this.tuning,
          track_name: 'guitar',
          columns_per_beat: Number(this.referenceColumnsPerBeat) || 4,
          default_duration_beats: 0.25,
        };
      }

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

  toggleSettings(): void {
    this.showSettings = !this.showSettings;
  }

  toggleReferenceTab(): void {
    this.showReferenceTab = !this.showReferenceTab;
  }
}
