import { bindable } from 'aurelia';

export class InstrumentSelector {
  @bindable instruments: string[] = [];
  @bindable selected: string[] = [];

  toggle(instrument: string): void {
    const index = this.selected.indexOf(instrument);
    if (index > -1) {
      this.selected = this.selected.filter((item) => item !== instrument);
    } else {
      this.selected = [...this.selected, instrument];
    }
  }

  isSelected(instrument: string): boolean {
    return this.selected.includes(instrument);
  }
}
