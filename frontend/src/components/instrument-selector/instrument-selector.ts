import { bindable } from 'aurelia';

export class InstrumentSelector {
  @bindable instruments: string[] = [];
  @bindable selected: string[] = [];

  toggle(instrument: string): void {
    const index = this.selected.indexOf(instrument);
    if (index > -1) {
      this.selected.splice(index, 1);
    } else {
      this.selected.push(instrument);
    }
  }

  isSelected(instrument: string): boolean {
    return this.selected.includes(instrument);
  }
}
