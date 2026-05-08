import { bindable } from 'aurelia';

export class ProgressBar {
  @bindable value = 0;

  get safeValue(): number {
    return Math.min(100, Math.max(0, Number(this.value) || 0));
  }
}
