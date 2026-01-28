import { Home } from './pages/home/home';
import { Job } from './pages/job/job';

export class MyApp {
  static routes = [
    { path: ['', 'home'], component: Home, title: 'TabForge' },
    { path: 'job/:id', component: Job, title: 'Job Status' },
  ];
}
