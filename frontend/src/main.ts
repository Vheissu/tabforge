import { Aurelia, Registration } from 'aurelia';
import { RouterConfiguration } from '@aurelia/router';
import { MyApp } from './my-app';
import { ApiService } from './services/api-service';
import './styles/global.css';

Aurelia.register(RouterConfiguration, Registration.singleton(ApiService, ApiService))
  .app(MyApp)
  .start();
