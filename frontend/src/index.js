import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import login from './Login';
import Mainapp from './Mainapp'
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import {Loadingtoken} from './Loadingtoken'
import { HashRouter } from 'react-router-dom'
import reportWebVitals from './reportWebVitals';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <Mainapp />
    </BrowserRouter>
  </React.StrictMode>
);



// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();

