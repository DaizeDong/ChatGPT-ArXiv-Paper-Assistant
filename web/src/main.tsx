import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

import { App } from "./App";
import "./styles.css";

function routerBaseName() {
  const basePath = import.meta.env.BASE_URL || "/";
  return basePath.endsWith("/") && basePath !== "/" ? basePath.slice(0, -1) : basePath;
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter basename={routerBaseName()}>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
