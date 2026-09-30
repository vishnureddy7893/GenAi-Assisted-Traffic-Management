/** Application entry point. */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles/global.css";

const container = document.getElementById("root");

// Failing loudly beats rendering nothing: a silently missing mount point looks
// identical to an application that loaded and had nothing to show.
if (container === null) {
  throw new Error('Mount point #root was not found in index.html.');
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
