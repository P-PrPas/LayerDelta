import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

function Workspace() {
  const [state, setState] = useState({ phase: "loading" });
  useEffect(() => {
    fetch("/workspace")
      .then(response => {
        if (!response.ok) throw new Error("Could not open workspace");
        return response.json();
      })
      .then(data => setState({ phase: "ready", ...data }))
      .catch(error => setState({ phase: "error", message: error.message }));
  }, []);
  return <section data-state={state.phase}>
    <h1>LayerDelta workspace</h1>
    <output>{state.phase === "ready" ? state.name : state.message || "Opening workspace…"}</output>
  </section>;
}

createRoot(document.getElementById("workspace")).render(<Workspace />);
