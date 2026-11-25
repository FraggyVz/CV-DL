// Builds the results view (src/viewer.tsx) as a Streamlit component for the cloud demo.
// Output: deploy/streamlit/results_component/index.html and its assets, committed so
// Streamlit Community Cloud needs no Node. Run `npm run build:streamlit` after changing
// App.tsx's result cards, ModelPipeline.tsx or app.css.
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  root: fileURLToPath(new URL("./viewer", import.meta.url)),
  base: "./",
  plugins: [react()],
  build: {
    outDir: fileURLToPath(new URL("../deploy/streamlit/results_component", import.meta.url)),
    emptyOutDir: true,
  },
});
