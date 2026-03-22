var _a;
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
    base: (_a = process.env.VITE_BASE_PATH) !== null && _a !== void 0 ? _a : "/",
    plugins: [react()],
    build: {
        outDir: "dist",
        emptyOutDir: true
    }
});
