import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(webRoot, "..");
const sourceRoot = path.join(repoRoot, "out", "web_data");
const targetRoot = path.join(webRoot, "public", "web_data");

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function copyRecursive(sourcePath, targetPath) {
  const stats = fs.statSync(sourcePath);
  if (stats.isDirectory()) {
    ensureDir(targetPath);
    for (const entry of fs.readdirSync(sourcePath)) {
      copyRecursive(path.join(sourcePath, entry), path.join(targetPath, entry));
    }
    return;
  }
  ensureDir(path.dirname(targetPath));
  fs.copyFileSync(sourcePath, targetPath);
}

function writeFallbackIndex() {
  const fallback = {
    schema_version: 1,
    latest_date: null,
    dates: [],
    months: [],
    years: []
  };
  ensureDir(path.join(targetRoot, "hot"));
  fs.writeFileSync(path.join(targetRoot, "hot", "index.json"), JSON.stringify(fallback, null, 2));
}

fs.rmSync(targetRoot, { recursive: true, force: true });

if (fs.existsSync(sourceRoot)) {
  copyRecursive(sourceRoot, targetRoot);
} else {
  writeFallbackIndex();
}
