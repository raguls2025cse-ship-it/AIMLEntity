/**
 * api.js
 * Centralized API service for EntitySync AI frontend.
 * All calls go through VITE_API_URL — never hard-coded.
 */

const API_BASE = import.meta.env.VITE_API_URL || "";
const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === "true";


/** Generic fetch wrapper with error handling */
async function apiFetch(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const resp = await fetch(url, options);
  if (!resp.ok) {
    let errorDetail = `HTTP ${resp.status}`;
    try {
      const json = await resp.json();
      errorDetail = json.detail || JSON.stringify(json);
    } catch {
      errorDetail = await resp.text();
    }
    throw new Error(errorDetail);
  }
  return resp;
}

/** Check backend health and ML model status */
export async function getBackendHealth() {
  const resp = await apiFetch("/api/health");
  return resp.json();
}

/**
 * Upload all three TSV datasets.
 * @param {File} source1
 * @param {File} source2
 * @param {File} source3
 */
export async function uploadDatasets(source1, source2, source3) {
  const form = new FormData();
  form.append("source1", source1);
  form.append("source2", source2);
  form.append("source3", source3);

  const resp = await apiFetch("/api/upload", {
    method: "POST",
    body: form,
  });
  return resp.json();
}

/**
 * Start the entity matching pipeline.
 * Returns { job_id, status, message }.
 */
export async function runEntityMatching() {
  const resp = await apiFetch("/api/match", { method: "POST" });
  return resp.json();
}

/**
 * Poll job status.
 * @param {string} jobId
 */
export async function getMatchingStatus(jobId) {
  const resp = await apiFetch(`/api/match/status/${jobId}`);
  return resp.json();
}

/**
 * Get matching results with enriched data.
 */
export async function getResults() {
  const resp = await apiFetch("/api/results");
  return resp.json();
}

/**
 * Download matching_results.tsv
 */
export function downloadMatchingResults() {
  window.location.href = `${API_BASE}/api/results/matching`;
}

/**
 * Download candidate_pairs.tsv
 */
export function downloadCandidatePairs() {
  window.location.href = `${API_BASE}/api/results/candidates`;
}
