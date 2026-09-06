/**
 * Fetch helpers for the FastAPI backend.
 * In dev, Vite proxies /api → http://127.0.0.1:8000.
 */

async function readError(response) {
  let detail = `Request failed (${response.status})`;
  try {
    const body = await response.json();
    detail = body.detail || body.error || detail;
  } catch {
    /* keep status text */
  }
  const err = new Error(detail);
  err.status = response.status;
  throw err;
}

async function jsonFetch(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  if (!response.ok) await readError(response);
  return response.json();
}

export async function fetchHealth() {
  const response = await fetch("/api/health");
  if (!response.ok) await readError(response);
  return response.json();
}

export async function fetchConfig(market = "miami") {
  return jsonFetch(`/api/config/${market}`);
}

export async function fetchMacro(market = "miami", horizonDays) {
  const query =
    horizonDays != null ? `?horizon_days=${encodeURIComponent(horizonDays)}` : "";
  return jsonFetch(`/api/macro/${market}${query}`);
}

export async function validateInventory(market, rows) {
  return jsonFetch("/api/inventory/validate", {
    method: "POST",
    body: JSON.stringify({ market, rows }),
  });
}

export async function fitDemand(body) {
  return jsonFetch("/api/demand/fit", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function fetchCurrentDemand(market = "miami") {
  return jsonFetch(`/api/demand/current?market=${encodeURIComponent(market)}`);
}

export async function optimizePlan(body) {
  return jsonFetch("/api/optimize", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/**
 * SSE simulate. Calls onProgress for progress events; resolves with the result payload.
 */
export async function simulatePlan(body, { onProgress } = {}) {
  const response = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) await readError(response);
  if (!response.body) throw new Error("Simulate response had no body");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() || "";
    for (const chunk of chunks) {
      const lines = chunk.split("\n");
      let event = "message";
      let data = "";
      for (const line of lines) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!data) continue;
      const payload = JSON.parse(data);
      if (event === "progress" && onProgress) onProgress(payload);
      if (event === "result") result = payload;
    }
  }
  if (!result) throw new Error("Simulate stream ended without a result event");
  return result;
}

export async function fetchSensitivity(body) {
  return jsonFetch("/api/sensitivity", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function fetchDemandCurve(body) {
  return jsonFetch("/api/demand/curve", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
