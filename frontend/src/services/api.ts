
const API_BASE = (
  import.meta.env.VITE_API_URL ||
  "https://smart-hospitality-management-system.onrender.com/api"
).replace(/^["']|["']$/g, "").replace(/\/+$/, "");

if (!API_BASE.startsWith("https://") && !API_BASE.startsWith("http://")) {
  throw new Error(`Invalid VITE_API_URL: ${API_BASE}`);
}

async function parseResponse(response: Response) {
  const raw = await response.text();

  if (!response.ok) {
    let message = raw || `HTTP ${response.status}`;

    try {
      const errorData = JSON.parse(raw);
      message =
        errorData?.detail?.[0]?.msg ??
        errorData?.detail ??
        message;
    } catch {
      // Keep the original response text for HTML or other errors.
    }

    throw new Error(
      `API ${response.status}: ${String(message).slice(0, 300)}`
    );
  }

  if (!raw.trim()) {
    throw new Error(
      `Empty API response (HTTP ${response.status}) from ${response.url}`
    );
  }

  try {
    return JSON.parse(raw);
  } catch {
    throw new Error(
      `Expected JSON from ${response.url}; received: ${raw.slice(0, 200)}`
    );
  }
}

export async function predictBooking(payload: Record<string, unknown>) {
  const response = await fetch(`${API_BASE}/booking/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  return parseResponse(response);
}

export async function predictSentiment(
  text: string,
  model: string = "tfidf_logistic_regression"
) {
  const response = await fetch(`${API_BASE}/sentiment/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, model }),
  });

  return parseResponse(response);
}

export async function getDemandForecast(
  horizon: number,
  model: string = "random_forest"
) {
  const response = await fetch(
    `${API_BASE}/demand/forecast?horizon=${horizon}&model=${encodeURIComponent(model)}`
  );

  return parseResponse(response);
}

export async function predictFood(file: File) {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${API_BASE}/food/predict`, {
    method: "POST",
    body: form,
  });

  return parseResponse(response);
}
