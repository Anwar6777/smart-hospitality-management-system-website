const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api";

export async function predictBooking(payload: Record<string, unknown>) {
  const response = await fetch(`${API_BASE}/booking/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail?.[0]?.msg ?? data?.detail ?? "Prediction failed");
  }
  return data;
}

export async function predictSentiment(text: string, model: string = "tfidf_logistic_regression") {
  const response = await fetch(`${API_BASE}/sentiment/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, model }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data?.detail?.[0]?.msg ?? data?.detail ?? "Sentiment prediction failed");
  return data;
}

export async function getDemandForecast(horizon: number, model: string = "random_forest") {
  const response = await fetch(`${API_BASE}/demand/forecast?horizon=${horizon}&model=${encodeURIComponent(model)}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data?.detail ?? "Demand forecast failed");
  return data;
}

export async function predictFood(file: File) {
  const form = new FormData()
  form.append("file", file)
  const response = await fetch(`${API_BASE}/food/predict`, { method: "POST", body: form })
  const data = await response.json()
  if (!response.ok) throw new Error(data?.detail ?? "Food classification failed")
  return data
}
