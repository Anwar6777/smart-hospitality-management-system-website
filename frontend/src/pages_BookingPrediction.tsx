import { useState } from "react";
import type { FormEvent } from "react";
import { predictBooking } from "./services/api";

type FormState = Record<string, string | number>;

const initial: FormState = {
  hotel: "City Hotel", lead_time: 30, arrival_date_year: 2017, arrival_date_month: "July",
  arrival_date_week_number: 27, arrival_date_day_of_month: 15, stays_in_weekend_nights: 1,
  stays_in_week_nights: 2, adults: 2, children: 0, babies: 0, meal: "BB", country: "PRT",
  market_segment: "Online TA", distribution_channel: "TA/TO", is_repeated_guest: 0,
  previous_cancellations: 0, previous_bookings_not_canceled: 0, reserved_room_type: "A",
  assigned_room_type: "A", booking_changes: 0, deposit_type: "No Deposit", days_in_waiting_list: 0,
  customer_type: "Transient", adr: 100, required_car_parking_spaces: 0, total_of_special_requests: 0,
  has_agent: 1,
};

const fields: Array<{ key: string; label: string; type?: string }> = [
  { key: "lead_time", label: "Lead time (days)", type: "number" },
  { key: "arrival_date_day_of_month", label: "Arrival day", type: "number" },
  { key: "stays_in_weekend_nights", label: "Weekend nights", type: "number" },
  { key: "stays_in_week_nights", label: "Week nights", type: "number" },
  { key: "adults", label: "Adults", type: "number" },
  { key: "children", label: "Children", type: "number" },
  { key: "babies", label: "Babies", type: "number" },
  { key: "adr", label: "Average Daily Rate", type: "number" },
  { key: "previous_cancellations", label: "Previous cancellations", type: "number" },
  { key: "previous_bookings_not_canceled", label: "Previous bookings not canceled", type: "number" },
  { key: "booking_changes", label: "Booking changes", type: "number" },
  { key: "days_in_waiting_list", label: "Days in waiting list", type: "number" },
  { key: "total_of_special_requests", label: "Special requests", type: "number" },
];

const selects: Array<{ key: string; label: string; options: string[] }> = [
  { key: "hotel", label: "Hotel", options: ["City Hotel", "Resort Hotel"] },
  { key: "arrival_date_month", label: "Arrival month", options: ["January","February","March","April","May","June","July","August","September","October","November","December"] },
  { key: "meal", label: "Meal", options: ["BB","HB","FB","SC","Undefined"] },
  { key: "market_segment", label: "Market segment", options: ["Online TA","Offline TA/TO","Direct","Groups","Corporate","Complementary","Aviation","Undefined"] },
  { key: "distribution_channel", label: "Distribution channel", options: ["TA/TO","Direct","Corporate","GDS","Undefined"] },
  { key: "deposit_type", label: "Deposit type", options: ["No Deposit","Non Refund","Refundable"] },
  { key: "customer_type", label: "Customer type", options: ["Transient","Transient-Party","Contract","Group"] },
  { key: "reserved_room_type", label: "Reserved room type", options: ["A","B","C","D","E","F","G","H","L","P"] },
  { key: "assigned_room_type", label: "Assigned room type", options: ["A","B","C","D","E","F","G","H","I","K","L","P"] },
];

function toPayload(form: FormState) {
  const numeric = new Set(fields.map((x) => x.key));
  return Object.fromEntries(Object.entries(form).map(([key, value]) => [key, numeric.has(key) ? Number(value) : value]));
}

export default function BookingPrediction() {
  const [form, setForm] = useState(initial);
  const [modelName, setModelName] = useState<"random_forest" | "ann">("ann");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function update(key: string, value: string) { setForm((prev) => ({ ...prev, [key]: value })); }

  async function submit(e: FormEvent) {
    e.preventDefault(); setLoading(true); setError(""); setResult(null);
    try { setResult(await predictBooking({ ...toPayload(form), model_name: modelName })); }
    catch (err) { setError(err instanceof Error ? err.message : "Prediction failed"); }
    finally { setLoading(false); }
  }

  return <section>
    <h1>Booking Cancellation Prediction</h1>
    <p className="lead">Enter booking details to estimate the probability that a reservation will be cancelled.</p>

    <form className="prediction-form" onSubmit={submit}>
      <div className="form-section"><h2>Prediction model</h2><div className="form-grid">
        <label>Model<select value={modelName} onChange={(e) => setModelName(e.target.value as "random_forest" | "ann")}>
          <option value="random_forest">Random Forest</option>
          <option value="ann">ANN</option>
        </select></label>
        <div className="model-help"><strong>{modelName === "ann" ? "ANN" : "Random Forest"}</strong><span>{modelName === "ann" ? "Scaled 58-feature neural network." : "300-tree classifier with balanced class weights."}</span></div>
      </div></div>

      <div className="form-section"><h2>Booking details</h2><div className="form-grid">
        {selects.slice(0, 2).map((f) => <label key={f.key}>{f.label}<select value={String(form[f.key])} onChange={(e) => update(f.key,e.target.value)}>{f.options.map(o=><option key={o}>{o}</option>)}</select></label>)}
        {fields.slice(0, 7).map((f) => <label key={f.key}>{f.label}<input type={f.type ?? "text"} value={String(form[f.key])} onChange={(e)=>update(f.key,e.target.value)} /></label>)}
      </div></div>
      <div className="form-section"><h2>Customer & booking behaviour</h2><div className="form-grid">
        {selects.slice(2).map((f) => <label key={f.key}>{f.label}<select value={String(form[f.key])} onChange={(e) => update(f.key,e.target.value)}>{f.options.map(o=><option key={o}>{o}</option>)}</select></label>)}
        {fields.slice(7).map((f) => <label key={f.key}>{f.label}<input type={f.type ?? "text"} value={String(form[f.key])} onChange={(e)=>update(f.key,e.target.value)} /></label>)}
      </div></div>
      <button className="primary-button" disabled={loading}>{loading ? "Predicting…" : "Predict Cancellation"}</button>
    </form>

    {error && <div className="error-box">{error}</div>}
    {result && <div className="result-card">
      <div><div className="eyebrow">MODEL RESULT</div><h2>{result.prediction === "likely_to_cancel" ? "Likely to cancel" : "Likely to stay"}</h2></div>
      <div className="result-grid">
        <div><span>Cancellation probability</span><strong>{(result.cancellation_probability * 100).toFixed(1)}%</strong></div>
        <div><span>Stay probability</span><strong>{(result.stay_probability * 100).toFixed(1)}%</strong></div>
        <div><span>Risk level</span><strong className={`risk-${result.risk_level}`}>{result.risk_level.toUpperCase()}</strong></div>
        <div><span>Model</span><strong>{result.model}</strong></div>
      </div>
    </div>}
  </section>;
}
