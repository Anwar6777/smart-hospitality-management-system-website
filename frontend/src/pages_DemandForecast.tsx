import { useEffect, useState } from "react";
import { getDemandForecast } from "./services/api";

type ForecastPoint = { date: string; predicted_demand: number };
type Forecast = {
  model: string;
  model_label: string;
  horizon: number;
  last_historical_date: string;
  forecast_start_date: string;
  forecast_end_date: string;
  total_predicted_demand: number;
  average_daily_demand: number;
  peak_date: string;
  peak_demand: number;
  historical_rows: number;
  metrics: Record<string, number>;
  forecast: ForecastPoint[];
};

const horizons = [7, 14, 30];
const models = [
  { value: "random_forest", label: "Random Forest", help: "Recursive tree-based forecasting with lag, rolling and calendar features." },
  { value: "linear_regression", label: "Linear Regression", help: "Recursive linear baseline using the same engineered demand features." },
  { value: "lstm", label: "LSTM", help: "14-day sequence model using demand and cyclical calendar features." },
  { value: "lstm_improved", label: "Improved LSTM", help: "Detrended residual LSTM using a causal 14-day local average." },
];

export default function DemandForecast() {
  const [horizon, setHorizon] = useState(7);
  const [model, setModel] = useState("lstm_improved");
  const [data, setData] = useState<Forecast | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function loadForecast(nextHorizon = horizon, nextModel = model) {
    setLoading(true);
    setError("");
    try {
      setData(await getDemandForecast(nextHorizon, nextModel));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load forecast");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void loadForecast(7, "random_forest"); }, []);

  const maxDemand = Math.max(...(data?.forecast.map((item) => item.predicted_demand) ?? [1]));
  const selectedModel = models.find((item) => item.value === model) ?? models[0];

  return (
    <section>
      <h1>Hotel Demand Forecasting</h1>
      <p className="lead">
        Forecast expected daily room arrivals with the supplied Random Forest, Linear Regression,
        LSTM, or Improved LSTM demand models.
      </p>

      <div className="forecast-toolbar">
        <div className="demand-model-field">
          <label htmlFor="demand-model">Forecast model</label>
          <select
            id="demand-model"
            value={model}
            disabled={loading}
            onChange={(event) => {
              const nextModel = event.target.value;
              setModel(nextModel);
              void loadForecast(horizon, nextModel);
            }}
          >
            {models.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
          </select>
          <span>{selectedModel.help}</span>
        </div>

        <div>
          <span className="toolbar-label">Forecast horizon</span>
          <div className="horizon-buttons">
            {horizons.map((value) => (
              <button
                key={value}
                className={horizon === value ? "horizon-button active" : "horizon-button"}
                onClick={() => { setHorizon(value); void loadForecast(value, model); }}
                disabled={loading}
              >
                {value} days
              </button>
            ))}
          </div>
        </div>
        <button className="primary-button" onClick={() => void loadForecast()} disabled={loading}>
          {loading ? "Forecasting…" : "Refresh forecast"}
        </button>
      </div>

      {error && <div className="error-box">{error}</div>}

      {data && !error && (
        <>
          <div className="result-grid forecast-summary">
            <div><span>Total predicted arrivals</span><strong>{data.total_predicted_demand.toLocaleString()}</strong></div>
            <div><span>Average / day</span><strong>{data.average_daily_demand}</strong></div>
            <div><span>Peak date</span><strong>{data.peak_date}</strong></div>
            <div><span>Peak demand</span><strong>{data.peak_demand}</strong></div>
          </div>

          <div className="forecast-card">
            <div className="forecast-card-header">
              <div>
                <h2>Forecast timeline</h2>
                <p>Data available through {data.last_historical_date}. Forecast starts on {data.forecast_start_date}.</p>
              </div>
              <span className="model-badge">{data.model_label}</span>
            </div>

            <div className="forecast-bars">
              {data.forecast.map((point) => (
                <div className="forecast-row" key={point.date}>
                  <div className="forecast-date">{point.date}</div>
                  <div className="forecast-track">
                    <div className="forecast-fill" style={{ width: `${Math.max(4, (point.predicted_demand / maxDemand) * 100)}%` }} />
                  </div>
                  <strong>{point.predicted_demand}</strong>
                </div>
              ))}
            </div>
          </div>

          <div className="forecast-card">
            <div className="forecast-card-header">
              <div>
                <h2>Model validation</h2>
                <p>Validation metrics supplied with the trained artifact.</p>
              </div>
            </div>
            {Object.keys(data.metrics).length > 0 ? (
              <div className="result-grid">
                <div><span>MAE</span><strong>{data.metrics.MAE}</strong></div>
                <div><span>RMSE</span><strong>{data.metrics.RMSE}</strong></div>
                <div><span>MAPE</span><strong>{data.metrics.MAPE}%</strong></div>
                <div><span>R²</span><strong>{data.metrics.R2}</strong></div>
              </div>
            ) : (
              <div className="info-banner">The supplied {data.model_label} model does not include validation metric records, so no scores are displayed here.</div>
            )}
          </div>
        </>
      )}
    </section>
  );
}
