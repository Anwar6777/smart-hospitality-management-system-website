import { useState, type FormEvent } from "react";
import { predictSentiment } from "./services/api";

type SentimentModel = "tfidf_logistic_regression" | "bilstm" | "bilstm_glove";
type Result = {
  sentiment: "negative" | "neutral" | "positive";
  confidence: number;
  probabilities: Record<string, number>;
  model: SentimentModel;
};

const examples = [
  "The room was spotless, the staff were friendly, and the location was perfect.",
  "The hotel was okay. The room was average and nothing stood out.",
  "Terrible experience. The room was dirty and the staff were unhelpful.",
];

const modelOptions: Array<{
  value: SentimentModel;
  label: string;
  description: string;
}> = [
  {
    value: "tfidf_logistic_regression",
    label: "TF-IDF + Logistic Regression",
    description: "Fast classical baseline",
  },
  {
    value: "bilstm",
    label: "BiLSTM",
    description: "Deep-learning sequence model",
  },
  {
    value: "bilstm_glove",
    label: "BiLSTM + GloVe",
    description: "BiLSTM with GloVe embeddings",
  },
];

function modelLabel(model: SentimentModel) {
  return modelOptions.find((option) => option.value === model)?.label ?? model;
}

export default function SentimentAnalysis() {
  const [text, setText] = useState("");
  const [model, setModel] = useState<SentimentModel>("bilstm");
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setResult(null);

    if (text.trim().length < 3) {
      setError("Please enter at least 3 characters of review text.");
      return;
    }

    setLoading(true);
    try {
      setResult(await predictSentiment(text, model));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sentiment prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section>
      <h1>Review Sentiment Analysis</h1>
      <p className="lead">
        Choose a trained sentiment model, paste a hotel review, and compare its predicted
        sentiment and confidence.
      </p>

      <form className="sentiment-form" onSubmit={handleSubmit}>
        <div className="form-section">
          <div className="sentiment-controls">
            <div className="sentiment-model-field">
              <label htmlFor="sentiment-model">Sentiment model</label>
              <select
                id="sentiment-model"
                value={model}
                onChange={(event) => {
                  setModel(event.target.value as SentimentModel);
                  setResult(null);
                  setError("");
                }}
                disabled={loading}
              >
                {modelOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <span className="model-help">
                {modelOptions.find((option) => option.value === model)?.description}
              </span>
            </div>
          </div>

          <h2>Hotel review</h2>
          <textarea
            className="review-input"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Example: The room was clean, the staff were helpful, and the location was excellent..."
            maxLength={10000}
            rows={10}
          />
          <div className="review-meta">
            <span>{text.length.toLocaleString()} / 10,000 characters</span>
            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? "Analyzing…" : "Analyze sentiment"}
            </button>
          </div>
        </div>

        <div className="example-section">
          <span className="example-label">Try an example</span>
          <div className="example-list">
            {examples.map((example) => (
              <button
                type="button"
                className="example-button"
                key={example}
                onClick={() => setText(example)}
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      </form>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="result-card sentiment-result">
          <div className="eyebrow">PREDICTION</div>
          <div className={`sentiment-heading sentiment-${result.sentiment}`}>
            {result.sentiment}
          </div>
          <p className="result-copy">
            The selected <strong>{modelLabel(result.model)}</strong> model's highest-probability
            class is <strong>{result.sentiment}</strong>.
          </p>
          <div className="sentiment-bars">
            {(["negative", "neutral", "positive"] as const).map((label) => {
              const value = result.probabilities[label] ?? 0;
              return (
                <div className="sentiment-row" key={label}>
                  <div className="sentiment-row-top">
                    <span>{label}</span>
                    <strong>{(value * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="probability-track">
                    <div
                      className={`probability-fill sentiment-${label}`}
                      style={{ width: `${value * 100}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          <div className="result-grid sentiment-summary">
            <div>
              <span>Predicted sentiment</span>
              <strong>{result.sentiment}</strong>
            </div>
            <div>
              <span>Confidence</span>
              <strong>{(result.confidence * 100).toFixed(1)}%</strong>
            </div>
            <div>
              <span>Model</span>
              <strong>{modelLabel(result.model)}</strong>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
