import { useState } from 'react'
import { predictFood } from './services/api'

export default function FoodClassification() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string>('')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const choose = (f: File | null) => {
    setResult(null); setError(''); setFile(f)
    if (f) setPreview(URL.createObjectURL(f)); else setPreview('')
  }

  const submit = async () => {
    if (!file) return
    setLoading(true); setError('')
    try { setResult(await predictFood(file)) }
    catch (e: any) { setError(e?.response?.data?.detail || e?.message || 'Prediction failed.') }
    finally { setLoading(false) }
  }

  return <div className="page">
    <div className="page-header"><div><h1>Food Image Classification</h1><p>Upload a food image and classify it across all 101 Food-101 categories.</p></div></div>
    <div className="two-column">
      <section className="card">
        <h2>Upload image</h2>
        <label className="upload-zone">
          <input type="file" accept="image/jpeg,image/png,image/webp,image/bmp" onChange={e => choose(e.target.files?.[0] || null)} />
          {preview ? <img src={preview} className="food-preview" alt="Selected food" /> : <div><strong>Choose a food image</strong><span>JPG, PNG, WEBP or BMP · max 10 MB</span></div>}
        </label>
        <button className="primary-button" disabled={!file || loading} onClick={submit}>{loading ? 'Classifying…' : 'Classify image'}</button>
        {error && <div className="error-box">{error}</div>}
      </section>
      <section className="card">
        <h2>Prediction</h2>
        {!result && !error && <p className="muted">Your classification result will appear here.</p>}
        {result && <>
          <div className="prediction-hero"><span>Top prediction</span><strong>{result.prediction.label}</strong><b>{(result.prediction.confidence * 100).toFixed(1)}%</b></div>
          <h3>Top 5 predictions</h3>
          <div className="prediction-list">{result.top_predictions.map((x: any) => <div className="prediction-row" key={x.label}><span>{x.label}</span><strong>{(x.confidence * 100).toFixed(1)}%</strong></div>)}</div>
          <div className="result-meta"><span>Model</span><strong>{result.model}</strong></div>
        </>}
      </section>
    </div>
    <div className="info-banner"><strong>Model:</strong> Trained MobileNetV2 transfer-learning model using the full <strong>Food-101 dataset</strong> (101 classes). The production package stores the trained model and class mapping together in a single <code>model.pkl</code> artifact.</div>
  </div>
}
