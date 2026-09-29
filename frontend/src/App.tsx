import { NavLink, Route, Routes } from "react-router-dom";
import BookingPrediction from "./pages_BookingPrediction";
import SentimentAnalysis from "./pages_SentimentAnalysis";
import DemandForecast from "./pages_DemandForecast";
import FoodClassification from "./pages_FoodClassification";

const modules = [
  { path: "/booking", title: "Booking Prediction", short: "Bookings", icon: "01", description: "Identify cancellation risk before arrival.", tone: "violet" },
  { path: "/sentiment", title: "Review Sentiment", short: "Reviews", icon: "02", description: "Understand guest feedback instantly.", tone: "blue" },
  { path: "/demand", title: "Demand Forecast", short: "Demand", icon: "03", description: "Plan staffing and capacity with forecasts.", tone: "amber" },
  { path: "/food", title: "Food Classification", short: "Food AI", icon: "04", description: "Classify food images with computer vision.", tone: "green" },
];

function Dashboard() {
  return (
    <section className="dashboard-page">
      <div className="hero-card">
        <div className="hero-copy">
          <div className="eyebrow">SMART HOSPITALITY ·</div>
          <h1>One intelligent workspace for modern hospitality.</h1>
          <p className="lead">Four machine-learning tools, one clean workspace — helping teams understand bookings, guests, demand, and food operations faster.</p>
          <div className="hero-actions">
            <NavLink className="primary-button hero-button" to="/booking">Open booking AI <span>→</span></NavLink>
            <NavLink className="secondary-button hero-button" to="/demand">View demand forecast</NavLink>
          </div>
        </div>
        <div className="hero-orbit" aria-hidden="true">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="orbit-core">SH</div>
          <span className="orbit-dot dot-one">01</span><span className="orbit-dot dot-two">02</span><span className="orbit-dot dot-three">03</span><span className="orbit-dot dot-four">04</span>
        </div>
      </div>

      <div className="section-heading">
        <div><div className="eyebrow">AI MODULES</div><h2>Choose a workspace</h2></div>
        <span className="module-count">4 integrated modules</span>
      </div>
      <div className="module-grid">
        {modules.map((module) => (
          <NavLink className={`module-card module-${module.tone}`} to={module.path} key={module.path}>
            <div className="module-top"><span className="module-number">{module.icon}</span><span className="module-arrow">↗</span></div>
            <div className="module-icon">✦</div>
            <h3>{module.title}</h3><p>{module.description}</p>
            <span className="module-link">Launch module <b>→</b></span>
          </NavLink>
        ))}
      </div>

      <div className="dashboard-bottom">
        <div className="status-card modern-status">
          <div className="status-main"><span className="status-dot" /><div><strong>AI services online</strong><span>Backend inference API is ready for development.</span></div></div>
          <span className="api-pill">API v0.5.0</span>
        </div>
        <div className="tip-card"><span className="tip-icon">✦</span><div><strong>Built for mobile too</strong><span>Use the menu below to move between AI tools on smaller screens.</span></div></div>
      </div>
    </section>
  );
}

export default function App() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark">SH</div><div><strong>Smart Hospitality</strong><span>Management System</span></div></div>
        <nav className="desktop-nav">
          <NavLink to="/" end><span className="nav-icon">⌂</span>Dashboard</NavLink>
          {modules.map((module) => <NavLink to={module.path} key={module.path}><span className="nav-number">{module.icon}</span>{module.title}</NavLink>)}
        </nav>
        <div className="sidebar-footer"><span>Platform status</span><strong><i /> All systems ready</strong></div>
      </aside>

      <main className="main">
        <header className="topbar"><div className="mobile-brand"><div className="brand-mark">SH</div><strong>Smart Hospitality</strong></div><div className="environment"><span className="environment-dot" /> Development environment <span className="api-pill">v0.5.0</span></div></header>
        <div className="content"><Routes><Route path="/" element={<Dashboard />} /><Route path="/booking" element={<BookingPrediction />} /><Route path="/sentiment" element={<SentimentAnalysis />} /><Route path="/demand" element={<DemandForecast />} /><Route path="/food" element={<FoodClassification />} /></Routes></div>
      </main>

      <nav className="mobile-nav">
        <NavLink to="/" end><span>⌂</span>Home</NavLink>
        {modules.map((module) => <NavLink to={module.path} key={module.path}><span>{module.icon}</span>{module.short}</NavLink>)}
      </nav>
    </div>
  );
}
