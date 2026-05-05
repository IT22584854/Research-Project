import { useState } from "react";
import Turns from "./pages/Turns";
import EvaluateTurn from "./pages/EvaluateTurn";
import SessionEval from "./pages/SessionEval";
import BatchEval from "./pages/BatchEval";
import ManualEval from "./pages/ManualEval";
import "./App.css";

export default function App() {
  const [tab, setTab] = useState("turns");

  return (
    <div className="app-container">
      <div className="app-header">
        <div className="header-content">
          <h1 className="app-title">RAG Evaluation Dashboard</h1>
          <p className="app-subtitle">Comprehensive evaluation suite for RAG systems</p>
        </div>
      </div>

      <div className="tab-navigation">
        <button 
          className={`tab-button ${tab === "turns" ? "active" : ""}`}
          onClick={() => setTab("turns")}
        >
          📋 All Turns
        </button>
        <button 
          className={`tab-button ${tab === "evaluate" ? "active" : ""}`}
          onClick={() => setTab("evaluate")}
        >
          🔍 Evaluate Turn
        </button>
        <button 
          className={`tab-button ${tab === "session" ? "active" : ""}`}
          onClick={() => setTab("session")}
        >
          � Session Evaluation
        </button>
        <button 
          className={`tab-button ${tab === "batch" ? "active" : ""}`}
          onClick={() => setTab("batch")}
        >
          🚀 Batch Evaluation
        </button>
        <button 
          className={`tab-button ${tab === "manual" ? "active" : ""}`}
          onClick={() => setTab("manual")}
        >
          ✍️ Manual Eval
        </button>
      </div>

      <div className="main-content">
        {tab === "turns" && <Turns />}
        {tab === "evaluate" && <EvaluateTurn />}
        {tab === "session" && <SessionEval />}
        {tab === "batch" && <BatchEval />}
        {tab === "manual" && <ManualEval />}
      </div>
    </div>
  );
}