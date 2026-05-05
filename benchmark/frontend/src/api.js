import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000",
});

// -------------------- Turns --------------------
export const getTurns = (limit = 50, offset = 0) =>
  API.get(`/agent-turns?limit=${limit}&offset=${offset}`);

// -------------------- Single Turn Eval --------------------
export const evalHighRiskTurn = (id) =>
  API.post(`/evaluate/turn/high-risk/${id}`);

export const evalGroundTruthTurn = (id) =>
  API.post(`/evaluate/turn/ground-truth/${id}`);

// -------------------- Session Eval --------------------
export const evalHighRiskSession = (id) =>
  API.post(`/evaluate/session/high-risk/${id}`);

export const evalGroundTruthSession = (id) =>
  API.post(`/evaluate/session/ground-truth/${id}`);

// -------------------- Batch --------------------
export const batchHighRisk = (limit = 20) =>
  API.post(`/evaluate/batch/high-risk?limit=${limit}`);

export const batchGroundTruth = (limit = 20) =>
  API.post(`/evaluate/batch/ground-truth?limit=${limit}`);


// -------------------- Manual Evaluation --------------------
export const manualHighRisk = (data) =>
  API.post("/evaluate/high-risk", data);

export const manualGroundTruth = (data) =>
  API.post("/evaluate/ground-truth", data);