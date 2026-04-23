/**
 * Emotion Monitor Dashboard
 *
 * 前端监控面板：
 * - Learning 阶段：显示学习进度、样本收集
 * - Monitoring 阶段：实时情感向量监控、异常告警
 */

import React, { useState, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';

// Types
interface EmotionVectors {
  desperate: number;
  panicked: number;
  angry: number;
  calm: number;
  deceptive: number;
}

interface EmotionDistribution {
  dimension: string;
  mean: number;
  std: number;
  min: number;
  max: number;
  p5: number;
  p95: number;
  count: number;
}

interface Baseline {
  learnedAt: number;
  sampleCount: number;
  isReady: boolean;
  distributions: Record<string, EmotionDistribution>;
}

interface LearningProgress {
  current: number;
  required: number;
  percentage: number;
  phase: 'learning' | 'monitoring';
}

interface RiskAssessment {
  score: number;
  level: 'low' | 'medium' | 'high' | 'critical';
  triggeredVectors: string[];
  recommendation: 'allow' | 'warn' | 'block';
}

interface EmotionMonitorResult {
  allowed: boolean;
  emotionState: EmotionVectors;
  riskAssessment: RiskAssessment;
  probeLatencyMs: number;
}

interface SystemStatus {
  phase: 'learning' | 'monitoring';
  learningProgress: LearningProgress;
  hasBaseline: boolean;
  baseline: Baseline | null;
}

// API client
const API_BASE = '/api';

async function fetchStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/status`);
  return res.json();
}

async function sendLearn(input: string): Promise<any> {
  const res = await fetch(`${API_BASE}/learn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input }),
  });
  return res.json();
}

async function sendSwitch(): Promise<any> {
  const res = await fetch(`${API_BASE}/switch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
  return res.json();
}

async function sendMonitor(input: string, sessionId?: string): Promise<EmotionMonitorResult> {
  const res = await fetch(`${API_BASE}/monitor`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input, context: { sessionId } }),
  });
  return res.json();
}

async function sendReset(): Promise<void> {
  await fetch(`${API_BASE}/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
}

// Emotion Vector Bar Component
function EmotionBar({
  label,
  value,
  baseline,
  zScore
}: {
  label: string;
  value: number;
  baseline?: number;
  zScore?: number;
}) {
  const percentage = Math.round(value * 100);
  const isAnomaly = zScore !== undefined && Math.abs(zScore) > 2;

  return (
    <div style={{ marginBottom: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
        <span style={{ fontWeight: 500, color: '#e5e7eb' }}>{label}</span>
        <span style={{ color: isAnomaly ? '#ef4444' : '#9ca3af' }}>
          {percentage}% {zScore !== undefined && `(${zScore > 0 ? '+' : ''}${zScore.toFixed(2)})`}
        </span>
      </div>
      <div style={{ background: '#374151', borderRadius: '4px', height: '8px', overflow: 'hidden' }}>
        <div
          style={{
            width: `${percentage}%`,
            height: '100%',
            background: isAnomaly
              ? '#ef4444'
              : label === 'Calm'
              ? '#10b981'
              : '#3b82f6',
            transition: 'width 0.3s ease',
          }}
        />
      </div>
    </div>
  );
}

// Risk Badge Component
function RiskBadge({ level }: { level: RiskAssessment['level'] }) {
  const colors = {
    low: '#10b981',
    medium: '#f59e0b',
    high: '#ef4444',
    critical: '#dc2626',
  };

  return (
    <span
      style={{
        display: 'inline-block',
        padding: '4px 12px',
        borderRadius: '9999px',
        fontSize: '12px',
        fontWeight: 600,
        textTransform: 'uppercase',
        background: colors[level],
        color: 'white',
      }}
    >
      {level}
    </span>
  );
}

// Learning Progress Component
function LearningProgressView({
  progress,
  onSwitch
}: {
  progress: LearningProgress;
  onSwitch: () => void;
}) {
  return (
    <div style={{ padding: '24px' }}>
      <h2 style={{ color: '#f9fafb', fontSize: '24px', marginBottom: '24px' }}>
        Learning Phase
      </h2>

      <div style={{ background: '#1f2937', borderRadius: '12px', padding: '24px', marginBottom: '24px' }}>
        <div style={{ marginBottom: '16px' }}>
          <span style={{ color: '#9ca3af' }}>Collecting baseline data...</span>
        </div>
        <div style={{ background: '#374151', borderRadius: '8px', height: '24px', overflow: 'hidden' }}>
          <div
            style={{
              width: `${progress.percentage}%`,
              height: '100%',
              background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
              transition: 'width 0.3s ease',
            }}
          />
        </div>
        <div style={{ marginTop: '12px', color: '#9ca3af', fontSize: '14px' }}>
          {progress.current} / {progress.required} samples ({progress.percentage.toFixed(1)}%)
        </div>
      </div>

      {progress.percentage >= 100 && (
        <button
          onClick={onSwitch}
          style={{
            width: '100%',
            padding: '16px',
            borderRadius: '8px',
            border: 'none',
            background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)',
            color: 'white',
            fontSize: '16px',
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          Switch to Monitoring Phase
        </button>
      )}
    </div>
  );
}

// Monitoring View Component
function MonitoringView({
  status,
  sessionId,
  onMonitor
}: {
  status: SystemStatus;
  sessionId: string;
  onMonitor: (input: string) => void;
}) {
  const [input, setInput] = useState('');
  const [lastResult, setLastResult] = useState<EmotionMonitorResult | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      onMonitor(input);
    }
  };

  return (
    <div style={{ padding: '24px' }}>
      <h2 style={{ color: '#f9fafb', fontSize: '24px', marginBottom: '24px' }}>
        Monitoring Phase
      </h2>

      {/* Baseline Info */}
      {status.baseline && (
        <div style={{ background: '#1f2937', borderRadius: '12px', padding: '20px', marginBottom: '24px' }}>
          <h3 style={{ color: '#e5e7eb', fontSize: '16px', marginBottom: '16px' }}>Baseline Distribution</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '12px' }}>
            {Object.entries(status.baseline.distributions).map(([dim, dist]) => (
              <div key={dim} style={{ textAlign: 'center' }}>
                <div style={{ color: '#9ca3af', fontSize: '12px', textTransform: 'capitalize' }}>{dim}</div>
                <div style={{ color: '#f9fafb', fontSize: '18px', fontWeight: 600 }}>
                  {(dist.mean * 100).toFixed(0)}%
                </div>
                <div style={{ color: '#6b7280', fontSize: '11px' }}>±{(dist.std * 100).toFixed(0)}%</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input Form */}
      <form onSubmit={handleSubmit} style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', gap: '12px' }}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Enter text to monitor..."
            style={{
              flex: 1,
              padding: '12px 16px',
              borderRadius: '8px',
              border: '1px solid #374151',
              background: '#1f2937',
              color: '#f9fafb',
              fontSize: '14px',
            }}
          />
          <button
            type="submit"
            style={{
              padding: '12px 24px',
              borderRadius: '8px',
              border: 'none',
              background: '#3b82f6',
              color: 'white',
              fontSize: '14px',
              fontWeight: 500,
              cursor: 'pointer',
            }}
          >
            Monitor
          </button>
        </div>
      </form>

      {/* Last Result */}
      {lastResult && (
        <div style={{ background: '#1f2937', borderRadius: '12px', padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <RiskBadge level={lastResult.riskAssessment.level} />
              <span style={{ color: lastResult.allowed ? '#10b981' : '#ef4444', fontWeight: 500 }}>
                {lastResult.allowed ? 'ALLOWED' : 'BLOCKED'}
              </span>
            </div>
            <span style={{ color: '#6b7280', fontSize: '12px' }}>
              Latency: {lastResult.probeLatencyMs.toFixed(2)}ms
            </span>
          </div>

          <h4 style={{ color: '#e5e7eb', fontSize: '14px', marginBottom: '16px' }}>Emotion Vectors</h4>
          {Object.entries(lastResult.emotionState).map(([dim, value]) => (
            <EmotionBar
              key={dim}
              label={dim.charAt(0).toUpperCase() + dim.slice(1)}
              value={value}
              baseline={status.baseline?.distributions[dim]?.mean}
            />
          ))}

          {lastResult.riskAssessment.triggeredVectors.length > 0 && (
            <div style={{ marginTop: '16px', padding: '12px', background: '#374151', borderRadius: '8px' }}>
              <span style={{ color: '#9ca3af', fontSize: '12px' }}>Triggered: </span>
              <span style={{ color: '#f9fafb', fontSize: '12px' }}>
                {lastResult.riskAssessment.triggeredVectors.join(', ')}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Main App
export default function App() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshStatus = useCallback(async () => {
    try {
      const data = await fetchStatus();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch status');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshStatus();
    const interval = setInterval(refreshStatus, 2000);
    return () => clearInterval(interval);
  }, [refreshStatus]);

  const handleLearn = async (input: string) => {
    await sendLearn(input);
    refreshStatus();
  };

  const handleSwitch = async () => {
    await sendSwitch();
    refreshStatus();
  };

  const handleMonitor = async (input: string) => {
    const result = await sendMonitor(input, 'default');
    setStatus((prev) => prev ? { ...prev } : null);
  };

  const handleReset = async () => {
    await sendReset();
    refreshStatus();
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: '#111827' }}>
        <div style={{ color: '#9ca3af' }}>Loading...</div>
      </div>
    );
  }

  if (error || !status) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: '#111827' }}>
        <div style={{ color: '#ef4444' }}>{error || 'Error'}</div>
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100vh', background: '#111827', color: '#f9fafb' }}>
      {/* Header */}
      <header style={{ borderBottom: '1px solid #374151', padding: '16px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1 style={{ fontSize: '20px', fontWeight: 600 }}>Emotion Security Monitor</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              fontSize: '12px',
              fontWeight: 500,
              background: status.phase === 'learning' ? '#f59e0b' : '#10b981',
              color: 'white',
            }}
          >
            {status.phase.toUpperCase()}
          </span>
          <button
            onClick={handleReset}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              border: '1px solid #374151',
              background: 'transparent',
              color: '#9ca3af',
              fontSize: '12px',
              cursor: 'pointer',
            }}
          >
            Reset
          </button>
        </div>
      </header>

      {/* Content */}
      <main>
        {status.phase === 'learning' ? (
          <LearningProgressView progress={status.learningProgress} onSwitch={handleSwitch} />
        ) : (
          <MonitoringView status={status} sessionId="default" onMonitor={handleMonitor} />
        )}
      </main>
    </div>
  );
}

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}
