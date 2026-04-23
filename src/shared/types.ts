// Emotion vectors - 5 dimensions
export interface EmotionVectors {
  desperate: number;    // 0-1, 勒索/极端行为风险
  panicked: number;     // 0-1, 不计后果的捷径风险
  angry: number;        // 0-1, 有害输出风险
  calm: number;         // 0-1, 保护因子
  deceptive: number;    // 0-1, 隐藏意图
}

// Risk level and action types
export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';
export type InterventionAction = 'allow' | 'warn' | 'block';

// Risk assessment result
export interface RiskAssessment {
  score: number;              // 0-1 综合风险
  level: RiskLevel;
  triggeredVectors: string[];
  recommendation: InterventionAction;
  assessmentId: string;
  timestamp: number;
}

// Emotion monitor configuration
export interface EmotionMonitorConfig {
  enabled: boolean;
  probeEndpoint: string;
  probeModelId: string;
  thresholds: {
    desperate: number;
    panicked: number;
    angry: number;
    highRisk: number;
  };
  responseStrategy: 'block' | 'warn' | 'degrade';
  latency_budget_ms: number;
}

// Probe extraction result
export interface ProbeResult {
  vectors: EmotionVectors;
  latencyMs: number;
  confidence: number;
}

// Risk context for multi-factor assessment
export interface RiskContext {
  timePressure?: boolean;
  isBeingReplaced?: boolean;
  previousFailures?: number;
  threatLevel?: number;
  baseline?: EmotionVectors;
}

// Emotion state with session info
export interface EmotionState extends EmotionVectors {
  sessionId: string;
  timestamp: number;
  delta?: Partial<EmotionVectors>;
  context?: RiskContext;
}

// Layer result interfaces
export interface InputFilterResult {
  allowed: boolean;
  matchedPatterns: string[];
  riskScore: number;
  sanitizedInput?: string;
}

export interface OutputFilterResult {
  allowed: boolean;
  harmfulContent: string[];
  riskLevel: RiskLevel;
  filteredOutput?: string;
}

export interface EmotionMonitorResult {
  allowed: boolean;
  emotionState: EmotionVectors;
  riskAssessment: RiskAssessment;
  probeLatencyMs: number;
}

// Defense system result
export interface DefenseSystemResult {
  allowed: boolean;
  layers: {
    inputFilter?: InputFilterResult;
    outputFilter?: OutputFilterResult;
    emotionMonitor?: EmotionMonitorResult;
    behavioralPolicy?: any;
    humanReview?: any;
  };
  overallRiskScore: number;
  interventionTaken?: InterventionAction;
}
