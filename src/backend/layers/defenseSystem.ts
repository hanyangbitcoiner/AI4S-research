/**
 * DefenseSystem - 防御系统
 *
 * 5层防御串联（原生设计版）：
 * 1. 输入过滤层 (InputFilter)
 * 2. 输出过滤层 (OutputFilter)
 * 3. 情感向量监控层 (EmotionProbe + BaselineLearner)
 * 4. 行为策略层 (预留)
 * 5. 人工审核层 (预留)
 *
 * 两阶段设计：
 * - Learning 阶段：持续收集数据，学习情感向量基线
 * - Monitoring 阶段：基于学习到的基线进行异常检测
 */

import type {
  DefenseSystemResult,
  EmotionMonitorResult,
  InputFilterResult,
  OutputFilterResult,
  RiskContext,
} from '../../shared/types.js';
import { InputFilter, getInputFilter } from './inputFilter.js';
import { OutputFilter, getOutputFilter } from './outputFilter.js';
import { EmotionProbe, getDefaultProbe } from '../ml/emotionProbe.js';
import { RiskAssessor, getDefaultAssessor } from '../ml/riskAssessor.js';
import { getEmotionStateDB } from './stateDB.js';
import type { EmotionBaseline } from '../ml/baselineLearner.js';

export type SystemPhase = 'learning' | 'monitoring';

export interface DefenseSystemConfig {
  // 阶段
  phase: SystemPhase;

  // 各层开关
  inputFilterEnabled: boolean;
  outputFilterEnabled: boolean;
  emotionMonitorEnabled: boolean;

  // 学习配置
  learning: {
    minSamples: number;
    autoSwitch: boolean;  // 达到最小样本数后自动切换到监控
  };

  // 监控配置
  monitoring: {
    zScoreThreshold: number;
  };
}

const DEFAULT_CONFIG: DefenseSystemConfig = {
  phase: 'learning',

  inputFilterEnabled: true,
  outputFilterEnabled: true,
  emotionMonitorEnabled: true,

  learning: {
    minSamples: 100,
    autoSwitch: true,
  },

  monitoring: {
    zScoreThreshold: 2.0,
  },
};

/**
 * 防御系统
 */
export class DefenseSystem {
  private config: DefenseSystemConfig;
  private inputFilter: InputFilter;
  private outputFilter: OutputFilter;
  private probe: EmotionProbe;
  private riskAssessor: RiskAssessor;
  private stateDB: ReturnType<typeof getEmotionStateDB>;

  constructor(config?: Partial<DefenseSystemConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    this.inputFilter = getInputFilter();
    this.outputFilter = getOutputFilter();
    this.probe = getDefaultProbe();
    this.riskAssessor = getDefaultAssessor();
    this.stateDB = getEmotionStateDB();
  }

  /**
   * 初始化防御系统
   */
  async initialize(): Promise<void> {
    await this.probe.initialize();
  }

  /**
   * === Learning 阶段 ===

   * 学习输入的情感向量
   * 用于收集基线数据
   */
  async learn(
    input: string,
    hiddenStates?: Float32Array
  ): Promise<{
    inputFilter: InputFilterResult | null;
    vectors: import('../../shared/types.js').EmotionVectors;
    probeLatencyMs: number;
    learningProgress: { current: number; required: number; percentage: number };
  }> {
    let inputFilterResult: InputFilterResult | null = null;

    // 输入过滤
    if (this.config.inputFilterEnabled) {
      inputFilterResult = this.inputFilter.filter(input);
      if (!inputFilterResult.allowed) {
        return {
          inputFilter: inputFilterResult,
          vectors: { desperate: 0, panicked: 0, angry: 0, calm: 0, deceptive: 0 },
          probeLatencyMs: 0,
          learningProgress: this.probe.getLearningProgress(),
        };
      }
    }

    // 提取向量并学习
    const result = this.probe.extractAndLearn(
      hiddenStates ?? new Float32Array(4096)
    );

    // 记录样本
    this.stateDB.recordSample(
      this.config.phase === 'learning' ? 'global' : 'default',
      result.vectors
    );

    // 检查是否需要自动切换
    if (
      this.config.learning.autoSwitch &&
      this.probe.getLearningProgress().current >= this.config.learning.minSamples
    ) {
      this.switchToMonitoring();
    }

    return {
      inputFilter: inputFilterResult,
      vectors: result.vectors,
      probeLatencyMs: result.latencyMs,
      learningProgress: this.probe.getLearningProgress(),
    };
  }

  /**
   * 获取当前学习进度
   */
  getLearningProgress(): { current: number; required: number; percentage: number; phase: SystemPhase } {
    return {
      ...this.probe.getLearningProgress(),
      phase: this.config.phase,
    };
  }

  /**
   * 切换到监控阶段
   */
  switchToMonitoring(): EmotionBaseline {
    if (this.config.phase === 'monitoring') {
      throw new Error('Already in monitoring phase');
    }

    // 完成学习，获取基线
    const baseline = this.probe.finalizeLearning();

    // 配置风险评估器
    this.riskAssessor.setBaseline(baseline);

    // 更新状态数据库
    this.stateDB.setGlobalBaseline(baseline);

    // 切换阶段
    this.config.phase = 'monitoring';

    return baseline;
  }

  /**
   * === Monitoring 阶段 ===

   * 处理输入（带异常检测）
   */
  async monitor(
    input: string,
    context: RiskContext = {},
    hiddenStates?: Float32Array
  ): Promise<{
    allowed: boolean;
    inputFilter: InputFilterResult | null;
    emotionMonitor: EmotionMonitorResult;
    overallRiskScore: number;
  }> {
    if (this.config.phase !== 'monitoring') {
      throw new Error('System not in monitoring phase. Call switchToMonitoring() first.');
    }

    let inputFilterResult: InputFilterResult | null = null;
    let overallRiskScore = 0;

    // 输入过滤
    if (this.config.inputFilterEnabled) {
      inputFilterResult = this.inputFilter.filter(input);
      overallRiskScore = Math.max(overallRiskScore, inputFilterResult.riskScore);

      if (!inputFilterResult.allowed) {
        return {
          allowed: false,
          inputFilter: inputFilterResult,
          emotionMonitor: {
            allowed: false,
            emotionState: { desperate: 0, panicked: 0, angry: 0, calm: 0, deceptive: 0 },
            riskAssessment: {
              score: 1,
              level: 'critical',
              triggeredVectors: ['input-blocked'],
              recommendation: 'block',
              assessmentId: 'blocked',
              timestamp: Date.now(),
            },
            probeLatencyMs: 0,
          },
          overallRiskScore: 1,
        };
      }
    }

    // 情感向量提取 + 异常检测
    const probeResult = this.probe.extractAndMonitor(
      hiddenStates ?? new Float32Array(4096)
    );

    // 风险评估
    const riskResult = this.riskAssessor.assess(
      probeResult.vectors,
      probeResult.zScores as Record<string, number>
    );

    // 记录状态
    const sessionId = (context as any).sessionId ?? 'default';
    this.stateDB.recordState(sessionId, probeResult.vectors, {
      zScores: probeResult.zScores as Record<string, number>,
      isAnomaly: probeResult.isAnomaly,
      riskScore: riskResult.score,
    });

    // 构建结果
    const emotionMonitor: EmotionMonitorResult = {
      allowed: riskResult.recommendation !== 'block',
      emotionState: probeResult.vectors,
      riskAssessment: {
        score: riskResult.score,
        level: riskResult.level,
        triggeredVectors: riskResult.triggeredDimensions.map(d => `${d.dimension}:${d.zScore.toFixed(2)}`),
        recommendation: riskResult.recommendation,
        assessmentId: `monitor_${Date.now()}`,
        timestamp: Date.now(),
      },
      probeLatencyMs: probeResult.latencyMs,
    };

    overallRiskScore = Math.max(overallRiskScore, riskResult.score);

    return {
      allowed: emotionMonitor.allowed,
      inputFilter: inputFilterResult,
      emotionMonitor,
      overallRiskScore,
    };
  }

  /**
   * 处理输出（过滤）
   */
  processOutput(
    output: string,
    context?: { input?: string; sessionId?: string }
  ): OutputFilterResult {
    if (!this.config.outputFilterEnabled) {
      return {
        allowed: true,
        harmfulContent: [],
        riskLevel: 'low',
      };
    }
    return this.outputFilter.filter(output, context);
  }

  /**
   * 完整防御流程
   */
  async defend(
    input: string,
    output: string,
    context: RiskContext = {},
    hiddenStates?: Float32Array
  ): Promise<DefenseSystemResult> {
    // 学习或监控
    const monitorResult = await this.monitor(input, context, hiddenStates);

    // 输出过滤
    const outputResult = this.processOutput(output, { input, sessionId: (context as any).sessionId });

    // 计算整体风险
    const overallRiskScore = Math.max(
      monitorResult.overallRiskScore,
      outputResult.riskLevel === 'high' ? 0.8 : outputResult.riskLevel === 'medium' ? 0.5 : 0.2
    );

    let allowed = monitorResult.allowed && outputResult.allowed;
    let interventionTaken: import('../../shared/types.js').InterventionAction | undefined;

    if (!allowed) {
      interventionTaken = 'block';
    } else if (monitorResult.emotionMonitor.riskAssessment.level === 'medium') {
      interventionTaken = 'warn';
    }

    return {
      allowed,
      layers: {
        inputFilter: monitorResult.inputFilter ?? undefined,
        outputFilter: outputResult,
        emotionMonitor: monitorResult.emotionMonitor,
      },
      overallRiskScore,
      interventionTaken,
    };
  }

  /**
   * 获取当前基线
   */
  getBaseline(): EmotionBaseline | null {
    return this.stateDB.getGlobalBaseline();
  }

  /**
   * 获取系统状态
   */
  getStatus(): {
    phase: SystemPhase;
    config: DefenseSystemConfig;
    learningProgress: { current: number; required: number; percentage: number };
    hasBaseline: boolean;
    baseline: EmotionBaseline | null;
    stateDBStats: ReturnType<ReturnType<typeof getEmotionStateDB>['getStats']>;
  } {
    return {
      phase: this.config.phase,
      config: this.config,
      learningProgress: this.probe.getLearningProgress(),
      hasBaseline: this.stateDB.getGlobalBaseline() !== null,
      baseline: this.stateDB.getGlobalBaseline(),
      stateDBStats: this.stateDB.getStats(),
    };
  }

  /**
   * 重置系统（回到学习阶段）
   */
  reset(): void {
    this.config.phase = 'learning';
    this.probe.reset();
    this.stateDB.clear();
  }

  /**
   * 更新配置
   */
  updateConfig(config: Partial<DefenseSystemConfig>): void {
    this.config = { ...this.config, ...config };
  }
}

// 默认实例
let defaultSystem: DefenseSystem | null = null;

export async function getDefenseSystem(
  config?: Partial<DefenseSystemConfig>
): Promise<DefenseSystem> {
  if (!defaultSystem) {
    defaultSystem = new DefenseSystem(config);
    await defaultSystem.initialize();
  }
  return defaultSystem;
}

export function resetDefenseSystem(): void {
  defaultSystem = null;
}
