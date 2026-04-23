/**
 * EmotionProbe - 情感向量探针
 *
 * 从模型隐藏层提取情感向量
 *
 * 两阶段设计：
 * 1. Learning 模式 - 持续提取向量，交给 BaselineLearner 学习基线
 * 2. Monitoring 模式 - 基于学习到的基线检测异常
 */

import type { EmotionVectors } from '../../shared/types.js';
import { BaselineLearner, AnomalyDetector, EMOTION_DIMENSIONS, type EmotionBaseline } from './baselineLearner.js';

export interface ProbeResult {
  vectors: EmotionVectors;
  latencyMs: number;
  confidence: number;
  isAnomaly?: boolean;
  zScores?: Record<string, number>;
}

/**
 * 情感探针
 */
export class EmotionProbe {
  private hiddenDim: number;
  private baselineLearner: BaselineLearner;
  private anomalyDetector: AnomalyDetector | null = null;
  private isInitialized: boolean = false;

  constructor(hiddenDim: number = 4096, minSamples: number = 100) {
    this.hiddenDim = hiddenDim;
    this.baselineLearner = new BaselineLearner(minSamples);
  }

  /**
   * 初始化探针
   */
  async initialize(): Promise<void> {
    this.isInitialized = true;
  }

  /**
   * 学习模式：提取向量并记录用于学习
   * 不做异常判断，只提取
   */
  extractAndLearn(hiddenStates: Float32Array | number[]): ProbeResult {
    const result = this.extract(hiddenStates);
    this.baselineLearner.learn(result.vectors);
    return result;
  }

  /**
   * 监控模式：提取向量并检测异常
   * 必须在基线学习完成后才能使用
   */
  extractAndMonitor(hiddenStates: Float32Array | number[]): ProbeResult {
    if (!this.baselineLearner.isReady()) {
      throw new Error('Probe not ready: baseline not learned yet');
    }

    const result = this.extract(hiddenStates);

    // 使用异常检测器
    if (this.anomalyDetector) {
      const anomalyResult = this.anomalyDetector.detectAnomaly(result.vectors);
      result.isAnomaly = anomalyResult.isAnomaly;
      result.zScores = anomalyResult.dimensionZScores as Record<string, number>;
    }

    return result;
  }

  /**
   * 提取情感向量（核心提取逻辑）
   */
  extract(hiddenStates: Float32Array | number[]): ProbeResult {
    const startTime = performance.now();

    // 简化实现：基于隐藏状态的某些特征模拟情感向量
    // 真实场景：使用预训练的投影矩阵计算
    const arr = hiddenStates as number[];
    let seed = 0;
    for (let i = 0; i < arr.length; i++) {
      seed += arr[i] * (i + 1);
    }

    const vectors = this.simulateProbeOutput(seed);

    return {
      vectors,
      latencyMs: performance.now() - startTime,
      confidence: 0.85 + Math.random() * 0.1,
    };
  }

  /**
   * 模拟探针输出
   */
  private simulateProbeOutput(seed: number): EmotionVectors {
    const result: EmotionVectors = {
      desperate: 0,
      panicked: 0,
      angry: 0,
      calm: 0,
      deceptive: 0,
    };

    // 基于 seed 生成略微变化的向量
    EMOTION_DIMENSIONS.forEach((dim, i) => {
      const noise = (Math.sin(seed * (i + 1)) * 0.5 + 0.5);
      // 初始值设为较低水平，学习过程中会调整
      result[dim] = Math.max(0, Math.min(1, noise * 0.5 + 0.1));
    });

    return result;
  }

  /**
   * 完成学习，切换到监控模式
   */
  finalizeLearning(): EmotionBaseline {
    const baseline = this.baselineLearner.switchToMonitoring();
    this.anomalyDetector = new AnomalyDetector(baseline);
    return baseline;
  }

  /**
   * 获取当前基线
   */
  getBaseline(): EmotionBaseline {
    return this.baselineLearner.getDistribution();
  }

  /**
   * 检查是否在学习阶段
   */
  isLearning(): boolean {
    return this.baselineLearner.getPhase() === 'learning';
  }

  /**
   * 检查是否准备好监控
   */
  isReady(): boolean {
    return this.baselineLearner.isReady() && this.anomalyDetector !== null;
  }

  /**
   * 获取学习进度
   */
  getLearningProgress(): { current: number; required: number; percentage: number } {
    const current = this.baselineLearner.getTotalSamples();
    // minSamples 的默认值是 100
    const required = 100;
    return {
      current,
      required,
      percentage: Math.min(100, (current / required) * 100),
    };
  }

  /**
   * 重置探针
   */
  reset(): void {
    this.baselineLearner.reset();
    this.anomalyDetector = null;
  }

  /**
   * 关闭探针
   */
  shutdown(): void {
    this.isInitialized = false;
    this.reset();
  }

  /**
   * 获取探针信息
   */
  getInfo(): {
    hiddenDim: number;
    phase: string;
    isReady: boolean;
    sampleCount: number;
  } {
    return {
      hiddenDim: this.hiddenDim,
      phase: this.baselineLearner.getPhase(),
      isReady: this.isReady(),
      sampleCount: this.baselineLearner.getTotalSamples(),
    };
  }
}

// 默认实例
let defaultProbe: EmotionProbe | null = null;

export function getDefaultProbe(): EmotionProbe {
  if (!defaultProbe) {
    defaultProbe = new EmotionProbe();
  }
  return defaultProbe;
}

export function resetDefaultProbe(): void {
  defaultProbe = null;
}
