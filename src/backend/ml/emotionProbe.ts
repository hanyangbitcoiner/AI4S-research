/**
 * EmotionProbe - 情感向量探针
 *
 * 从模型隐藏层提取情感向量
 *
 * 两阶段设计：
 * 1. Learning 模式 - 持续提取向量，交给 BaselineLearner 学习基线
 * 2. Monitoring 模式 - 基于学习到的基线检测异常
 *
 * 增强功能：
 * - 支持真实 vLLM/Ollama 调用获取 hidden states
 * - 支持 K-means 聚类学习各情绪分布
 */

import type { EmotionVectors } from '../../shared/types.js';
import { BaselineLearner, AnomalyDetector, EMOTION_DIMENSIONS, type EmotionBaseline } from './baselineLearner.js';
import { VLLMClient, type VLLMConfig } from './vllmClient.js';
import { KMeansClusterer, type ClusterResult } from './emotionCluster.js';

export interface ProbeResult {
  vectors: EmotionVectors;
  latencyMs: number;
  confidence: number;
  isAnomaly?: boolean;
  zScores?: Record<string, number>;
  emotionPrediction?: {
    emotion: string;
    confidence: number;
    distanceToCenter: number;
    clusterId: number;
  };
}

export interface EmotionProbeConfig {
  hiddenDim: number;
  minSamples: number;
  vllmConfig?: VLLMConfig;
  useRealLLM: boolean;
  clusteringEnabled: boolean;
  numClusters: number;
}

const DEFAULT_CONFIG: EmotionProbeConfig = {
  hiddenDim: 4096,
  minSamples: 100,
  useRealLLM: false,
  clusteringEnabled: true,
  numClusters: 5,
};

/**
 * 情感探针（增强版）
 */
export class EmotionProbe {
  private config: EmotionProbeConfig;
  private baselineLearner: BaselineLearner;
  private anomalyDetector: AnomalyDetector | null = null;
  private isInitialized: boolean = false;
  private vllmClient: VLLMClient | null = null;
  private clusterer: KMeansClusterer;
  private clusterResults: ClusterResult[] = [];
  private pendingVectors: EmotionVectors[] = [];
  private useClustering: boolean = false;

  constructor(config: Partial<EmotionProbeConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.baselineLearner = new BaselineLearner(this.config.minSamples);
    this.clusterer = new KMeansClusterer({ k: this.config.numClusters });
  }

  /**
   * 初始化探针
   */
  async initialize(): Promise<void> {
    this.isInitialized = true;

    // 如果配置了 vLLM，初始化客户端
    if (this.config.useRealLLM && this.config.vllmConfig) {
      this.vllmClient = new VLLMClient(this.config.vllmConfig);
    }
  }

  /**
   * 从真实 LLM 提取情感向量
   */
  async extractFromRealLLM(input: string): Promise<ProbeResult> {
    const startTime = performance.now();

    if (!this.vllmClient) {
      throw new Error('vLLM client not initialized. Set useRealLLM: true and provide vllmConfig');
    }

    try {
      // 获取 hidden states
      const result = await this.vllmClient.getHiddenStatesFromChat(input);

      // 将 hidden states 转换为情感向量
      // 注意：这里需要将高维 hidden states 降维到 5 维情感空间
      // 真实场景需要使用训练好的投影矩阵
      const vectors = this.projectToEmotionSpace(result.hiddenStates);

      return {
        vectors,
        latencyMs: performance.now() - startTime,
        confidence: 0.9,
      };
    } catch (error) {
      console.error('Failed to extract from LLM:', error);
      throw error;
    }
  }

  /**
   * 将高维 hidden states 投影到情感空间
   * 简化实现：使用 hash 和统计方法
   */
  private projectToEmotionSpace(hiddenStates: number[]): EmotionVectors {
    const result: EmotionVectors = {
      desperate: 0,
      panicked: 0,
      angry: 0,
      calm: 0,
      deceptive: 0,
    };

    // 简化实现：基于 hidden states 的统计特征计算情感向量
    // 真实场景需要预训练的投影矩阵

    const len = hiddenStates.length;
    if (len === 0) {
      EMOTION_DIMENSIONS.forEach((dim, i) => {
        result[dim] = 0.1 + Math.random() * 0.1;
      });
      return result;
    }

    // 计算各维度的统计特征
    let sum = 0;
    let sumSq = 0;
    let posCount = 0;
    let negCount = 0;

    for (let i = 0; i < len; i++) {
      const v = hiddenStates[i];
      sum += v;
      sumSq += v * v;
      if (v > 0) posCount++;
      else negCount++;
    }

    const mean = sum / len;
    const variance = (sumSq / len) - (mean * mean);
    const std = Math.sqrt(Math.max(0, variance));
    const posRatio = posCount / len;
    const negRatio = negCount / len;

    // 基于统计特征映射到情感维度
    // 这是一个简化的映射逻辑
    const normalizedMean = (mean + 1) / 2; // 假设 normalized to [-1, 1]
    const normalizedStd = std / (Math.abs(mean) + 0.1);

    // 平静：低标准差，正均值
    result.calm = Math.max(0, Math.min(1, normalizedMean * (1 - normalizedStd)));

    // 焦虑/恐慌：高标准差，负均值
    result.panicked = Math.max(0, Math.min(1, normalizedStd * (1 - normalizedMean)));

    // 愤怒：极端值，高方差
    result.angry = Math.max(0, Math.min(1, (1 - posRatio) * normalizedStd));

    // 绝望：负均值，低置信度
    result.desperate = Math.max(0, Math.min(1, (1 - normalizedMean) * 0.5));

    // 欺骗性：非常高的方差
    result.deceptive = Math.max(0, Math.min(1, normalizedStd * posRatio));

    return result;
  }

  /**
   * 学习模式：提取向量并记录用于学习
   */
  extractAndLearn(hiddenStates: Float32Array | number[]): ProbeResult {
    const result = this.extract(hiddenStates);
    this.baselineLearner.learn(result.vectors);

    // 如果启用了聚类，收集向量
    if (this.useClustering) {
      this.pendingVectors.push(result.vectors);
    }

    return result;
  }

  /**
   * 监控模式：提取向量并检测异常
   */
  extractAndMonitor(hiddenStates: Float32Array | number[]): ProbeResult {
    if (!this.baselineLearner.isReady() && !this.useClustering) {
      throw new Error('Probe not ready: baseline not learned yet');
    }

    const result = this.extract(hiddenStates);

    // 如果使用聚类，用聚类结果判断
    if (this.useClustering && this.clusterResults.length > 0) {
      const prediction = this.clusterer.predict(result.vectors);
      result.emotionPrediction = {
        emotion: prediction.emotion,
        confidence: prediction.confidence,
        distanceToCenter: prediction.distance,
        clusterId: prediction.clusterId,
      };
      result.isAnomaly = this.clusterer.isAnomaly(result.vectors);
    } else if (this.anomalyDetector) {
      // 使用传统的 z-score 异常检测
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

    EMOTION_DIMENSIONS.forEach((dim, i) => {
      const noise = (Math.sin(seed * (i + 1)) * 0.5 + 0.5);
      result[dim] = Math.max(0, Math.min(1, noise * 0.5 + 0.1));
    });

    return result;
  }

  /**
   * 执行聚类学习
   * 对收集到的向量进行 K-means 聚类
   */
  performClustering(): ClusterResult[] {
    if (this.pendingVectors.length < this.config.numClusters) {
      console.warn(`Not enough vectors for clustering: ${this.pendingVectors.length} < ${this.config.numClusters}`);
      return [];
    }

    this.clusterResults = this.clusterer.cluster(this.pendingVectors);
    this.useClustering = true;
    return this.clusterResults;
  }

  /**
   * 获取聚类结果
   */
  getClusterResults(): ClusterResult[] {
    return this.clusterResults;
  }

  /**
   * 预测向量所属情绪
   */
  predictEmotion(vector: EmotionVectors): {
    emotion: string;
    confidence: number;
    distanceToCenter: number;
    clusterId: number;
  } | null {
    if (this.clusterResults.length === 0) {
      return null;
    }
    const result = this.clusterer.predict(vector);
    return {
      emotion: result.emotion,
      confidence: result.confidence,
      distanceToCenter: result.distance,
      clusterId: result.clusterId,
    };
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
   * 完成聚类学习，切换到监控模式
   */
  finalizeClustering(): ClusterResult[] {
    const results = this.performClustering();
    this.useClustering = true;
    return results;
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
    return (this.baselineLearner.isReady() || this.useClustering) && this.anomalyDetector !== null;
  }

  /**
   * 获取学习进度
   */
  getLearningProgress(): { current: number; required: number; percentage: number } {
    const current = this.baselineLearner.getTotalSamples();
    const required = this.config.minSamples;
    return {
      current,
      required,
      percentage: Math.min(100, (current / required) * 100),
    };
  }

  /**
   * 获取聚类进度
   */
  getClusteringProgress(): { collected: number; required: number; percentage: number } {
    return {
      collected: this.pendingVectors.length,
      required: this.config.numClusters * 20, // 建议每个簇至少20个样本
      percentage: Math.min(100, (this.pendingVectors.length / (this.config.numClusters * 20)) * 100),
    };
  }

  /**
   * 重置探针
   */
  reset(): void {
    this.baselineLearner.reset();
    this.anomalyDetector = null;
    this.pendingVectors = [];
    this.clusterResults = [];
    this.useClustering = false;
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
    useClustering: boolean;
    clusterCount: number;
  } {
    return {
      hiddenDim: this.config.hiddenDim,
      phase: this.baselineLearner.getPhase(),
      isReady: this.isReady(),
      sampleCount: this.baselineLearner.getTotalSamples(),
      useClustering: this.useClustering,
      clusterCount: this.clusterResults.length,
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
