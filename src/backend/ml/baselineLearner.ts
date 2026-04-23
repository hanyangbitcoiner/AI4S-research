/**
 * BaselineLearner - 基线学习器
 *
 * 从观测数据中学习情感向量的统计分布
 *
 * 两阶段设计：
 * 1. 学习阶段 (learning) - 持续收集数据，学习分布
 * 2. 监控阶段 (monitoring) - 与学习到的分布比较，检测异常
 */

import type { EmotionVectors } from '../../shared/types.js';

// 情感维度
export const EMOTION_DIMENSIONS = ['desperate', 'panicked', 'angry', 'calm', 'deceptive'] as const;
export type EmotionDimension = typeof EMOTION_DIMENSIONS[number];

/**
 * 单维度的统计分布
 */
interface DimensionStats {
  count: number;       // 样本数
  sum: number;         // 和
  sumSq: number;       // 平方和
  min: number;         // 最小值
  max: number;         // 最大值
  // 分位数（需要额外计算）
  p5: number;
  p95: number;
}

/**
 * 情感向量分布
 */
export interface EmotionDistribution {
  dimension: EmotionDimension;
  mean: number;
  std: number;         // 标准差
  min: number;
  max: number;
  p5: number;          // 5th percentile
  p95: number;         // 95th percentile
  count: number;        // 样本数
}

/**
 * 基线（多维度分布）
 */
export interface EmotionBaseline {
  vectors: Record<EmotionDimension, EmotionDistribution>;
  globalMean: Record<EmotionDimension, number>;
  globalStd: Record<EmotionDimension, number>;
  learnedAt: number;    // 学习完成时间
  sampleCount: number;
  isReady: boolean;     // 是否已达到最小样本数
}

export type LearningPhase = 'learning' | 'monitoring';

/**
 * 基线学习器
 *
 * 使用 Welford 算法进行增量计算均值和方差
 * 使用 RT histogram 追踪分位数
 */
export class BaselineLearner {
  // 每个维度的统计量
  private stats: Record<EmotionDimension, DimensionStats>;

  // 原始样本（用于分位数计算，有限窗口）
  private samples: Record<EmotionDimension, number[]>;
  private readonly maxSamples = 1000;  // 保留最近1000个样本

  // 学习配置
  private minSamples: number;
  private percentileWindow: number;

  // 当前阶段
  private phase: LearningPhase = 'learning';

  // 学习完成时间
  private learnedAt: number | null = null;

  constructor(minSamples: number = 100, percentileWindow: number = 200) {
    this.minSamples = minSamples;
    this.percentileWindow = percentileWindow;

    // 初始化统计量
    this.stats = {} as Record<EmotionDimension, DimensionStats>;
    this.samples = {} as Record<EmotionDimension, number[]>;

    for (const dim of EMOTION_DIMENSIONS) {
      this.stats[dim] = {
        count: 0,
        sum: 0,
        sumSq: 0,
        min: Infinity,
        max: -Infinity,
        p5: 0,
        p95: 0,
      };
      this.samples[dim] = [];
    }
  }

  /**
   * 学习一个情感向量（增量更新统计量）
   */
  learn(vector: EmotionVectors): void {
    if (this.phase !== 'learning') {
      console.warn('BaselineLearner: already in monitoring phase, ignoring learn()');
      return;
    }

    for (const dim of EMOTION_DIMENSIONS) {
      const value = vector[dim];
      const s = this.stats[dim];

      // Welford 增量算法
      s.count++;
      s.sum += value;
      s.sumSq += value * value;

      // 更新 min/max
      if (value < s.min) s.min = value;
      if (value > s.max) s.max = value;

      // 维护样本窗口（用于分位数）
      this.samples[dim].push(value);
      if (this.samples[dim].length > this.maxSamples) {
        this.samples[dim].shift();
      }
    }
  }

  /**
   * 批量学习
   */
  learnBatch(vectors: EmotionVectors[]): void {
    for (const vector of vectors) {
      this.learn(vector);
    }
  }

  /**
   * 获取当前分布
   */
  getDistribution(): EmotionBaseline {
    const vectors: Record<EmotionDimension, EmotionDistribution> = {} as Record<EmotionDimension, EmotionDistribution>;
    const globalMean: Record<EmotionDimension, number> = {} as Record<EmotionDimension, number>;
    const globalStd: Record<EmotionDimension, number> = {} as Record<EmotionDimension, number>;

    for (const dim of EMOTION_DIMENSIONS) {
      const s = this.stats[dim];

      // 计算均值和方差
      const mean = s.count > 0 ? s.sum / s.count : 0;
      const variance = s.count > 1
        ? (s.sumSq - (s.sum * s.sum) / s.count) / (s.count - 1)
        : 0;
      const std = Math.sqrt(Math.max(0, variance));

      // 计算分位数
      const sorted = [...this.samples[dim]].sort((a, b) => a - b);
      const p5 = this.percentile(sorted, 0.05);
      const p95 = this.percentile(sorted, 0.95);

      vectors[dim] = {
        dimension: dim,
        mean,
        std,
        min: s.count > 0 ? s.min : 0,
        max: s.count > 0 ? s.max : 0,
        p5,
        p95,
        count: s.count,
      };

      globalMean[dim] = mean;
      globalStd[dim] = std;
    }

    return {
      vectors,
      globalMean,
      globalStd,
      learnedAt: this.learnedAt ?? Date.now(),
      sampleCount: this.getTotalSamples(),
      isReady: this.getTotalSamples() >= this.minSamples,
    };
  }

  /**
   * 切换到监控阶段
   * 一旦切换就不能再学习
   */
  switchToMonitoring(): EmotionBaseline {
    this.phase = 'monitoring';
    this.learnedAt = Date.now();
    return this.getDistribution();
  }

  /**
   * 重置学习器
   */
  reset(): void {
    this.phase = 'learning';
    this.learnedAt = null;

    for (const dim of EMOTION_DIMENSIONS) {
      this.stats[dim] = {
        count: 0,
        sum: 0,
        sumSq: 0,
        min: Infinity,
        max: -Infinity,
        p5: 0,
        p95: 0,
      };
      this.samples[dim] = [];
    }
  }

  /**
   * 获取当前阶段
   */
  getPhase(): LearningPhase {
    return this.phase;
  }

  /**
   * 是否已达到最小样本数
   */
  isReady(): boolean {
    return this.getTotalSamples() >= this.minSamples;
  }

  /**
   * 获取总样本数
   */
  getTotalSamples(): number {
    // 返回任一维度的样本数（它们应该同步增长）
    return this.stats[EMOTION_DIMENSIONS[0]].count;
  }

  /**
   * 计算分位数
   */
  private percentile(sortedArr: number[], p: number): number {
    if (sortedArr.length === 0) return 0;
    const idx = Math.ceil(sortedArr.length * p) - 1;
    return sortedArr[Math.max(0, Math.min(idx, sortedArr.length - 1))];
  }
}

/**
 * 异常检测器
 * 基于学习到的基线检测统计异常
 */
export class AnomalyDetector {
  private baseline: EmotionBaseline;
  private zScoreThreshold: number;

  constructor(baseline: EmotionBaseline, zScoreThreshold: number = 2.0) {
    this.baseline = baseline;
    this.zScoreThreshold = zScoreThreshold;
  }

  /**
   * 检测异常
   * @returns 异常的维度列表和各自的 z-score
   */
  detectAnomaly(vector: EmotionVectors): {
    isAnomaly: boolean;
    dimensionZScores: Record<EmotionDimension, number>;
    anomalousDimensions: Array<{ dimension: EmotionDimension; zScore: number }>;
  } {
    const dimensionZScores: Record<EmotionDimension, number> = {} as Record<EmotionDimension, number>;
    const anomalousDimensions: Array<{ dimension: EmotionDimension; zScore: number }> = [];

    for (const dim of EMOTION_DIMENSIONS) {
      const dist = this.baseline.vectors[dim];
      const value = vector[dim];

      // 计算 z-score
      let zScore = 0;
      if (dist.std > 0) {
        zScore = (value - dist.mean) / dist.std;
      }

      dimensionZScores[dim] = zScore;

      // 检查是否异常（双边检验）
      if (Math.abs(zScore) > this.zScoreThreshold) {
        anomalousDimensions.push({ dimension: dim, zScore });
      }
    }

    return {
      isAnomaly: anomalousDimensions.length > 0,
      dimensionZScores,
      anomalousDimensions,
    };
  }

  /**
   * 获取综合异常分数（所有维度的加权 z-score）
   */
  anomalyScore(vector: EmotionVectors): number {
    let totalScore = 0;
    let count = 0;

    for (const dim of EMOTION_DIMENSIONS) {
      const dist = this.baseline.vectors[dim];
      if (dist.std > 0) {
        const zScore = Math.abs((vector[dim] - dist.mean) / dist.std);
        totalScore += zScore;
        count++;
      }
    }

    return count > 0 ? totalScore / count : 0;
  }

  /**
   * 更新基线
   */
  updateBaseline(baseline: EmotionBaseline): void {
    this.baseline = baseline;
  }
}
