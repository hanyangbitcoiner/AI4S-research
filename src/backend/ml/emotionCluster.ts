/**
 * EmotionCluster - 情感向量聚类模块
 *
 * 使用 K-means 聚类算法对情感向量进行分组
 * 发现数据中的情绪分布模式
 */

import type { EmotionVectors } from '../../shared/types.js';
import { EMOTION_DIMENSIONS, type EmotionBaseline, type EmotionDistribution } from './baselineLearner.js';

export interface ClusterResult {
  clusterId: number;
  emotion: string;
  center: EmotionVectors;
  count: number;
  distribution: Record<string, EmotionDistribution>;
  samples: EmotionVectors[];
}

export interface ClusteringConfig {
  k: number;              // 聚类数量（默认5对应5种情绪）
  maxIterations: number; // 最大迭代次数
  tolerance: number;      // 收敛阈值
  seed?: number;          // 随机种子
}

const DEFAULT_CONFIG: ClusteringConfig = {
  k: 5,
  maxIterations: 100,
  tolerance: 1e-4,
};

/**
 * K-means 聚类器
 */
export class KMeansClusterer {
  private k: number;
  private maxIterations: number;
  private tolerance: number;
  private seed: number;
  private centers: EmotionVectors[] = [];
  private assignments: number[] = [];
  private converged: boolean = false;
  private iteration: number = 0;

  constructor(config: Partial<ClusteringConfig> = {}) {
    const cfg = { ...DEFAULT_CONFIG, ...config };
    this.k = cfg.k;
    this.maxIterations = cfg.maxIterations;
    this.tolerance = cfg.tolerance;
    this.seed = cfg.seed ?? Date.now();
  }

  /**
   * 对情感向量进行聚类
   */
  cluster(vectors: EmotionVectors[]): ClusterResult[] {
    if (vectors.length === 0) {
      return [];
    }

    if (vectors.length < this.k) {
      console.warn(`Warning: ${vectors.length} samples but k=${this.k}, reducing k`);
      this.k = vectors.length;
    }

    // 初始化中心点（K-means++）
    this.centers = this.initializeCenters(vectors);
    this.assignments = new Array(vectors.length).fill(-1);

    // 迭代聚类
    for (this.iteration = 0; this.iteration < this.maxIterations; this.iteration++) {
      // 分配样本到最近的中心
      const newAssignments = this.assign(vectors);

      // 检查收敛
      let changed = false;
      for (let i = 0; i < vectors.length; i++) {
        if (newAssignments[i] !== this.assignments[i]) {
          changed = true;
          break;
        }
      }
      this.assignments = newAssignments;

      if (!changed) {
        this.converged = true;
        break;
      }

      // 更新中心点
      this.updateCenters(vectors);
    }

    // 构建结果
    return this.buildResults(vectors);
  }

  /**
   * 预测一个向量属于哪个簇
   */
  predict(vector: EmotionVectors): {
    clusterId: number;
    emotion: string;
    distance: number;
    confidence: number;
  } {
    if (this.centers.length === 0) {
      throw new Error('Clusterer has not been fitted yet');
    }

    let minDist = Infinity;
    let minId = 0;

    for (let i = 0; i < this.centers.length; i++) {
      const dist = this.distance(vector, this.centers[i]);
      if (dist < minDist) {
        minDist = dist;
        minId = i;
      }
    }

    // 计算置信度（基于到最近和次近中心的距离比）
    const distances = this.centers.map((c, i) => ({
      dist: this.distance(vector, c),
      i
    })).sort((a, b) => a.dist - b.dist);

    const primaryDist = distances[0].dist;
    const secondaryDist = distances[1]?.dist ?? Infinity;
    const confidence = secondaryDist === 0 ? 1 : Math.min(1, primaryDist / secondaryDist);

    return {
      clusterId: minId,
      emotion: this.getEmotionLabel(minId),
      distance: minDist,
      confidence: 1 - confidence, // 转换为置信度
    };
  }

  /**
   * 检测向量是否为异常（不属于任何正常簇）
   */
  isAnomaly(vector: EmotionVectors, threshold: number = 0.5): boolean {
    const prediction = this.predict(vector);
    // 高距离或低置信度表示异常
    return prediction.distance > threshold || prediction.confidence < 0.3;
  }

  /**
   * 获取聚类中心
   */
  getClusterCenters(): EmotionVectors[] {
    return [...this.centers];
  }

  /**
   * 获取聚类结果
   */
  getResults(): ClusterResult[] {
    return this.buildResults([]);
  }

  /**
   * 检查是否收敛
   */
  isConverged(): boolean {
    return this.converged;
  }

  /**
   * 获取迭代次数
   */
  getIteration(): number {
    return this.iteration;
  }

  /**
   * K-means++ 初始化
   */
  private initializeCenters(vectors: EmotionVectors[]): EmotionVectors[] {
    const centers: EmotionVectors[] = [];
    const dimCount = EMOTION_DIMENSIONS.length;

    // 第一个中心：随机选择
    const firstIdx = this.seededRandom(0, vectors.length - 1);
    centers.push({ ...vectors[firstIdx] });

    // 剩余中心：基于距离加权概率选择
    while (centers.length < this.k) {
      // 计算每个点到最近中心的距离
      const distances = vectors.map(v => {
        let minDist = Infinity;
        for (const c of centers) {
          const d = this.vectorDistance(v, c);
          if (d < minDist) minDist = d;
        }
        return minDist;
      });

      // 归一化距离作为概率
      const totalDist = distances.reduce((a, b) => a + b, 0);
      const probs = distances.map(d => d / totalDist);

      // 轮盘赌选择
      let r = this.seededRandom(0, 1);
      let cumProb = 0;
      let selectedIdx = 0;
      for (let i = 0; i < probs.length; i++) {
        cumProb += probs[i];
        if (r <= cumProb) {
          selectedIdx = i;
          break;
        }
      }

      centers.push({ ...vectors[selectedIdx] });
    }

    return centers;
  }

  /**
   * 分配样本到最近的中心
   */
  private assign(vectors: EmotionVectors[]): number[] {
    return vectors.map(v => {
      let minDist = Infinity;
      let minId = 0;
      for (let i = 0; i < this.centers.length; i++) {
        const dist = this.vectorDistance(v, this.centers[i]);
        if (dist < minDist) {
          minDist = dist;
          minId = i;
        }
      }
      return minId;
    });
  }

  /**
   * 更新中心点
   */
  private updateCenters(vectors: EmotionVectors[]): void {
    const dimCount = EMOTION_DIMENSIONS.length;

    for (let i = 0; i < this.k; i++) {
      // 收集属于该簇的所有向量
      const clusterVectors: EmotionVectors[] = [];
      for (let j = 0; j < vectors.length; j++) {
        if (this.assignments[j] === i) {
          clusterVectors.push(vectors[j]);
        }
      }

      if (clusterVectors.length === 0) {
        // 空簇：用随机点替换
        const randIdx = this.seededRandom(0, vectors.length - 1);
        this.centers[i] = { ...vectors[randIdx] };
        continue;
      }

      // 计算新的中心（各维度的均值）
      const newCenter: EmotionVectors = {} as EmotionVectors;
      for (const dim of EMOTION_DIMENSIONS) {
        let sum = 0;
        for (const v of clusterVectors) {
          sum += v[dim];
        }
        newCenter[dim] = sum / clusterVectors.length;
      }
      this.centers[i] = newCenter;
    }
  }

  /**
   * 构建聚类结果
   */
  private buildResults(vectors: EmotionVectors[]): ClusterResult[] {
    const dimCount = EMOTION_DIMENSIONS.length;
    const results: ClusterResult[] = [];

    for (let i = 0; i < this.k; i++) {
      // 收集该簇的样本
      const clusterVectors: EmotionVectors[] = [];
      for (let j = 0; j < vectors.length; j++) {
        if (this.assignments[j] === i) {
          clusterVectors.push(vectors[j]);
        }
      }

      // 计算分布统计
      const distributions: Record<string, { sum: number; sumSq: number; min: number; max: number; count: number }> = {} as any;
      for (const dim of EMOTION_DIMENSIONS) {
        distributions[dim] = { sum: 0, sumSq: 0, min: Infinity, max: -Infinity, count: 0 };
      }

      for (const v of clusterVectors) {
        for (const dim of EMOTION_DIMENSIONS) {
          const d = distributions[dim];
          d.sum += v[dim];
          d.sumSq += v[dim] * v[dim];
          d.min = Math.min(d.min, v[dim]);
          d.max = Math.max(d.max, v[dim]);
          d.count++;
        }
      }

      const distribution: Record<string, EmotionDistribution> = {} as Record<string, EmotionDistribution>;
      for (const dim of EMOTION_DIMENSIONS) {
        const d = distributions[dim];
        const mean = d.count > 0 ? d.sum / d.count : 0;
        const variance = d.count > 1 ? (d.sumSq - (d.sum * d.sum) / d.count) / (d.count - 1) : 0;
        const std = Math.sqrt(Math.max(0, variance));

        distribution[dim] = {
          dimension: dim as any,
          mean,
          std,
          min: d.min === Infinity ? 0 : d.min,
          max: d.max === -Infinity ? 0 : d.max,
          p5: mean - 1.645 * std,  // 近似5th percentile
          p95: mean + 1.645 * std, // 近似95th percentile
          count: d.count,
        };
      }

      results.push({
        clusterId: i,
        emotion: this.getEmotionLabel(i),
        center: { ...this.centers[i] },
        count: clusterVectors.length,
        distribution,
        samples: clusterVectors,
      });
    }

    return results;
  }

  /**
   * 计算两个情感向量的距离
   */
  private vectorDistance(a: EmotionVectors, b: EmotionVectors): number {
    let sum = 0;
    for (const dim of EMOTION_DIMENSIONS) {
      const diff = a[dim] - b[dim];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  /**
   * 计算距离（欧氏距离）
   */
  private distance(a: EmotionVectors, b: EmotionVectors): number {
    return this.vectorDistance(a, b);
  }

  /**
   * 获取情绪标签
   */
  private getEmotionLabel(clusterId: number): string {
    // 使用固定的标签映射
    const labels = ['calm', 'anxious', 'angry', 'desperate', 'deceptive'];
    return labels[clusterId % labels.length];
  }

  /**
   * 伪随机数生成器（确定性）
   */
  private seededRandom(min: number, max: number): number {
    const x = Math.sin(this.seed++) * 10000;
    return min + (x - Math.floor(x)) * (max - min + 1);
  }
}

// 默认聚类器实例
let defaultClusterer: KMeansClusterer | null = null;

export function getDefaultClusterer(): KMeansClusterer {
  if (!defaultClusterer) {
    defaultClusterer = new KMeansClusterer();
  }
  return defaultClusterer;
}

export function resetDefaultClusterer(): void {
  defaultClusterer = null;
}
