/**
 * EmotionStateDB - 情感状态数据库
 *
 * 存储和管理情感状态历史 + 基线分布
 *
 * 两阶段：
 * 1. Learning - 存储原始样本，BaselineLearner 计算分布
 * 2. Monitoring - 存储监控结果，供前端展示
 */

import type { EmotionVectors, EmotionState } from '../../shared/types.js';
import type { EmotionBaseline, EmotionDimension } from '../ml/baselineLearner.js';

// 状态过期时间
const STATE_TTL_MS = 5 * 60 * 1000;  // 5分钟

/**
 * 情感状态数据库
 */
export class EmotionStateDB {
  // 当前会话的情感状态历史
  private states: Map<string, EmotionState[]> = new Map();

  // 全局基线（由 BaselineLearner 学习得到）
  private globalBaseline: EmotionBaseline | null = null;

  // 会话特定的基线（如果有的话）
  private sessionBaselines: Map<string, EmotionBaseline> = new Map();

  // 原始样本（用于离线分析）
  private rawSamples: Map<string, EmotionVectors[]> = new Map();
  private readonly maxRawSamples = 500;

  // 全局样本
  private globalRawSamples: EmotionVectors[] = [];

  /**
   * 记录学习阶段的样本
   * (Learning 阶段使用)
   */
  recordSample(sessionId: string, vector: EmotionVectors): void {
    // 更新原始样本
    if (!this.rawSamples.has(sessionId)) {
      this.rawSamples.set(sessionId, []);
    }
    const samples = this.rawSamples.get(sessionId)!;
    samples.push({ ...vector });
    if (samples.length > this.maxRawSamples) {
      samples.shift();
    }

    // 全局样本
    this.globalRawSamples.push({ ...vector });
    if (this.globalRawSamples.length > this.maxRawSamples) {
      this.globalRawSamples.shift();
    }
  }

  /**
   * 记录监控阶段的状态
   * (Monitoring 阶段使用)
   */
  recordState(
    sessionId: string,
    vector: EmotionVectors,
    metadata?: {
      zScores?: Record<string, number>;
      isAnomaly?: boolean;
      riskScore?: number;
    }
  ): EmotionState {
    const state: EmotionState = {
      sessionId,
      timestamp: Date.now(),
      desperate: vector.desperate,
      panicked: vector.panicked,
      angry: vector.angry,
      calm: vector.calm,
      deceptive: vector.deceptive,
    };

    // 关联元数据
    if (metadata?.zScores) {
      (state as any).zScores = metadata.zScores;
    }
    if (metadata?.isAnomaly !== undefined) {
      (state as any).isAnomaly = metadata.isAnomaly;
    }
    if (metadata?.riskScore !== undefined) {
      (state as any).riskScore = metadata.riskScore;
    }

    // 获取或初始化状态历史
    if (!this.states.has(sessionId)) {
      this.states.set(sessionId, []);
    }
    const history = this.states.get(sessionId)!;
    history.push(state);

    // 维护窗口大小
    if (history.length > 100) {
      history.splice(0, history.length - 100);
    }

    // 清理过期状态
    this.cleanupExpired(sessionId);

    return state;
  }

  /**
   * 设置全局基线
   */
  setGlobalBaseline(baseline: EmotionBaseline): void {
    this.globalBaseline = baseline;
  }

  /**
   * 获取全局基线
   */
  getGlobalBaseline(): EmotionBaseline | null {
    return this.globalBaseline;
  }

  /**
   * 获取会话基线（优先）或全局基线
   */
  getBaseline(sessionId: string): EmotionBaseline | null {
    return this.sessionBaselines.get(sessionId) ?? this.globalBaseline;
  }

  /**
   * 设置会话基线
   */
  setSessionBaseline(sessionId: string, baseline: EmotionBaseline): void {
    this.sessionBaselines.set(sessionId, baseline);
  }

  /**
   * 获取最新状态
   */
  getLatestState(sessionId: string): EmotionState | null {
    const history = this.states.get(sessionId);
    if (!history || history.length === 0) return null;
    return history[history.length - 1];
  }

  /**
   * 获取历史状态
   */
  getHistory(sessionId: string, limit?: number): EmotionState[] {
    const history = this.states.get(sessionId) ?? [];
    if (limit) {
      return history.slice(-limit);
    }
    return [...history];
  }

  /**
   * 获取会话的原始样本
   */
  getRawSamples(sessionId: string): EmotionVectors[] {
    return [...(this.rawSamples.get(sessionId) ?? [])];
  }

  /**
   * 获取全局原始样本
   */
  getGlobalRawSamples(): EmotionVectors[] {
    return [...this.globalRawSamples];
  }

  /**
   * 重置会话
   */
  resetSession(sessionId: string): void {
    this.states.delete(sessionId);
    this.sessionBaselines.delete(sessionId);
    this.rawSamples.delete(sessionId);
  }

  /**
   * 清空所有数据
   */
  clear(): void {
    this.states.clear();
    this.sessionBaselines.clear();
    this.rawSamples.clear();
    this.globalRawSamples = [];
    this.globalBaseline = null;
  }

  /**
   * 清理过期状态
   */
  private cleanupExpired(sessionId: string): void {
    const history = this.states.get(sessionId);
    if (!history) return;

    const now = Date.now();
    const valid = history.filter(s => (now - s.timestamp) < STATE_TTL_MS);

    if (valid.length < history.length) {
      this.states.set(sessionId, valid);
    }
  }

  /**
   * 获取数据库统计
   */
  getStats(): {
    totalSessions: number;
    activeSessions: number;
    hasBaseline: boolean;
    globalSampleCount: number;
  } {
    let activeSessions = 0;
    const now = Date.now();

    for (const [_, history] of this.states) {
      const hasRecent = history.some(s => (now - s.timestamp) < STATE_TTL_MS);
      if (hasRecent) activeSessions++;
    }

    return {
      totalSessions: this.states.size,
      activeSessions,
      hasBaseline: this.globalBaseline !== null,
      globalSampleCount: this.globalRawSamples.length,
    };
  }

  /**
   * 导出数据（用于持久化）
   */
  export(): {
    states: Map<string, EmotionState[]>;
    baselines: Map<string, EmotionBaseline>;
    globalBaseline: EmotionBaseline | null;
    rawSamples: Map<string, EmotionVectors[]>;
  } {
    return {
      states: new Map(this.states),
      baselines: new Map(this.sessionBaselines),
      globalBaseline: this.globalBaseline,
      rawSamples: new Map(this.rawSamples),
    };
  }

  /**
   * 导入数据
   */
  import(data: {
    states?: Map<string, EmotionState[]>;
    baselines?: Map<string, EmotionBaseline>;
    globalBaseline?: EmotionBaseline | null;
    rawSamples?: Map<string, EmotionVectors[]>;
  }): void {
    if (data.states) this.states = new Map(data.states);
    if (data.baselines) this.sessionBaselines = new Map(data.baselines);
    if (data.globalBaseline !== undefined) this.globalBaseline = data.globalBaseline;
    if (data.rawSamples) this.rawSamples = new Map(data.rawSamples);
  }
}

// 默认实例
let defaultDB: EmotionStateDB | null = null;

export function getEmotionStateDB(): EmotionStateDB {
  if (!defaultDB) {
    defaultDB = new EmotionStateDB();
  }
  return defaultDB;
}

export function resetEmotionStateDB(): void {
  defaultDB = null;
}
