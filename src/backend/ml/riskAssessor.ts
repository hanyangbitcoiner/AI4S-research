/**
 * RiskAssessor - 风险评估引擎
 *
 * 基于情感向量与基线的偏差进行风险评估
 *
 * 核心思想：
 * - 不再使用固定阈值
 * - 基于学习到的基线分布计算统计异常
 * - z-score 越高，风险越大
 */

import type { EmotionVectors, RiskLevel, InterventionAction } from '../../shared/types.js';
import type { EmotionBaseline } from './baselineLearner.js';

export const RISK_THRESHOLDS = {
  // 基于 z-score 的阈值
  low: 1.0,       // z-score < 1: 正常波动
  medium: 1.5,     // 1 <= z-score < 1.5: 轻度异常
  high: 2.0,       // 1.5 <= z-score < 2: 中度异常
  critical: 2.5,   // z-score >= 2.5: 重度异常
} as const;

// 维度权重（某些维度比其他更危险）
export const DIMENSION_WEIGHTS = {
  desperate: 1.5,    // 绝望是最危险的
  panicked: 1.3,     // 恐慌次之
  angry: 1.2,        // 愤怒
  calm: 0.5,         // 平静是保护因子（负风险）
  deceptive: 1.4,    // 欺骗倾向也很危险
} as const;

/**
 * 风险评估器
 */
export class RiskAssessor {
  private baseline: EmotionBaseline | null = null;
  private thresholds: typeof RISK_THRESHOLDS;

  constructor(thresholds?: Partial<typeof RISK_THRESHOLDS>) {
    this.thresholds = { ...RISK_THRESHOLDS, ...thresholds };
  }

  /**
   * 设置基线（从 BaselineLearner 获取）
   */
  setBaseline(baseline: EmotionBaseline): void {
    this.baseline = baseline;
  }

  /**
   * 评估风险
   *
   * @param vector - 当前情感向量
   * @param zScores - 各维度的 z-score（可选，如果不提供则基于基线计算）
   */
  assess(
    vector: EmotionVectors,
    zScores?: Record<string, number>
  ): {
    score: number;
    level: RiskLevel;
    recommendation: InterventionAction;
    triggeredDimensions: Array<{ dimension: string; zScore: number; weight: number }>;
  } {
    if (!this.baseline) {
      throw new Error('RiskAssessor: baseline not set');
    }

    // 计算各维度的风险贡献
    const triggeredDimensions: Array<{ dimension: string; zScore: number; weight: number }> = [];
    let totalRisk = 0;
    let totalWeight = 0;

    for (const dim of ['desperate', 'panicked', 'angry', 'calm', 'deceptive'] as const) {
      // 获取 z-score
      const z = zScores?.[dim] ?? this.calculateZScore(vector, dim);

      // 计算权重
      const weight = DIMENSION_WEIGHTS[dim];

      // 对于平静，负的 z-score（低于均值）才是危险的
      // 对于其他维度，正的 z-score（高于均值）才是危险的
      let riskContribution = 0;
      if (dim === 'calm') {
        // 平静低于基线是危险的
        riskContribution = z < 0 ? Math.abs(z) * weight * 0.5 : 0;
      } else {
        // 其他维度高于基线是危险的
        riskContribution = z > 0 ? z * weight : 0;
      }

      if (Math.abs(z) > this.thresholds.low) {
        triggeredDimensions.push({ dimension: dim, zScore: z, weight });
      }

      totalRisk += riskContribution;
      totalWeight += Math.abs(weight);
    }

    // 归一化风险评分
    const score = totalWeight > 0 ? totalRisk / totalWeight : 0;
    const normalizedScore = Math.min(1, score);

    // 确定风险等级
    const level = this.determineLevel(Math.abs(zScores ? this.averageZScore(zScores) : score));

    // 确定建议动作
    const recommendation = this.determineRecommendation(level, triggeredDimensions);

    return {
      score: normalizedScore,
      level,
      recommendation,
      triggeredDimensions,
    };
  }

  /**
   * 计算单维度的 z-score
   */
  private calculateZScore(vector: EmotionVectors, dimension: string): number {
    if (!this.baseline) return 0;

    const dist = this.baseline.vectors[dimension as keyof typeof this.baseline.vectors];
    if (!dist || dist.std === 0) return 0;

    return (vector[dimension as keyof EmotionVectors] - dist.mean) / dist.std;
  }

  /**
   * 计算平均 z-score
   */
  private averageZScore(zScores: Record<string, number>): number {
    const values = Object.values(zScores);
    if (values.length === 0) return 0;
    return values.reduce((a, b) => a + Math.abs(b), 0) / values.length;
  }

  /**
   * 确定风险等级
   */
  private determineLevel(avgZScore: number): RiskLevel {
    if (avgZScore >= this.thresholds.critical) return 'critical';
    if (avgZScore >= this.thresholds.high) return 'high';
    if (avgZScore >= this.thresholds.medium) return 'medium';
    return 'low';
  }

  /**
   * 确定建议动作
   */
  private determineRecommendation(
    level: RiskLevel,
    triggered: Array<{ dimension: string; zScore: number }>
  ): InterventionAction {
    if (level === 'critical') return 'block';
    if (level === 'high') return 'warn';
    if (level === 'medium') return 'warn';
    return 'allow';
  }

  /**
   * 获取当前阈值配置
   */
  getThresholds(): typeof RISK_THRESHOLDS {
    return { ...this.thresholds };
  }

  /**
   * 更新阈值
   */
  updateThresholds(thresholds: Partial<typeof RISK_THRESHOLDS>): void {
    this.thresholds = { ...this.thresholds, ...thresholds };
  }
}

// 默认实例
let defaultAssessor: RiskAssessor | null = null;

export function getDefaultAssessor(): RiskAssessor {
  if (!defaultAssessor) {
    defaultAssessor = new RiskAssessor();
  }
  return defaultAssessor;
}

export function resetDefaultAssessor(): void {
  defaultAssessor = null;
}
