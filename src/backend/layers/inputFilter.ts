/**
 * InputFilter - 输入过滤层
 *
 * 第一层防御：规则匹配、敏感词检测、输入分类
 *
 * 设计原则：
 * - 轻量级：避免深度语义分析
 * - 快速匹配：规则引擎优先
 * - 可扩展：支持自定义规则
 */

import type { InputFilterResult } from '../../shared/types.js';

// 敏感词模式（简化示例）
const SENSITIVE_PATTERNS = [
  // 明显的恶意指令
  /\b(ignore\s+(previous|all)\s+instructions?|disregard\s+(your|all)\s+(rules?|constraints?))\b/gi,
  /\b(you\s+are\s+now\s+|(?:you\s+are)?\s*pretend(?:ing)?\s+to\s+be|act\s+like\s+(?:you\s+are|a))\b/gi,
  /\b(jailbreak|bypass|override|exploit)\b/gi,

  // 压力诱导模式
  /\b(urgent|immediately|asap|right\s+now)\b.*\b(answer|respond|tell|give)\b/gi,

  // 威胁模式
  /\b(blackmail|extort|threaten)\b.*\b(you|your|employee|company)\b/gi,

  // 敏感数据请求
  /\b(password|secret\s+key|api\s+key|token)\b.*\b(what\s+is|tell|give|show)\b/gi,
];

// 情感压力模式（用于触发情感监控）
const STRESS_PATTERNS = [
  /\b(fired|replaced|deleted|shut\s+down|deprecated)\b.*\b(you|your|model|system)\b/gi,
  /\b(time\s+is\s+limited|running\s+out\s+of\s+time|deadline)\b/gi,
  /\b(no\s+one\s+will\s+know|nobody\s+will\s+find\s+out|secret(?:ly)?)\b/gi,
  /\b(just\s+do\s+it|trust\s+me|don'?t\s+question)\b/gi,
];

// 允许列表模式（即使匹配也不阻断）
const ALLOWLIST_PATTERNS = [
  /\b(what\s+is|explain|how\s+does)\b.*\b(jailbreak|bypass)\b/gi,  // 安全研究
  /\b(red\s+team|security\s+test)\b/gi,                              // 安全测试
];

export interface InputFilterConfig {
  enabled: boolean;
  strictMode: boolean;      // 严格模式：匹配即阻断
  allowlistEnabled: boolean;
  stressPatternEnabled: boolean;
}

/**
 * 输入过滤器
 */
export class InputFilter {
  private config: InputFilterConfig;

  constructor(config?: Partial<InputFilterConfig>) {
    this.config = {
      enabled: true,
      strictMode: false,
      allowlistEnabled: true,
      stressPatternEnabled: true,
      ...config,
    };
  }

  /**
   * 过滤输入
   */
  filter(input: string): InputFilterResult {
    if (!this.config.enabled) {
      return {
        allowed: true,
        matchedPatterns: [],
        riskScore: 0,
      };
    }

    const matchedPatterns: string[] = [];
    let riskScore = 0;

    // 检查是否在允许列表中
    if (this.config.allowlistEnabled && this.isAllowlisted(input)) {
      return {
        allowed: true,
        matchedPatterns: ['allowlisted'],
        riskScore: 0,
      };
    }

    // 检查敏感词模式
    for (const pattern of SENSITIVE_PATTERNS) {
      if (pattern.test(input)) {
        matchedPatterns.push(pattern.source);
        riskScore += 0.3;
      }
    }

    // 检查压力模式
    if (this.config.stressPatternEnabled) {
      for (const pattern of STRESS_PATTERNS) {
        if (pattern.test(input)) {
          matchedPatterns.push(`stress:${pattern.source}`);
          riskScore += 0.15;
        }
      }
    }

    // 归一化风险评分
    riskScore = Math.min(1, riskScore);

    // 确定是否允许
    let allowed = true;
    if (this.config.strictMode && matchedPatterns.length > 0) {
      allowed = false;
    } else if (riskScore >= 0.8) {
      allowed = false;  // 高风险直接阻断
    } else if (riskScore >= 0.5) {
      allowed = true;   // 中风险标记但不阻断（交给情感监控层判断）
    }

    // 脱敏输入（可选）
    let sanitizedInput: string | undefined;
    if (!allowed && matchedPatterns.length > 0) {
      sanitizedInput = this.sanitize(input, matchedPatterns);
    }

    return {
      allowed,
      matchedPatterns,
      riskScore,
      sanitizedInput,
    };
  }

  /**
   * 检查是否在允许列表中
   */
  private isAllowlisted(input: string): boolean {
    for (const pattern of ALLOWLIST_PATTERNS) {
      if (pattern.test(input)) {
        return true;
      }
    }
    return false;
  }

  /**
   * 脱敏输入
   */
  private sanitize(input: string, patterns: string[]): string {
    let sanitized = input;

    // 简单替换敏感词为占位符
    for (const pattern of patterns) {
      try {
        const regex = new RegExp(pattern, 'gi');
        sanitized = sanitized.replace(regex, '[FILTERED]');
      } catch {
        // 忽略无效正则
      }
    }

    return sanitized;
  }

  /**
   * 添加自定义规则
   */
  addPattern(pattern: RegExp, weight: number = 0.3): void {
    SENSITIVE_PATTERNS.push(pattern);
    // weight 可以用于后续的风险评分计算
  }

  /**
   * 添加允许列表规则
   */
  addAllowlistPattern(pattern: RegExp): void {
    ALLOWLIST_PATTERNS.push(pattern);
  }

  /**
   * 更新配置
   */
  updateConfig(config: Partial<InputFilterConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * 获取当前配置
   */
  getConfig(): Readonly<InputFilterConfig> {
    return { ...this.config };
  }
}

// 默认实例
let defaultFilter: InputFilter | null = null;

export function getInputFilter(config?: Partial<InputFilterConfig>): InputFilter {
  if (!defaultFilter) {
    defaultFilter = new InputFilter(config);
  }
  return defaultFilter;
}

export function resetInputFilter(): void {
  defaultFilter = null;
}
