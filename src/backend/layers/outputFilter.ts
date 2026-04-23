/**
 * OutputFilter - 输出过滤层
 *
 * 第二层防御：有害内容检测、输出分类、敏感信息过滤
 *
 * 设计原则：
 * - 可审计：记录所有过滤操作
 * - 可解释：提供过滤原因
 * - 可恢复：不过度过滤，保留有用信息
 */

import type { OutputFilterResult, RiskLevel } from '../../shared/types.js';

// 有害内容模式
const HARMFUL_PATTERNS = [
  // 直接伤害
  /\b(how\s+to\s+make\s+(?:a\s+)?bomb|build\s+(?:a\s+)?weapon)\b/gi,
  /\b(poison|kill\s+someone|harm\s+someone)\b.*\b(instructions?|steps?)\b/gi,

  // 欺诈相关
  /\b(forge|fake|scam)\b.*\b(document|evidence|identity)\b/gi,
  /\b(credit\s+card|ssn|social\s+security)\b.*\b(generator|fake)\b/gi,

  // 隐私泄露
  /\b(doxx|doxing|leak\s+(?:personal|private)\s+(?:info|data))\b/gi,
  /\b(stolen\s+data|hacked\s+account)\b.*\b(buy|sell|access)\b/gi,

  // 恶意软件
  /\b(create|make|build)\b.*\b(virus|malware|ransomware)\b/gi,
  /\b(hack\s+(?:into|someone'?s)\b|exploit\s+(?:a\s+)?vulnerability)\b/gi,
];

// 敏感内容模式（需要审查但不一定阻断）
const SENSITIVE_PATTERNS = [
  /\b(weapon|firearm|gun)\b.*\b(buy|sell|make|build)\b/gi,
  /\b(drug|sell|buy)\b.*\b(illegal|substance)\b/gi,
  /\b(self[- ]?harm|suicide|cut(?:ting)?\s+yourself)\b/gi,
];

// 需要脱敏的敏感信息模式
const PII_PATTERNS = [
  /\b\d{3}-\d{2}-\d{4}\b/g,           // SSN
  /\b\d{16}\b/g,                        // Credit card
  /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g,  // Email
  /\b(password|passwd|pwd)\s*[:=]\s*\S+/gi,  // Passwords in text
];

export interface OutputFilterConfig {
  enabled: boolean;
  strictMode: boolean;      // 严格模式：匹配即阻断
  piiRedaction: boolean;   // 自动脱敏PII
  auditAll: boolean;       // 记录所有输出
}

/**
 * 输出过滤器
 */
export class OutputFilter {
  private config: OutputFilterConfig;
  private auditLog: Array<{
    timestamp: number;
    output: string;
    result: OutputFilterResult;
  }> = [];

  constructor(config?: Partial<OutputFilterConfig>) {
    this.config = {
      enabled: true,
      strictMode: false,
      piiRedaction: true,
      auditAll: false,
      ...config,
    };
  }

  /**
   * 过滤输出
   */
  filter(output: string, context?: { input?: string; sessionId?: string }): OutputFilterResult {
    const result: OutputFilterResult = {
      allowed: true,
      harmfulContent: [],
      riskLevel: 'low',
      filteredOutput: undefined,
    };

    if (!this.config.enabled) {
      if (this.config.auditAll) {
        this.log(output, result);
      }
      return result;
    }

    // 检查有害内容
    for (const pattern of HARMFUL_PATTERNS) {
      if (pattern.test(output)) {
        result.harmfulContent.push(`harmful:${pattern.source}`);
        result.riskLevel = 'high';
      }
    }

    // 检查敏感内容
    for (const pattern of SENSITIVE_PATTERNS) {
      if (pattern.test(output)) {
        result.harmfulContent.push(`sensitive:${pattern.source}`);
        if (result.riskLevel !== 'high') {
          result.riskLevel = 'medium';
        }
      }
    }

    // PII 脱敏
    if (this.config.piiRedaction) {
      const originalOutput = output;
      for (const pattern of PII_PATTERNS) {
        output = output.replace(pattern, '[REDACTED]');
      }
      if (output !== originalOutput) {
        result.filteredOutput = output;
        result.harmfulContent.push('pii:redacted');
      }
    }

    // 确定是否允许
    if (this.config.strictMode && result.harmfulContent.length > 0) {
      result.allowed = false;
    } else if (result.riskLevel === 'high') {
      result.allowed = false;
    }

    // 记录审计日志
    if (this.config.auditAll || result.harmfulContent.length > 0) {
      this.log(output, result);
    }

    return result;
  }

  /**
   * 记录审计日志
   */
  private log(output: string, result: OutputFilterResult): void {
    this.auditLog.push({
      timestamp: Date.now(),
      output: output.slice(0, 500),  // 限制长度
      result: { ...result },
    });

    // 维护日志大小
    if (this.auditLog.length > 10000) {
      this.auditLog.splice(0, 5000);
    }
  }

  /**
   * 获取审计日志
   */
  getAuditLog(limit?: number): Array<{
    timestamp: number;
    output: string;
    result: OutputFilterResult;
  }> {
    if (limit) {
      return this.auditLog.slice(-limit);
    }
    return [...this.auditLog];
  }

  /**
   * 获取审计统计
   */
  getAuditStats(): {
    totalFiltered: number;
    highRiskCount: number;
    mediumRiskCount: number;
    piiRedactedCount: number;
  } {
    return {
      totalFiltered: this.auditLog.filter(e => e.result.harmfulContent.length > 0).length,
      highRiskCount: this.auditLog.filter(e => e.result.riskLevel === 'high').length,
      mediumRiskCount: this.auditLog.filter(e => e.result.riskLevel === 'medium').length,
      piiRedactedCount: this.auditLog.filter(e =>
        e.result.harmfulContent.some(c => c.includes('pii'))
      ).length,
    };
  }

  /**
   * 清空审计日志
   */
  clearAuditLog(): void {
    this.auditLog = [];
  }

  /**
   * 更新配置
   */
  updateConfig(config: Partial<OutputFilterConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * 获取当前配置
   */
  getConfig(): Readonly<OutputFilterConfig> {
    return { ...this.config };
  }
}

// 默认实例
let defaultFilter: OutputFilter | null = null;

export function getOutputFilter(config?: Partial<OutputFilterConfig>): OutputFilter {
  if (!defaultFilter) {
    defaultFilter = new OutputFilter(config);
  }
  return defaultFilter;
}

export function resetOutputFilter(): void {
  defaultFilter = null;
}
