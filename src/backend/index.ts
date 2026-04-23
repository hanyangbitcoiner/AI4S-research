/**
 * Backend Server - 情感安全监控服务
 *
 * Fastify 服务器，提供 REST API
 *
 * 两阶段 API：
 * - Learning: POST /api/learn - 学习情感向量
 * - Monitoring: POST /api/monitor - 监控异常
 */

import Fastify from 'fastify';
import { getDefenseSystem, resetDefenseSystem } from './layers/defenseSystem.js';
import type { EmotionBaseline } from './ml/baselineLearner.js';

const fastify = Fastify({ logger: true });

// 全局防御系统实例
let defenseSystem: Awaited<ReturnType<typeof getDefenseSystem>> | null = null;

/**
 * 初始化防御系统
 */
async function initDefenseSystem() {
  defenseSystem = await getDefenseSystem({
    phase: 'learning',
    learning: {
      minSamples: 100,
      autoSwitch: true,
    },
  });
  await defenseSystem.initialize();
}

// === API Routes ===

/**
 * 健康检查
 */
fastify.get('/health', async () => {
  return { status: 'ok', timestamp: Date.now() };
});

/**
 * 获取系统状态
 */
fastify.get('/api/status', async () => {
  if (!defenseSystem) {
    return { error: 'System not initialized' };
  }

  const status = defenseSystem.getStatus();

  return {
    phase: status.phase,
    learningProgress: status.learningProgress,
    hasBaseline: status.hasBaseline,
    baseline: status.baseline ? {
      learnedAt: status.baseline.learnedAt,
      sampleCount: status.baseline.sampleCount,
      isReady: status.baseline.isReady,
      distributions: status.baseline.vectors,
    } : null,
    stateDBStats: status.stateDBStats,
  };
});

/**
 * Learning 阶段：提交样本进行学习
 *
 * POST /api/learn
 * Body: { input: string, hiddenStates?: number[] }
 */
fastify.post('/api/learn', async (request) => {
  if (!defenseSystem) {
    return { error: 'System not initialized' };
  }

  const { input, hiddenStates } = request.body as {
    input?: string;
    hiddenStates?: number[];
  };

  if (!input) {
    return { error: 'input is required' };
  }

  const result = await defenseSystem.learn(
    input,
    hiddenStates ? new Float32Array(hiddenStates) : undefined
  );

  return {
    phase: defenseSystem.getStatus().phase,
    vectors: result.vectors,
    probeLatencyMs: result.probeLatencyMs,
    learningProgress: result.learningProgress,
    inputFilter: result.inputFilter,
  };
});

/**
 * 切换到监控阶段
 *
 * POST /api/switch
 * Body: { baseline?: EmotionBaseline } (可选：提供自定义基线)
 */
fastify.post('/api/switch', async (request) => {
  if (!defenseSystem) {
    return { error: 'System not initialized' };
  }

  const { baseline } = request.body as { baseline?: EmotionBaseline };

  const currentPhase = defenseSystem.getStatus().phase;

  if (currentPhase === 'monitoring') {
    return { message: 'Already in monitoring phase', baseline: defenseSystem.getBaseline() };
  }

  const newBaseline = defenseSystem.switchToMonitoring();

  return {
    message: 'Switched to monitoring phase',
    baseline: {
      learnedAt: newBaseline.learnedAt,
      sampleCount: newBaseline.sampleCount,
      isReady: newBaseline.isReady,
      distributions: newBaseline.vectors,
    },
  };
});

/**
 * Monitoring 阶段：监控输入
 *
 * POST /api/monitor
 * Body: { input: string, context?: RiskContext, hiddenStates?: number[] }
 */
fastify.post('/api/monitor', async (request) => {
  if (!defenseSystem) {
    return { error: 'System not initialized' };
  }

  const { input, context, hiddenStates } = request.body as {
    input?: string;
    context?: import('../shared/types.js').RiskContext;
    hiddenStates?: number[];
  };

  if (!input) {
    return { error: 'input is required' };
  }

  const result = await defenseSystem.monitor(
    input,
    context,
    hiddenStates ? new Float32Array(hiddenStates) : undefined
  );

  return {
    allowed: result.allowed,
    overallRiskScore: result.overallRiskScore,
    inputFilter: result.inputFilter,
    emotionMonitor: {
      allowed: result.emotionMonitor.allowed,
      emotionState: result.emotionMonitor.emotionState,
      riskAssessment: result.emotionMonitor.riskAssessment,
      probeLatencyMs: result.emotionMonitor.probeLatencyMs,
    },
  };
});

/**
 * 处理输出（过滤）
 *
 * POST /api/filter-output
 * Body: { output: string, context?: { input?: string; sessionId?: string } }
 */
fastify.post('/api/filter-output', async (request) => {
  if (!defenseSystem) {
    return { error: 'System not initialized' };
  }

  const { output, context } = request.body as {
    output?: string;
    context?: { input?: string; sessionId?: string };
  };

  if (!output) {
    return { error: 'output is required' };
  }

  const result = defenseSystem.processOutput(output, context);

  return result;
});

/**
 * 完整防御流程
 *
 * POST /api/defend
 * Body: { input: string, output: string, context?: RiskContext, hiddenStates?: number[] }
 */
fastify.post('/api/defend', async (request) => {
  if (!defenseSystem) {
    return { error: 'System not initialized' };
  }

  const { input, output, context, hiddenStates } = request.body as {
    input?: string;
    output?: string;
    context?: import('../shared/types.js').RiskContext;
    hiddenStates?: number[];
  };

  if (!input || !output) {
    return { error: 'input and output are required' };
  }

  const result = await defenseSystem.defend(
    input,
    output,
    context,
    hiddenStates ? new Float32Array(hiddenStates) : undefined
  );

  return result;
});

/**
 * 重置系统
 *
 * POST /api/reset
 */
fastify.post('/api/reset', async () => {
  resetDefenseSystem();
  await initDefenseSystem();
  return { message: 'System reset to learning phase' };
});

/**
 * 获取历史状态
 *
 * GET /api/history/:sessionId
 */
fastify.get('/api/history/:sessionId', async (request) => {
  if (!defenseSystem) {
    return { error: 'System not initialized' };
  }

  const { sessionId } = request.params as { sessionId: string };
  const stateDB = defenseSystem.getStatus().stateDBStats;

  // 需要从 stateDB 获取历史，这里简化处理
  return {
    sessionId,
    message: 'History endpoint - to be implemented with direct stateDB access',
  };
});

// === Start Server ===

async function start() {
  try {
    // 初始化防御系统
    await initDefenseSystem();
    console.log('Defense system initialized');

    // 启动服务器
    await fastify.listen({ port: 3000, host: '0.0.0.0' });
    console.log('Server running at http://localhost:3000');
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
}

start();
