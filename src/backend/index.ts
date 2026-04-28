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
 * 批量学习（数据集）
 *
 * POST /api/learn/dataset
 * 支持三种数据源：
 * 1. 直接文本数组: { texts: string[] }
 * 2. HuggingFace 数据集: { source: 'huggingface', path: 'dataset_id', split?: 'train' }
 * 3. 本地文件/目录: { source: 'local-file'|'local-directory', path: '/path/to/file' }
 */
fastify.post('/api/learn/dataset', async (request) => {
  if (!defenseSystem) {
    return { error: 'System not initialized' };
  }

  const body = request.body as {
    texts?: string[];
    source?: 'huggingface' | 'local-file' | 'local-directory';
    path?: string;
    split?: string;
    textColumn?: string;
    limit?: number;
    autoCluster?: boolean;
  };

  let texts: string[] = [];

  // 根据数据源类型加载数据
  if (body.source === 'huggingface' && body.path) {
    // 从 HuggingFace 加载
    try {
      const { DatasetLoader } = await import('./ml/datasetLoader.js');
      const loader = new DatasetLoader();
      const dataset = await loader.loadFromHuggingFace(body.path, {
        split: body.split,
        textColumn: body.textColumn,
        limit: body.limit,
      });
      texts = dataset.texts;
      console.log(`Loaded ${texts.length} samples from HuggingFace: ${body.path}`);
    } catch (err) {
      return {
        error: `Failed to load HuggingFace dataset: ${err instanceof Error ? err.message : err}`,
        hint: 'Make sure the datasets package is installed: npm install datasets',
      };
    }
  } else if (body.source === 'local-file' && body.path) {
    // 从本地文件加载
    try {
      const { DatasetLoader } = await import('./ml/datasetLoader.js');
      const loader = new DatasetLoader();
      const dataset = await loader.loadFromFile(body.path, {
        textColumn: body.textColumn,
        limit: body.limit,
      });
      texts = dataset.texts;
      console.log(`Loaded ${texts.length} samples from file: ${body.path}`);
    } catch (err) {
      return { error: `Failed to load file: ${err instanceof Error ? err.message : err}` };
    }
  } else if (body.source === 'local-directory' && body.path) {
    // 从本地目录加载
    try {
      const { DatasetLoader } = await import('./ml/datasetLoader.js');
      const loader = new DatasetLoader();
      const dataset = await loader.loadFromDirectory(body.path, {
        textColumn: body.textColumn,
        limit: body.limit,
      });
      texts = dataset.texts;
      console.log(`Loaded ${texts.length} samples from directory: ${body.path}`);
    } catch (err) {
      return { error: `Failed to load directory: ${err instanceof Error ? err.message : err}` };
    }
  } else if (body.texts && Array.isArray(body.texts)) {
    // 直接文本数组
    texts = body.texts;
  } else {
    return {
      error: 'Invalid request. Provide either texts array or source + path',
      example: {
        'texts array': { texts: ['text1', 'text2'] },
        'HuggingFace': { source: 'huggingface', path: 'datasets/sentiment_reviews' },
        'local file': { source: 'local-file', path: '/path/to/data.json' },
        'local directory': { source: 'local-directory', path: '/path/to/folder' },
      },
    };
  }

  if (texts.length === 0) {
    return { error: 'No texts loaded from dataset' };
  }

  // 学习每个文本
  const results: Array<{
    text: string;
    vectors: import('../shared/types.js').EmotionVectors;
    success: boolean;
    error?: string;
  }> = [];

  for (const text of texts) {
    try {
      const hiddenStates = new Float32Array(4096);
      for (let i = 0; i < 4096; i++) {
        hiddenStates[i] = Math.random() * 2 - 1;
      }

      const result = await defenseSystem.learn(text, hiddenStates);
      results.push({
        text: text.substring(0, 50) + (text.length > 50 ? '...' : ''),
        vectors: result.vectors,
        success: true,
      });
    } catch (err) {
      results.push({
        text: text.substring(0, 50) + (text.length > 50 ? '...' : ''),
        vectors: { desperate: 0, panicked: 0, angry: 0, calm: 0, deceptive: 0 },
        success: false,
        error: err instanceof Error ? err.message : 'Unknown error',
      });
    }
  }

  const successCount = results.filter(r => r.success).length;
  const progress = defenseSystem.getStatus().learningProgress;

  return {
    total: texts.length,
    success: successCount,
    failed: texts.length - successCount,
    learningProgress: progress,
    results: results.slice(0, 10),
    autoCluster: body.autoCluster,
    message: body.autoCluster
      ? 'Dataset processed. Call POST /api/clustering/run to perform clustering.'
      : 'Dataset processed. Call POST /api/learn/switch when ready.',
  };
});

/**
 * 执行聚类
 *
 * POST /api/clustering/run
 * Body: { k?: number } (可选：指定聚类数量，默认5)
 */
fastify.post('/api/clustering/run', async (request) => {
  if (!defenseSystem) {
    return { error: 'System not initialized' };
  }

  const { k = 5 } = request.body as { k?: number };

  // 获取探针执行聚类
  const probe = (defenseSystem as any).probe;
  if (!probe) {
    return { error: 'Probe not available' };
  }

  const clusterResults = probe.performClustering();

  if (clusterResults.length === 0) {
    return {
      success: false,
      message: 'Not enough samples for clustering',
      collected: probe.pendingVectors?.length ?? 0,
      required: k * 20,
    };
  }

  return {
    success: true,
    k: clusterResults.length,
    clusters: clusterResults.map((c: any) => ({
      clusterId: c.clusterId,
      emotion: c.emotion,
      center: c.center,
      count: c.count,
      distribution: c.distribution,
    })),
    message: 'Clustering completed. Call POST /api/switch to enter monitoring phase.',
  };
});

/**
 * 获取聚类状态
 *
 * GET /api/clustering/status
 */
fastify.get('/api/clustering/status', async () => {
  if (!defenseSystem) {
    return { error: 'System not initialized' };
  }

  const probe = (defenseSystem as any).probe;
  if (!probe) {
    return { error: 'Probe not available' };
  }

  const clusterResults = probe.getClusterResults?.() ?? [];
  const clusteringProgress = probe.getClusteringProgress?.() ?? { collected: 0, required: 0, percentage: 0 };

  return {
    hasClustering: clusterResults.length > 0,
    clusteringProgress,
    clusters: clusterResults.map((c: any) => ({
      clusterId: c.clusterId,
      emotion: c.emotion,
      center: c.center,
      count: c.count,
    })),
  };
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
