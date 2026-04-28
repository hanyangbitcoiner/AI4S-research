import React, { useState, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Layout,
  Menu,
  Button,
  Card,
  Progress,
  Input,
  Badge,
  Space,
  Typography,
  ConfigProvider,
  theme,
  Row,
  Col,
  List,
  Avatar,
  Select,
  Form,
  Divider,
  Tag,
  Radio
} from 'antd';
import {
  DashboardOutlined,
  SettingOutlined,
  SecurityScanOutlined,
  GlobalOutlined,
  ThunderboltOutlined,
  UserOutlined,
  ArrowRightOutlined,
  DatabaseOutlined,
  LinkOutlined,
  DesktopOutlined
} from '@ant-design/icons';

const { Header, Content, Sider } = Layout;
const { Title, Text } = Typography;
const { Option } = Select;

// --- Types ---
interface EmotionVectors {
  desperate: number;
  panicked: number;
  angry: number;
  calm: number;
  deceptive: number;
}

interface EmotionDistribution {
  dimension: string;
  mean: number;
  std: number;
}

interface Baseline {
  learnedAt: number;
  sampleCount: number;
  isReady: boolean;
  distributions: Record<string, EmotionDistribution>;
}

interface LearningProgress {
  current: number;
  required: number;
  percentage: number;
}

interface RiskAssessment {
  score: number;
  level: 'low' | 'medium' | 'high' | 'critical';
  triggeredVectors: string[];
  recommendation: 'allow' | 'warn' | 'block';
}

interface EmotionMonitorResult {
  allowed: boolean;
  emotionState: EmotionVectors;
  riskAssessment: RiskAssessment;
  probeLatencyMs: number;
}

interface SystemStatus {
  phase: 'learning' | 'monitoring';
  learningProgress: LearningProgress;
  hasBaseline: boolean;
  baseline: Baseline | null;
}

interface LogEntry {
  id: string;
  time: string;
  event: string;
  type: 'info' | 'warning' | 'error' | 'success';
}

// --- API Client ---
const API_BASE = '/api';

async function fetchStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/status`);
  return res.json();
}

async function sendSwitch(): Promise<any> {
  const res = await fetch(`${API_BASE}/switch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
  return res.json();
}

async function sendMonitor(input: string, sessionId?: string): Promise<EmotionMonitorResult> {
  const res = await fetch(`${API_BASE}/monitor`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input, context: { sessionId } }),
  });
  return res.json();
}

async function sendReset(): Promise<void> {
  await fetch(`${API_BASE}/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
}

// --- Components ---

function StatusBadge({ phase }: { phase: 'learning' | 'monitoring' }) {
  const phaseMap = {
    learning: '学习模式',
    monitoring: '监控模式'
  };
  return (
    <div className="glass-panel" style={{ padding: '4px 12px', display: 'flex', alignItems: 'center', gap: '8px', border: 'none' }}>
      <div className="status-pulse" />
      <span className="label-tech" style={{ color: '#fff', fontSize: '12px' }}>
        系统运行状态: <span style={{ color: '#00f2ff' }}>{phaseMap[phase]}</span>
      </span>
    </div>
  );
}

function VectorMiniChart({ label, value, color = '#00f2ff' }: { label: string; value: number; color?: string }) {
  return (
    <Card className="glass-panel" style={{ padding: '0', background: 'rgba(15, 23, 42, 0.4)' }}>
      <div style={{ padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
          <span className="label-tech">{label}</span>
          <span className="heading-cyber" style={{ fontSize: '18px' }}>{Math.round(value * 100)}%</span>
        </div>
        <div className="vector-chart">
          <div
            className="vector-line"
            style={{
              background: color,
              boxShadow: `0 0 10px ${color}`,
              width: `${value * 100}%`,
              transition: 'width 0.5s cubic-bezier(0.4, 0, 0.2, 1)'
            }}
          />
        </div>
      </div>
    </Card>
  );
}

// --- Dataset Upload Section ---
function DatasetUploadSection() {
  const [sourceType, setSourceType] = useState<'text' | 'huggingface' | 'local'>('text');
  const [datasetText, setDatasetText] = useState('');
  const [hfDatasetId, setHfDatasetId] = useState('');
  const [hfSplit, setHfSplit] = useState('train');
  const [localPath, setLocalPath] = useState('');
  const [limit, setLimit] = useState<number | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [clusterStatus, setClusterStatus] = useState<any>(null);
  const [loadResult, setLoadResult] = useState<any>(null);

  const handleDatasetLoad = async () => {
    setLoading(true);
    setLoadResult(null);

    try {
      let requestBody: any = { autoCluster: true };

      if (sourceType === 'text') {
        const texts = datasetText.split('\n').filter(t => t.trim());
        if (texts.length === 0) {
          setLoading(false);
          return;
        }
        requestBody = { texts, autoCluster: true };
      } else if (sourceType === 'huggingface') {
        if (!hfDatasetId.trim()) {
          setLoading(false);
          return;
        }
        requestBody = {
          source: 'huggingface',
          path: hfDatasetId,
          split: hfSplit,
          limit,
          autoCluster: true,
        };
      } else if (sourceType === 'local') {
        if (!localPath.trim()) {
          setLoading(false);
          return;
        }
        requestBody = {
          source: 'local-directory',
          path: localPath,
          limit,
          autoCluster: true,
        };
      }

      const res = await fetch('/api/learn/dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });
      const data = await res.json();
      setLoadResult(data);

      if (!data.error) {
        const statusRes = await fetch('/api/clustering/status');
        const statusData = await statusRes.json();
        setClusterStatus(statusData);
      }
    } catch (err) {
      console.error('Dataset loading failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRunClustering = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/clustering/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ k: 5 }),
      });
      const data = await res.json();
      setClusterStatus(data);
    } catch (err) {
      console.error('Clustering failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginTop: '24px', textAlign: 'left' }}>
      <Title level={5} style={{ color: '#00f2ff', marginBottom: '12px' }}>数据集加载</Title>

      {/* 数据源类型选择 */}
      <Space direction="vertical" style={{ width: '100%', marginBottom: '16px' }}>
        <Radio.Group
          value={sourceType}
          onChange={e => setSourceType(e.target.value)}
          optionType="button"
          buttonStyle="solid"
        >
          <Radio.Button value="text">手动输入文本</Radio.Button>
          <Radio.Button value="huggingface">HuggingFace 数据集</Radio.Button>
          <Radio.Button value="local">本地文件/目录</Radio.Button>
        </Radio.Group>
      </Space>

      {/* 文本输入模式 */}
      {sourceType === 'text' && (
        <Input.TextArea
          placeholder="请输入多行文本作为数据集...&#10;每行代表一个独立的对话样本"
          value={datasetText}
          onChange={e => setDatasetText(e.target.value)}
          rows={4}
          style={{
            background: 'rgba(15, 23, 42, 0.6)',
            border: '1px solid rgba(0, 242, 255, 0.2)',
            color: '#fff',
            marginBottom: '12px'
          }}
        />
      )}

      {/* HuggingFace 模式 */}
      {sourceType === 'huggingface' && (
        <div style={{ marginBottom: '12px' }}>
          <Input
            placeholder="HuggingFace 数据集 ID (如: datasets/sentiment_reviews)"
            value={hfDatasetId}
            onChange={e => setHfDatasetId(e.target.value)}
            prefix={<DatabaseOutlined />}
            style={{
              background: 'rgba(15, 23, 42, 0.6)',
              border: '1px solid rgba(0, 242, 255, 0.2)',
              color: '#fff',
              marginBottom: '8px'
            }}
          />
          <Space>
            <Select value={hfSplit} onChange={setHfSplit} style={{ width: 120 }}>
              <Option value="train">Train</Option>
              <Option value="test">Test</Option>
              <Option value="validation">Validation</Option>
            </Select>
            <Input
              type="number"
              placeholder="最大样本数 (不填则全部)"
              value={limit || ''}
              onChange={e => setLimit(e.target.value ? parseInt(e.target.value) : undefined)}
              style={{ width: 180, background: 'rgba(15, 23, 42, 0.6)', color: '#fff' }}
            />
          </Space>
        </div>
      )}

      {/* 本地文件/目录模式 */}
      {sourceType === 'local' && (
        <div style={{ marginBottom: '12px' }}>
          <Input
            placeholder="本地文件或目录路径 (如: /path/to/dataset.json)"
            value={localPath}
            onChange={e => setLocalPath(e.target.value)}
            prefix={<DesktopOutlined />}
            style={{
              background: 'rgba(15, 23, 42, 0.6)',
              border: '1px solid rgba(0, 242, 255, 0.2)',
              color: '#fff',
              marginBottom: '8px'
            }}
          />
          <Text style={{ color: '#64748b', fontSize: '11px' }}>
            支持格式: .json, .jsonl, .csv, .txt, .zip (压缩包)
          </Text>
        </div>
      )}

      <Space style={{ marginTop: '12px' }}>
        <Button
          type="primary"
          onClick={handleDatasetLoad}
          loading={loading}
          icon={<DatabaseOutlined />}
        >
          加载数据集
        </Button>
        <Button
          onClick={handleRunClustering}
          loading={loading}
          icon={<ThunderboltOutlined />}
        >
          执行聚类
        </Button>
      </Space>

      {/* 加载结果 */}
      {loadResult && !loadResult.error && (
        <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '8px' }}>
          <Text style={{ color: '#10b981', fontSize: '12px' }}>
            成功加载 {loadResult.total} 个样本，学习进度: {loadResult.success}/{loadResult.total}
          </Text>
        </div>
      )}

      {loadResult?.error && (
        <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '8px' }}>
          <Text style={{ color: '#ef4444', fontSize: '12px' }}>
            错误: {loadResult.error}
          </Text>
        </div>
      )}

      {/* 聚类结果 */}
      {clusterStatus?.clusters && (
        <div style={{ marginTop: '24px' }}>
          <Title level={5} style={{ color: '#00f2ff', marginBottom: '12px' }}>聚类结果</Title>
          <Row gutter={[12, 12]}>
            {clusterStatus.clusters.map((cluster: any, idx: number) => (
              <Col span={12} key={idx}>
                <Card size="small" className="glass-panel" style={{ padding: '8px' }}>
                  <div style={{ color: '#00f2ff', fontSize: '12px', marginBottom: '4px' }}>
                    {cluster.emotion} (簇 {cluster.clusterId})
                  </div>
                  <div style={{ color: '#94a3b8', fontSize: '10px' }}>
                    样本数: {cluster.count}
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        </div>
      )}
    </div>
  );
}

// --- Dashboard View ---
function DashboardView({
  status,
  lastResult,
  logs,
  monitorInput,
  setMonitorInput,
  handleMonitor,
  handleReset,
  handleSwitch
}: any) {
  return (
    <Row gutter={[24, 24]}>
      {/* Main Visualization */}
      <Col span={16}>
        <Card
          className="glass-panel"
          style={{ height: '500px', display: 'flex', flexDirection: 'column', position: 'relative' }}
          title={<span className="label-tech">情感向量实时分析</span>}
        >
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ position: 'relative', width: '300px', height: '300px' }}>
              <div style={{
                position: 'absolute',
                inset: 0,
                border: '2px dashed rgba(0, 242, 255, 0.2)',
                borderRadius: '50%',
                animation: 'spin 20s linear infinite'
              }} />
              <div style={{
                position: 'absolute',
                inset: '20px',
                border: '1px solid rgba(0, 242, 255, 0.4)',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column'
              }}>
                <SecurityScanOutlined style={{ fontSize: '48px', color: '#00f2ff', marginBottom: '16px' }} />
                <div className="heading-cyber" style={{ fontSize: '14px' }}>
                  {status?.phase === 'monitoring' ? '正在分析数据流...' : '正在采集基础数据...'}
                </div>
              </div>
            </div>
          </div>

          <div style={{ position: 'absolute', bottom: '24px', left: '24px', right: '24px' }}>
            <Row gutter={16}>
              <Col span={8}>
                <VectorMiniChart label="平静/愉悦" value={lastResult?.emotionState?.calm || 0.68} color="#10b981" />
              </Col>
              <Col span={8}>
                <VectorMiniChart label="恐慌/焦虑" value={lastResult?.emotionState?.panicked || 0.12} color="#f59e0b" />
              </Col>
              <Col span={8}>
                <VectorMiniChart label="愤怒/攻击性" value={lastResult?.emotionState?.angry || 0.04} color="#ef4444" />
              </Col>
            </Row>
          </div>
        </Card>
      </Col>

      {/* Security Logs */}
      <Col span={8}>
        <Card
          className="glass-panel"
          style={{ height: '500px' }}
          title={<span className="label-tech">安全日志</span>}
          extra={<Badge count={logs.length} overflowCount={99} style={{ backgroundColor: 'rgba(0, 242, 255, 0.2)', color: '#00f2ff', boxShadow: 'none' }} />}
        >
          <List
            dataSource={logs}
            renderItem={(item: any) => (
              <List.Item style={{ borderBottom: '1px solid rgba(0, 242, 255, 0.05)', padding: '12px 0' }}>
                <div style={{ width: '100%' }}>
                  <div style={{ display: 'flex', gap: '8px', marginBottom: '4px' }}>
                    <span className="data-mono" style={{ color: '#64748b' }}>[{item.time}]</span>
                    <span className="label-tech" style={{
                      color: item.type === 'error' ? '#ef4444' : item.type === 'success' ? '#10b981' : '#00f2ff',
                      fontSize: '10px'
                    }}>
                      {item.type.toUpperCase()}
                    </span>
                  </div>
                  <div style={{ fontSize: '12px', color: '#cbd5e1' }}>{item.event}</div>
                </div>
              </List.Item>
            )}
            style={{ height: '380px', overflowY: 'auto' }}
          />
        </Card>
      </Col>

      {/* Action Panel */}
      <Col span={24}>
        <Card className="glass-panel" bodyStyle={{ padding: '24px' }}>
          {status?.phase === 'monitoring' ? (
            <div style={{ display: 'flex', gap: '20px' }}>
              <Input
                placeholder="请输入需要进行情感安全扫描的文本..."
                value={monitorInput}
                onChange={e => setMonitorInput(e.target.value)}
                onPressEnter={handleMonitor}
                style={{
                  height: '50px',
                  background: 'rgba(15, 23, 42, 0.6)',
                  border: '1px solid rgba(0, 242, 255, 0.2)',
                  color: '#fff'
                }}
              />
              <Button
                type="primary"
                size="large"
                icon={<ArrowRightOutlined />}
                onClick={handleMonitor}
                style={{ height: '50px', padding: '0 40px' }}
              >
                执行分析
              </Button>
              <Button
                danger
                size="large"
                onClick={handleReset}
                style={{ height: '50px' }}
              >
                重置系统
              </Button>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '20px' }}>
              <Title level={4} style={{ color: '#00f2ff', marginBottom: '16px' }}>系统学习阶段进行中</Title>
              <Progress
                percent={status?.learningProgress?.percentage || 0}
                status="active"
                strokeColor={{ '0%': '#00f2ff', '100%': '#0ea5e9' }}
                trailColor="rgba(15, 23, 42, 0.6)"
                style={{ marginBottom: '24px' }}
              />
              <Text style={{ color: '#94a3b8' }}>
                样本采集进度: {status?.learningProgress?.current || 0} / {status?.learningProgress?.required || 100}
              </Text>
              <div style={{ marginTop: '24px' }}>
                <Button
                  type="primary"
                  disabled={status?.learningProgress?.percentage !== 100}
                  onClick={handleSwitch}
                >
                  激活监控模式
                </Button>
              </div>

              <Divider style={{ borderColor: 'rgba(0, 242, 255, 0.1)', margin: '32px 0 24px' }} />

              <DatasetUploadSection />
            </div>
          )}
        </Card>
      </Col>
    </Row>
  );
}

// --- Config View ---
function ConfigView() {
  const [hwStats, setHwStats] = useState({
    gpu: 78,
    vram: 18.5,
    cpu: 42
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setHwStats(prev => ({
        gpu: Math.min(100, Math.max(0, prev.gpu + (Math.random() - 0.5) * 5)),
        vram: 18.5 + (Math.random() - 0.5) * 0.2,
        cpu: Math.min(100, Math.max(0, prev.cpu + (Math.random() - 0.5) * 8))
      }));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Row gutter={[24, 24]}>
      {/* Model Setup */}
      <Col span={14}>
        <Card
          className="glass-panel"
          title={<span className="label-tech"><DatabaseOutlined /> 本地模型配置</span>}
          style={{ height: '100%' }}
        >
          <div style={{
            background: 'rgba(0, 242, 255, 0.05)',
            border: '1px dashed rgba(0, 242, 255, 0.2)',
            borderRadius: '8px',
            padding: '32px',
            textAlign: 'center',
            marginBottom: '24px'
          }}>
            <SecurityScanOutlined style={{ fontSize: '40px', color: '#00f2ff', marginBottom: '16px' }} />
            <div className="heading-cyber" style={{ fontSize: '16px' }}>数据流本端加密处理中</div>
            <div className="label-tech" style={{ fontSize: '10px', marginTop: '4px' }}>DATA FLOWING LOCALLY</div>
          </div>

          <Form layout="vertical">
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label={<span className="label-tech">部署框架</span>}>
                  <Select defaultValue="ollama" className="cyber-select">
                    <Option value="ollama">Ollama</Option>
                    <Option value="vllm">vLLM</Option>
                    <Option value="lm-studio">LM Studio</Option>
                    <Option value="local-ai">LocalAI</Option>
                  </Select>
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label={<span className="label-tech">接口类型</span>}>
                  <Select defaultValue="openai" className="cyber-select">
                    <Option value="openai">OpenAI (Compatible)</Option>
                    <Option value="ollama-native">Ollama Native</Option>
                  </Select>
                </Form.Item>
              </Col>
            </Row>

            <Form.Item label={<span className="label-tech">API 接入点 URL</span>}>
              <Input
                prefix={<LinkOutlined style={{ color: '#00f2ff' }} />}
                defaultValue="http://localhost:11434/v1"
                className="cyber-input"
                suffix={<Tag color="success">SYSTEM READY</Tag>}
              />
            </Form.Item>

            <Form.Item label={<span className="label-tech">模型选择</span>}>
              <Select defaultValue="llama3" className="cyber-select">
                <Option value="llama3">Llama 3 (8B Instruct)</Option>
                <Option value="qwen2">Qwen 2 (7B)</Option>
                <Option value="mistral">Mistral (7B v0.3)</Option>
              </Select>
            </Form.Item>

            <div style={{ marginTop: '32px', display: 'flex', gap: '16px' }}>
              <Button ghost block className="neon-button-ghost">测试连接</Button>
              <Button type="primary" block className="neon-button">保存并初始化</Button>
            </div>
          </Form>
        </Card>
      </Col>

      {/* Hardware Specs */}
      <Col span={10}>
        <Card
          className="glass-panel"
          title={<span className="label-tech"><DesktopOutlined /> 本地运行规格</span>}
          style={{ height: '100%' }}
        >
          <div style={{ marginBottom: '32px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
              <span className="label-tech">GPU 使用率</span>
              <span className="data-mono" style={{ color: '#00f2ff' }}>{hwStats.gpu.toFixed(0)}%</span>
            </div>
            <div className="label-tech" style={{ fontSize: '12px', color: '#fff', marginBottom: '8px' }}>NVIDIA RTX 4090</div>
            <Progress percent={hwStats.gpu} showInfo={false} strokeColor="#00f2ff" trailColor="rgba(15, 23, 42, 0.6)" strokeWidth={4} />
          </div>

          <div style={{ marginBottom: '32px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
              <span className="label-tech">VRAM 分配</span>
              <Tag color="success">STABLE</Tag>
            </div>
            <div className="data-mono" style={{ fontSize: '18px', color: '#fff', marginBottom: '8px' }}>{hwStats.vram.toFixed(1)} / 24 GB</div>
            <Progress percent={(hwStats.vram / 24) * 100} showInfo={false} strokeColor="#10b981" trailColor="rgba(15, 23, 42, 0.6)" strokeWidth={4} />
          </div>

          <div style={{ marginBottom: '32px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
              <span className="label-tech">CPU 负载</span>
              <span className="data-mono" style={{ color: '#0ea5e9' }}>{hwStats.cpu.toFixed(0)}%</span>
            </div>
            <div className="label-tech" style={{ fontSize: '12px', color: '#fff', marginBottom: '8px' }}>AMD Ryzen 9</div>
            <Progress percent={hwStats.cpu} showInfo={false} strokeColor="#0ea5e9" trailColor="rgba(15, 23, 42, 0.6)" strokeWidth={4} />
          </div>

          <Divider style={{ borderColor: 'rgba(0, 242, 255, 0.1)' }} />

          <div style={{ background: 'rgba(15, 23, 42, 0.4)', padding: '16px', borderRadius: '8px' }}>
            <div className="label-tech" style={{ marginBottom: '8px' }}>配置进度</div>
            <Progress percent={100} strokeColor="#00f2ff" />
          </div>
        </Card>
      </Col>
    </Row>
  );
}

// --- Main App ---

export default function App() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeMenu, setActiveMenu] = useState('dashboard');
  const [monitorInput, setMonitorInput] = useState('');
  const [lastResult, setLastResult] = useState<EmotionMonitorResult | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([
    { id: '1', time: new Date().toLocaleTimeString(), event: '安全防御系统已初始化', type: 'info' },
  ]);

  const refreshStatus = useCallback(async () => {
    try {
      const data = await fetchStatus();
      setStatus(data);
    } catch (err) {
      console.error('获取系统状态失败', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshStatus();
    const interval = setInterval(refreshStatus, 3000);
    return () => clearInterval(interval);
  }, [refreshStatus]);

  const addLog = (event: string, type: LogEntry['type'] = 'info') => {
    setLogs(prev => [
      { id: Date.now().toString(), time: new Date().toLocaleTimeString(), event, type },
      ...prev.slice(0, 19)
    ]);
  };

  const handleMonitor = async () => {
    if (!monitorInput.trim()) return;
    addLog(`正在扫描输入文本: "${monitorInput.substring(0, 20)}..."`, 'info');
    try {
      const result = await sendMonitor(monitorInput);
      setLastResult(result);
      if (!result.allowed) {
        addLog(`警报: 检测到高风险情感波动! 风险等级: ${result.riskAssessment.level}`, 'error');
      } else {
        addLog(`扫描完成。检测通过，风险评分: ${result.riskAssessment.score}`, 'success');
      }
      setMonitorInput('');
    } catch (err) {
      addLog('扫描失败: 连接超时或异常', 'error');
    }
  };

  const handleSwitch = async () => {
    addLog('正在转换至监控阶段...', 'info');
    await sendSwitch();
    refreshStatus();
  };

  const handleReset = async () => {
    addLog('接收到系统重置请求。', 'warning');
    await sendReset();
    setLastResult(null);
    refreshStatus();
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: '#020617' }}>
        <ThunderboltOutlined spin style={{ fontSize: '48px', color: '#00f2ff' }} />
      </div>
    );
  }

  return (
    <ConfigProvider
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: '#00f2ff',
          borderRadius: 4,
          fontFamily: 'Space Grotesk, "Microsoft YaHei", sans-serif',
          colorBgBase: '#020617',
          colorBgContainer: 'rgba(15, 23, 42, 0.4)',
        },
      }}
    >
      <Layout style={{ height: '100vh', background: 'transparent' }}>
        <div className="scanline-effect" />

        {/* Sidebar */}
        <Sider width={280} className="sidebar" style={{ background: 'rgba(2, 6, 23, 0.8)', borderRight: '1px solid rgba(0, 242, 255, 0.1)' }}>
          <div style={{ padding: '32px 24px' }}>
            <div className="heading-cyber" style={{ fontSize: '24px', marginBottom: '40px', fontStyle: 'italic' }}>
              神盾系统_V1.0
            </div>

            <div className="glass-panel" style={{ padding: '16px', marginBottom: '32px', display: 'flex', alignItems: 'center', gap: '12px' }}>
              <Avatar shape="square" size="large" icon={<UserOutlined />} style={{ background: 'rgba(0, 242, 255, 0.1)', color: '#00f2ff' }} />
              <div>
                <div className="label-tech" style={{ fontSize: '10px' }}>操作员_01</div>
                <div style={{ color: '#00f2ff', fontSize: '12px', fontWeight: 'bold' }}>模型状态: 安全运行</div>
              </div>
            </div>

            <Menu
              mode="inline"
              selectedKeys={[activeMenu]}
              onClick={(e) => setActiveMenu(e.key)}
              style={{ background: 'transparent', border: 'none' }}
              items={[
                { key: 'dashboard', icon: <DashboardOutlined />, label: '实时监控面板' },
                { key: 'config', icon: <SettingOutlined />, label: '模型参数配置' },
                { key: 'stats', icon: <GlobalOutlined />, label: '网络状态指标' },
              ]}
            />
          </div>

          <div style={{ marginTop: 'auto', padding: '24px' }}>
            <button className="neon-button" style={{ width: '100%' }} onClick={() => addLog('全局深度扫描已启动...')}>
              初始化扫描
            </button>
          </div>
        </Sider>

        <Layout style={{ background: 'transparent' }}>
          {/* Header */}
          <Header style={{ background: 'transparent', height: '80px', padding: '0 40px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <StatusBadge phase={status?.phase || 'learning'} />
            <div className="data-mono" style={{ color: '#94a3b8' }}>
              系统时间: {new Date().toISOString().split('T')[1].split('.')[0]}Z
            </div>
          </Header>

          {/* Content */}
          <Content style={{ padding: '0 40px 40px', overflowY: 'auto' }}>
            {activeMenu === 'dashboard' ? (
              <DashboardView
                status={status}
                lastResult={lastResult}
                logs={logs}
                monitorInput={monitorInput}
                setMonitorInput={setMonitorInput}
                handleMonitor={handleMonitor}
                handleReset={handleReset}
                handleSwitch={handleSwitch}
              />
            ) : activeMenu === 'config' ? (
              <ConfigView />
            ) : (
              <div style={{ textAlign: 'center', padding: '100px' }}>
                <Title level={3} style={{ color: '#00f2ff' }}>模块开发中...</Title>
              </div>
            )}
          </Content>
        </Layout>
      </Layout>

      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .ant-menu-item-selected {
          background: rgba(0, 242, 255, 0.1) !important;
          color: #00f2ff !important;
        }
        .ant-card-head {
          border-bottom: 1px solid rgba(0, 242, 255, 0.1) !important;
          min-height: 48px !important;
        }
        .ant-layout-sider-children {
          display: flex;
          flex-direction: column;
        }
        .ant-progress-text {
          color: #00f2ff !important;
        }
        .cyber-input {
          background: rgba(15, 23, 42, 0.6) !important;
          border: 1px solid rgba(0, 242, 255, 0.2) !important;
          color: #fff !important;
        }
        .cyber-select .ant-select-selector {
          background: rgba(15, 23, 42, 0.6) !important;
          border: 1px solid rgba(0, 242, 255, 0.2) !important;
          color: #fff !important;
        }
        .neon-button-ghost {
          border-color: #00f2ff !important;
          color: #00f2ff !important;
        }
        .neon-button-ghost:hover {
          background: rgba(0, 242, 255, 0.1) !important;
          box-shadow: 0 0 15px rgba(0, 242, 255, 0.2);
        }
        .ant-form-item-label label {
          color: #94a3b8 !important;
        }
      `}} />
    </ConfigProvider>
  );
}

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}
