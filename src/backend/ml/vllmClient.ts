/**
 * vLLM Client - vLLM 模型交互客户端
 *
 * 用于从 vLLM 服务器提取 hidden states
 */

export interface VLLMConfig {
  baseUrl: string;        // e.g., "http://localhost:8000"
  model: string;          // e.g., "meta-llama/Llama-3-8b"
  apiKey?: string;        // Optional API key
  timeout?: number;       // Request timeout in ms
  maxRetries?: number;    // Max retry attempts
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
}

export interface ChatCompletionResponse {
  id: string;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface HiddenStatesResult {
  hiddenStates: number[];  // 隐藏状态向量
  promptTokens: number;
  completionTokens: number;
  model: string;
}

export interface EmbeddingRequest {
  model: string;
  input: string | string[];
}

export interface EmbeddingResponse {
  model: string;
  embeddings: Array<{
    index: number;
    embedding: number[];
  }>;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

/**
 * vLLM 客户端
 */
export class VLLMClient {
  private config: Required<VLLMConfig>;
  private baseUrl: string;

  constructor(config: VLLMConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, '');
    this.config = {
      timeout: 30000,
      maxRetries: 3,
      apiKey: '',
      ...config,
    };
  }

  /**
   * 发送 chat completions 请求
   */
  async chat(messages: ChatMessage[], options?: {
    temperature?: number;
    maxTokens?: number;
  }): Promise<ChatCompletionResponse> {
    const request: ChatCompletionRequest = {
      model: this.config.model,
      messages,
      temperature: options?.temperature ?? 0.7,
      max_tokens: options?.maxTokens ?? 256,
    };

    return this.request<ChatCompletionRequest, ChatCompletionResponse>(
      '/v1/chat/completions',
      'POST',
      request
    );
  }

  /**
   * 获取文本的 embedding（如果 vLLM 支持）
   */
  async getEmbedding(text: string): Promise<number[]> {
    const request: EmbeddingRequest = {
      model: this.config.model,
      input: text,
    };

    const response = await this.request<EmbeddingRequest, EmbeddingResponse>(
      '/v1/embeddings',
      'POST',
      request
    );

    return response.embeddings[0]?.embedding ?? [];
  }

  /**
   * 获取 hidden states
   *
   * 注意：标准 vLLM API 不直接返回 hidden states
   * 需要使用以下方法之一：
   * 1. 使用 attention hook / logit processor（需修改 vLLM 服务端）
   * 2. 使用 embedding 接口作为代理（近似方案）
   * 3. 使用 Ollama 原生接口获取 hidden states
   *
   * 这里先返回 embedding 作为降维后的表示
   */
  async getHiddenStates(prompt: string): Promise<HiddenStatesResult> {
    // 尝试使用 embedding 接口
    // 在真实场景中，需要服务端支持返回中间层 hidden states
    const embedding = await this.getEmbedding(prompt);

    return {
      hiddenStates: embedding,
      promptTokens: Math.ceil(prompt.length / 4),  // 估算
      completionTokens: 0,
      model: this.config.model,
    };
  }

  /**
   * 获取隐藏状态（使用 chat 接口的近似方案）
   * 通过 embedding 层获取近似表示
   */
  async getHiddenStatesFromChat(input: string): Promise<HiddenStatesResult> {
    const messages: ChatMessage[] = [
      { role: 'user', content: input }
    ];

    // 获取 embedding
    const embedding = await this.getEmbedding(input);

    // 获取完整的 chat 响应（用于获取 token 统计）
    const chatResponse = await this.chat(messages, { maxTokens: 1 });

    return {
      hiddenStates: embedding,
      promptTokens: chatResponse.usage.prompt_tokens,
      completionTokens: chatResponse.usage.completion_tokens,
      model: this.config.model,
    };
  }

  /**
   * 批量获取 hidden states
   */
  async getHiddenStatesBatch(texts: string[]): Promise<HiddenStatesResult[]> {
    const results: HiddenStatesResult[] = [];

    for (const text of texts) {
      try {
        const result = await this.getHiddenStatesFromChat(text);
        results.push(result);
      } catch (error) {
        console.error(`Failed to get hidden states for text: ${text.substring(0, 50)}...`, error);
        // 使用零向量作为fallback
        results.push({
          hiddenStates: new Array(4096).fill(0),
          promptTokens: 0,
          completionTokens: 0,
          model: this.config.model,
        });
      }
    }

    return results;
  }

  /**
   * 健康检查
   */
  async healthCheck(): Promise<boolean> {
    try {
      await this.request<null, { model: string }>('/v1/models', 'GET', null);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * 获取可用模型列表
   */
  async listModels(): Promise<string[]> {
    try {
      const response = await this.request<null, { data: Array<{ id: string }> }>(
        '/v1/models',
        'GET',
        null
      );
      return response.data.map(m => m.id);
    } catch {
      return [];
    }
  }

  /**
   * 核心请求方法（带重试）
   */
  private async request<TRequest, TResponse>(
    endpoint: string,
    method: 'GET' | 'POST',
    body: TRequest | null,
    retryCount = 0
  ): Promise<TResponse> {
    const url = `${this.baseUrl}${endpoint}`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

      const response = await fetch(url, {
        method,
        headers,
        body: body !== null ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      return await response.json() as TResponse;
    } catch (error) {
      if (retryCount < this.config.maxRetries) {
        console.warn(`Request failed, retrying (${retryCount + 1}/${this.config.maxRetries})...`);
        return this.request(endpoint, method, body, retryCount + 1);
      }
      throw error;
    }
  }
}

// 默认客户端实例
let defaultClient: VLLMClient | null = null;

export function getVLLMClient(config?: VLLMConfig): VLLMClient {
  if (!defaultClient && config) {
    defaultClient = new VLLMClient(config);
  }
  if (!defaultClient) {
    // 返回一个使用默认配置的客户端（不会真正连接）
    defaultClient = new VLLMClient({
      baseUrl: 'http://localhost:8000',
      model: 'default',
    });
  }
  return defaultClient;
}

export function resetVLLMClient(): void {
  defaultClient = null;
}
