/**
 * DatasetLoader - 数据集加载器
 *
 * 支持多种数据源：
 * 1. HuggingFace 数据集
 * 2. 本地文件（json, jsonl, csv, txt）
 * 3. 本地文件夹（批量扫描）
 */

import fs from 'fs';
import path from 'path';

export interface DatasetSource {
  type: 'huggingface' | 'local-file' | 'local-directory';
  path: string;
}

export interface DatasetConfig {
  split?: string;
  format?: 'text' | 'json' | 'csv';
  textColumn?: string;
  limit?: number;
}

export interface LoadedDataset {
  texts: string[];
  source: DatasetSource;
  metadata: {
    count: number;
    format: string;
    loadedAt: number;
  };
}

/**
 * 数据集加载器
 */
export class DatasetLoader {
  private tempDir: string;

  constructor(tempDir: string = '/tmp/emotion-monitor-datasets') {
    this.tempDir = tempDir;
    this.ensureTempDir();
  }

  private ensureTempDir(): void {
    if (!fs.existsSync(this.tempDir)) {
      fs.mkdirSync(this.tempDir, { recursive: true });
    }
  }

  /**
   * 从 HuggingFace 加载数据集
   */
  async loadFromHuggingFace(datasetId: string, config: DatasetConfig = {}): Promise<LoadedDataset> {
    console.log(`Loading dataset from HuggingFace: ${datasetId}`);

    let datasets: any;
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      datasets = require('datasets');
    } catch {
      throw new Error('Please install datasets: npm install datasets');
    }

    const { loadDataset } = datasets;
    const split = config.split || 'train';
    const limit = config.limit || Infinity;

    const dataset = await loadDataset(datasetId, { split });
    const textColumn = config.textColumn || this.detectTextColumn(dataset);

    const texts: string[] = [];
    const rows = dataset[textColumn] || [];

    for (let i = 0; i < Math.min(rows.length, limit); i++) {
      const text = rows[i];
      if (typeof text === 'string') {
        texts.push(text);
      } else if (typeof text === 'object' && text !== null) {
        texts.push(JSON.stringify(text));
      }
    }

    return {
      texts,
      source: { type: 'huggingface', path: datasetId },
      metadata: {
        count: texts.length,
        format: 'huggingface',
        loadedAt: Date.now(),
      },
    };
  }

  /**
   * 从本地文件加载
   */
  async loadFromFile(filePath: string, config: DatasetConfig = {}): Promise<LoadedDataset> {
    console.log(`Loading dataset from file: ${filePath}`);

    if (!fs.existsSync(filePath)) {
      throw new Error(`File not found: ${filePath}`);
    }

    const ext = path.extname(filePath).toLowerCase();

    if (ext === '.json' || ext === '.jsonl') {
      return this.loadFromJson(filePath, config);
    } else if (ext === '.csv') {
      return this.loadFromCsv(filePath, config);
    } else if (ext === '.txt') {
      return this.loadFromTxt(filePath, config);
    } else {
      throw new Error(`Unsupported file format: ${ext}. Supported: .json, .jsonl, .csv, .txt`);
    }
  }

  /**
   * 从目录批量加载
   */
  async loadFromDirectory(dirPath: string, config: DatasetConfig = {}): Promise<LoadedDataset> {
    console.log(`Loading dataset from directory: ${dirPath}`);

    if (!fs.existsSync(dirPath)) {
      throw new Error(`Directory not found: ${dirPath}`);
    }

    const limit = config.limit || Infinity;
    const texts: string[] = [];

    const scanDir = async (dir: string) => {
      const entries = fs.readdirSync(dir, { withFileTypes: true });

      for (const entry of entries) {
        if (texts.length >= limit) break;

        const fullPath = path.join(dir, entry.name);

        if (entry.isDirectory()) {
          await scanDir(fullPath);
        } else if (entry.isFile()) {
          const ext = path.extname(entry.name).toLowerCase();
          if (['.txt', '.json', '.jsonl', '.csv'].includes(ext)) {
            try {
              const fileTexts = await this.loadFileContent(fullPath, ext, config);
              texts.push(...fileTexts);
            } catch (err) {
              console.warn(`Failed to load file ${fullPath}:`, err);
            }
          }
        }
      }
    };

    await scanDir(dirPath);

    return {
      texts,
      source: { type: 'local-directory', path: dirPath },
      metadata: {
        count: texts.length,
        format: 'directory',
        loadedAt: Date.now(),
      },
    };
  }

  /**
   * 从 JSON 文件加载
   */
  private async loadFromJson(filePath: string, config: DatasetConfig): Promise<LoadedDataset> {
    const content = fs.readFileSync(filePath, 'utf-8');
    const texts = await this.parseContent(content, '.json', config);

    return {
      texts,
      source: { type: 'local-file', path: filePath },
      metadata: {
        count: texts.length,
        format: 'json',
        loadedAt: Date.now(),
      },
    };
  }

  /**
   * 从 CSV 文件加载
   */
  private async loadFromCsv(filePath: string, config: DatasetConfig): Promise<LoadedDataset> {
    const content = fs.readFileSync(filePath, 'utf-8');
    const texts = await this.parseContent(content, '.csv', config);

    return {
      texts,
      source: { type: 'local-file', path: filePath },
      metadata: {
        count: texts.length,
        format: 'csv',
        loadedAt: Date.now(),
      },
    };
  }

  /**
   * 从 TXT 文件加载
   */
  private async loadFromTxt(filePath: string, config: DatasetConfig): Promise<LoadedDataset> {
    const content = fs.readFileSync(filePath, 'utf-8');
    const texts = await this.parseContent(content, '.txt', config);

    return {
      texts,
      source: { type: 'local-file', path: filePath },
      metadata: {
        count: texts.length,
        format: 'txt',
        loadedAt: Date.now(),
      },
    };
  }

  /**
   * 解析文件内容
   */
  private async loadFileContent(filePath: string, ext: string, config: DatasetConfig): Promise<string[]> {
    const content = fs.readFileSync(filePath, 'utf-8');
    return this.parseContent(content, ext, config);
  }

  /**
   * 解析内容为文本数组
   */
  private async parseContent(content: string, ext: string, config: DatasetConfig): Promise<string[]> {
    const limit = config.limit || Infinity;
    const texts: string[] = [];

    if (ext === '.txt') {
      const lines = content.split('\n').filter(l => l.trim());
      return lines.slice(0, limit);
    }

    if (ext === '.json') {
      try {
        const data = JSON.parse(content);
        const textColumn = config.textColumn || 'text';

        if (Array.isArray(data)) {
          for (const item of data) {
            if (texts.length >= limit) break;
            const text = this.extractText(item, textColumn);
            if (text) texts.push(text);
          }
        } else if (typeof data === 'object' && data !== null) {
          const text = this.extractText(data, textColumn);
          if (text) texts.push(text);
        }
      } catch {
        return this.parseContent(content, '.jsonl', config);
      }
      return texts;
    }

    if (ext === '.jsonl') {
      const lines = content.split('\n').filter(l => l.trim());
      const textColumn = config.textColumn || 'text';

      for (const line of lines) {
        if (texts.length >= limit) break;
        try {
          const obj = JSON.parse(line);
          const text = this.extractText(obj, textColumn);
          if (text) texts.push(text);
        } catch {
          // 忽略解析错误
        }
      }
      return texts;
    }

    if (ext === '.csv') {
      const lines = content.split('\n');
      if (lines.length < 2) return texts;

      const header = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
      const textColumn = config.textColumn || 'text';
      const textIndex = header.indexOf(textColumn);

      if (textIndex === -1) {
        for (let i = 1; i < lines.length && texts.length < limit; i++) {
          const cols = this.parseCSVLine(lines[i]);
          if (cols[0]) texts.push(cols[0]);
        }
      } else {
        for (let i = 1; i < lines.length && texts.length < limit; i++) {
          const cols = this.parseCSVLine(lines[i]);
          if (cols[textIndex]) texts.push(cols[textIndex]);
        }
      }
      return texts;
    }

    return texts;
  }

  /**
   * 从对象中提取文本
   */
  private extractText(obj: any, textColumn: string): string | null {
    if (typeof obj === 'string') return obj;
    if (typeof obj === 'object' && obj !== null) {
      if (obj[textColumn]) return String(obj[textColumn]);

      for (const key of ['text', 'content', 'sentence', 'document', 'input', 'query']) {
        if (obj[key]) return String(obj[key]);
      }

      for (const [key, value] of Object.entries(obj)) {
        if (typeof value === 'string' && value.trim()) {
          return value;
        }
      }
    }
    return null;
  }

  /**
   * 解析 CSV 行（处理引号）
   */
  private parseCSVLine(line: string): string[] {
    const result: string[] = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const char = line[i];

      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        result.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }

    result.push(current.trim());
    return result;
  }

  /**
   * 自动检测 JSON 中的文本列
   */
  private detectTextColumn(dataset: any): string {
    if (dataset.column_names) {
      const commonColumns = ['text', 'content', 'sentence', 'document', 'input'];
      for (const col of commonColumns) {
        if (dataset.column_names.includes(col)) return col;
      }
      return dataset.column_names[0];
    }
    return 'text';
  }

  /**
   * 下载 HuggingFace 数据集到本地
   */
  async downloadHFToLocal(datasetId: string, localPath: string): Promise<string> {
    console.log(`Downloading ${datasetId} to ${localPath}`);

    const loader = new DatasetLoader(path.dirname(localPath));
    const dataset = await this.loadFromHuggingFace(datasetId);

    if (!fs.existsSync(localPath)) {
      fs.mkdirSync(localPath, { recursive: true });
    }

    const outputFile = path.join(localPath, `${datasetId.replace('/', '_')}.json`);
    fs.writeFileSync(outputFile, JSON.stringify(dataset.texts, null, 2));

    return outputFile;
  }

  /**
   * 清理临时目录
   */
  cleanup(): void {
    if (fs.existsSync(this.tempDir)) {
      fs.rmSync(this.tempDir, { recursive: true, force: true });
    }
  }
}

// 默认实例
let defaultLoader: DatasetLoader | null = null;

export function getDatasetLoader(): DatasetLoader {
  if (!defaultLoader) {
    defaultLoader = new DatasetLoader();
  }
  return defaultLoader;
}
