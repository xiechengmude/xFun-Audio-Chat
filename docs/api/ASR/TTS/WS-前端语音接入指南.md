# xDAN-Audio-Chat 前端接入指南

> 版本: v1.0 | 更新时间: 2026-01-08

本文档提供三个核心接口的前端接入方式，包括 ASR（语音识别）、S2S（语音对话）、TTS（语音合成）。

---

## 目录

1. [服务端点概览](#1-服务端点概览)
2. [ASR 语音识别接口](#2-asr-语音识别接口)
3. [S2S 语音对话接口](#3-s2s-语音对话接口)
4. [TTS 语音合成接口](#4-tts-语音合成接口)
5. [通用工具函数](#5-通用工具函数)
6. [错误处理](#6-错误处理)
7. [完整示例项目](#7-完整示例项目)

---

## 1. 服务端点概览

| 服务 | 协议 | 端口 | 端点 | 用途 |
|------|------|------|------|------|
| ASR | HTTP | 8003 | `/api/transcribe` | 语音转文字 |
| S2S | WebSocket | 8002 | `/api/chat` | 实时语音对话 |
| TTS | HTTP | 8004 | `/api/synthesize` | 文字转语音 |

---

## 2. ASR 语音识别接口

### 2.1 接口说明

| 项目 | 说明 |
|------|------|
| 端点 | `POST /api/transcribe` |
| Content-Type | `multipart/form-data` |
| 支持格式 | WAV, MP3, FLAC, OGG 等 |
| 支持语言 | 中文、英文、日文 |

### 2.2 请求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| audio | File | 是 | 音频文件 |
| language | string | 否 | 识别语言，默认 "中文" |
| hotwords | string | 否 | 热词列表，逗号分隔 |

### 2.3 响应格式

```json
{
  "text": "识别出的文本内容",
  "language": "中文",
  "success": true
}
```

### 2.4 JavaScript/TypeScript 实现

```typescript
// types.ts
interface ASRResponse {
  text: string;
  language: string;
  success: boolean;
  error?: string;
}

interface ASROptions {
  language?: '中文' | '英文' | '日文';
  hotwords?: string[];
}

// asr-client.ts
class ASRClient {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:8003') {
    this.baseUrl = baseUrl;
  }

  /**
   * 转写音频文件
   */
  async transcribe(
    audioFile: File | Blob,
    options: ASROptions = {}
  ): Promise<ASRResponse> {
    const formData = new FormData();
    formData.append('audio', audioFile);

    if (options.language) {
      formData.append('language', options.language);
    }

    if (options.hotwords && options.hotwords.length > 0) {
      formData.append('hotwords', options.hotwords.join(','));
    }

    const response = await fetch(`${this.baseUrl}/api/transcribe`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`ASR request failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * 从 MediaRecorder 录制的 Blob 转写
   */
  async transcribeBlob(
    blob: Blob,
    options: ASROptions = {}
  ): Promise<ASRResponse> {
    // 转换为 File 对象
    const file = new File([blob], 'recording.wav', { type: blob.type });
    return this.transcribe(file, options);
  }

  /**
   * 健康检查
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      const data = await response.json();
      return data.status === 'healthy';
    } catch {
      return false;
    }
  }
}

export { ASRClient, ASRResponse, ASROptions };
```

### 2.5 React Hook 示例

```tsx
// useASR.ts
import { useState, useCallback } from 'react';
import { ASRClient, ASRResponse, ASROptions } from './asr-client';

const asrClient = new ASRClient('http://your-server:8003');

export function useASR() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ASRResponse | null>(null);

  const transcribe = useCallback(async (
    audio: File | Blob,
    options?: ASROptions
  ) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await asrClient.transcribe(
        audio instanceof Blob ? new File([audio], 'audio.wav') : audio,
        options
      );
      setResult(response);
      return response;
    } catch (err) {
      const message = err instanceof Error ? err.message : '识别失败';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { transcribe, isLoading, error, result };
}

// 使用示例
function VoiceInput() {
  const { transcribe, isLoading, result } = useASR();
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;
    chunksRef.current = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        chunksRef.current.push(e.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: 'audio/wav' });
      await transcribe(blob, { language: '中文' });
      stream.getTracks().forEach(track => track.stop());
    };

    mediaRecorder.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  };

  return (
    <div>
      <button
        onClick={recording ? stopRecording : startRecording}
        disabled={isLoading}
      >
        {recording ? '停止录音' : '开始录音'}
      </button>
      {isLoading && <p>识别中...</p>}
      {result && <p>识别结果: {result.text}</p>}
    </div>
  );
}
```

### 2.6 cURL 测试命令

```bash
# 基础调用
curl -X POST http://localhost:8003/api/transcribe \
  -F 'audio=@audio.wav' \
  -F 'language=中文'

# 带热词
curl -X POST http://localhost:8003/api/transcribe \
  -F 'audio=@audio.wav' \
  -F 'language=中文' \
  -F 'hotwords=人工智能,机器学习,深度学习'

# 健康检查
curl http://localhost:8003/health
```

---

## 3. S2S 语音对话接口

### 3.1 接口说明

| 项目 | 说明 |
|------|------|
| 端点 | `ws://host:8002/api/chat` |
| 协议 | WebSocket (Binary) |
| 音频格式 | Opus 编码, 24kHz |
| 特性 | 双向实时通信 |

### 3.2 消息协议

#### 消息类型 (第一个字节)

| 类型码 | 名称 | 方向 | 说明 |
|--------|------|------|------|
| 0x00 | HANDSHAKE | S→C | 握手响应 |
| 0x01 | AUDIO | 双向 | 音频数据 (Opus) |
| 0x02 | TEXT | S→C | 文本内容 |
| 0x03 | CONTROL | C→S | 控制信号 |
| 0x04 | METADATA | C→S | 元数据 (如自定义 system prompt) |
| 0x05 | ERROR | S→C | 错误信息 |
| 0x06 | PING | C→S | 心跳 |

#### 控制信号 (0x03 后的第二个字节)

| 控制码 | 名称 | 说明 |
|--------|------|------|
| 0x00 | START | 开始新一轮对话 |
| 0x01 | END_TURN | 结束当前轮次 |
| 0x02 | PAUSE | 暂停录音 |
| 0x03 | RESTART | 重新开始 |

### 3.3 通信流程

```
客户端                                     服务端
   |                                          |
   |  --------- WebSocket 连接 ---------->    |
   |  <------- 0x00 握手响应 -------------    |
   |                                          |
   |  --------- 0x03 0x00 (START) -------->   |
   |  --------- 0x01 [Opus音频] ---------->   |
   |  --------- 0x01 [Opus音频] ---------->   |
   |  ...                                     |
   |  --------- 0x03 0x02 (PAUSE) -------->   |
   |                                          |
   |  <------- 0x02 [处理中...] -----------   |
   |  <------- 0x01 [Opus音频] ------------   |
   |  <------- 0x02 [回复文本] ------------   |
   |  <------- 0x01 [Opus音频] ------------   |
   |  ...                                     |
```

### 3.4 TypeScript 实现

```typescript
// types.ts
interface S2SMessage {
  type: 'handshake' | 'audio' | 'text' | 'control' | 'error';
  data?: ArrayBuffer | string;
  action?: string;
}

interface S2SOptions {
  onAudio?: (audioData: ArrayBuffer) => void;
  onText?: (text: string) => void;
  onError?: (error: string) => void;
  onStateChange?: (state: ConnectionState) => void;
  systemPrompt?: string;
}

type ConnectionState = 'connecting' | 'connected' | 'recording' | 'processing' | 'responding' | 'disconnected';

// s2s-client.ts
class S2SClient {
  private ws: WebSocket | null = null;
  private audioContext: AudioContext | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private opusEncoder: any = null; // 需要 opus 编码库
  private state: ConnectionState = 'disconnected';
  private options: S2SOptions;
  private wsUrl: string;

  constructor(wsUrl: string, options: S2SOptions = {}) {
    this.wsUrl = wsUrl;
    this.options = options;
  }

  /**
   * 连接到服务器
   */
  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.setState('connecting');

      this.ws = new WebSocket(this.wsUrl);
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = () => {
        console.log('WebSocket connected');
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
        if (this.state === 'connecting') {
          this.setState('connected');
          resolve();
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };

      this.ws.onclose = () => {
        this.setState('disconnected');
      };

      // 连接超时
      setTimeout(() => {
        if (this.state === 'connecting') {
          reject(new Error('Connection timeout'));
        }
      }, 10000);
    });
  }

  /**
   * 处理收到的消息
   */
  private handleMessage(data: ArrayBuffer): void {
    const view = new Uint8Array(data);
    const msgType = view[0];
    const payload = view.slice(1);

    switch (msgType) {
      case 0x00: // HANDSHAKE
        console.log('Received handshake');
        break;

      case 0x01: // AUDIO
        this.setState('responding');
        this.options.onAudio?.(payload.buffer);
        break;

      case 0x02: // TEXT
        const text = new TextDecoder().decode(payload);
        this.options.onText?.(text);
        break;

      case 0x05: // ERROR
        const errorMsg = new TextDecoder().decode(payload);
        this.options.onError?.(errorMsg);
        break;
    }
  }

  /**
   * 发送控制信号
   */
  private sendControl(action: number): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const msg = new Uint8Array([0x03, action]);
      this.ws.send(msg);
    }
  }

  /**
   * 发送元数据 (如自定义 system prompt)
   */
  sendMetadata(metadata: object): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const jsonStr = JSON.stringify(metadata);
      const encoder = new TextEncoder();
      const payload = encoder.encode(jsonStr);
      const msg = new Uint8Array(1 + payload.length);
      msg[0] = 0x04; // METADATA
      msg.set(payload, 1);
      this.ws.send(msg);
    }
  }

  /**
   * 设置自定义 system prompt
   */
  setSystemPrompt(prompt: string): void {
    this.sendMetadata({ system_prompt: prompt });
  }

  /**
   * 开始新一轮对话
   */
  async startTurn(): Promise<void> {
    this.sendControl(0x00); // START
    this.setState('recording');
    await this.startRecording();
  }

  /**
   * 结束当前轮次
   */
  async endTurn(): Promise<void> {
    await this.stopRecording();
    this.sendControl(0x02); // PAUSE
    this.setState('processing');
  }

  /**
   * 开始录音
   */
  private async startRecording(): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 24000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      }
    });

    this.audioContext = new AudioContext({ sampleRate: 24000 });
    const source = this.audioContext.createMediaStreamSource(stream);

    // 使用 ScriptProcessorNode 或 AudioWorklet 获取 PCM 数据
    const processor = this.audioContext.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (e) => {
      if (this.state === 'recording') {
        const pcmData = e.inputBuffer.getChannelData(0);
        this.sendAudio(pcmData);
      }
    };

    source.connect(processor);
    processor.connect(this.audioContext.destination);
  }

  /**
   * 发送音频数据
   * 注意: 实际使用需要 Opus 编码
   */
  private sendAudio(pcmData: Float32Array): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      // 这里需要将 PCM 编码为 Opus
      // 使用 opus-recorder 或 libopus.js 等库
      const opusData = this.encodeOpus(pcmData);

      const msg = new Uint8Array(1 + opusData.length);
      msg[0] = 0x01; // AUDIO
      msg.set(opusData, 1);
      this.ws.send(msg);
    }
  }

  /**
   * Opus 编码 (需要实际的编码库)
   */
  private encodeOpus(pcmData: Float32Array): Uint8Array {
    // 使用 opus-recorder 或 libopus.js
    // 这里返回模拟数据
    const int16Data = new Int16Array(pcmData.length);
    for (let i = 0; i < pcmData.length; i++) {
      int16Data[i] = Math.max(-32768, Math.min(32767, pcmData[i] * 32768));
    }
    return new Uint8Array(int16Data.buffer);
  }

  /**
   * 停止录音
   */
  private async stopRecording(): Promise<void> {
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }
  }

  /**
   * 播放收到的音频
   */
  async playAudio(opusData: ArrayBuffer): Promise<void> {
    // 解码 Opus 并播放
    // 使用 opus-decoder 或 Web Audio API
    const audioContext = new AudioContext({ sampleRate: 24000 });

    // 需要先解码 Opus 到 PCM
    const pcmData = this.decodeOpus(opusData);

    const buffer = audioContext.createBuffer(1, pcmData.length, 24000);
    buffer.getChannelData(0).set(pcmData);

    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.start();
  }

  /**
   * Opus 解码 (需要实际的解码库)
   */
  private decodeOpus(opusData: ArrayBuffer): Float32Array {
    // 使用 opus-decoder 库
    // 这里返回模拟数据
    return new Float32Array(opusData.byteLength / 2);
  }

  /**
   * 设置状态
   */
  private setState(state: ConnectionState): void {
    this.state = state;
    this.options.onStateChange?.(state);
  }

  /**
   * 断开连接
   */
  disconnect(): void {
    this.stopRecording();
    this.ws?.close();
    this.setState('disconnected');
  }

  /**
   * 获取当前状态
   */
  getState(): ConnectionState {
    return this.state;
  }
}

export { S2SClient, S2SOptions, ConnectionState };
```

### 3.5 React 组件示例

```tsx
// VoiceChat.tsx
import { useState, useRef, useEffect, useCallback } from 'react';
import { S2SClient, ConnectionState } from './s2s-client';

interface Props {
  serverUrl: string;
  systemPrompt?: string;
}

export function VoiceChat({ serverUrl, systemPrompt }: Props) {
  const [state, setState] = useState<ConnectionState>('disconnected');
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([]);
  const [isRecording, setIsRecording] = useState(false);
  const clientRef = useRef<S2SClient | null>(null);
  const audioQueueRef = useRef<ArrayBuffer[]>([]);

  // 初始化客户端
  useEffect(() => {
    clientRef.current = new S2SClient(serverUrl, {
      onAudio: (data) => {
        audioQueueRef.current.push(data);
        playNextAudio();
      },
      onText: (text) => {
        // 过滤掉状态消息
        if (!text.startsWith('[')) {
          setMessages(prev => {
            const last = prev[prev.length - 1];
            if (last?.role === 'assistant') {
              return [...prev.slice(0, -1), { role: 'assistant', content: last.content + text }];
            }
            return [...prev, { role: 'assistant', content: text }];
          });
        }
      },
      onError: (error) => {
        console.error('S2S Error:', error);
      },
      onStateChange: (newState) => {
        setState(newState);
      },
    });

    return () => {
      clientRef.current?.disconnect();
    };
  }, [serverUrl]);

  // 播放音频队列
  const playNextAudio = useCallback(async () => {
    if (audioQueueRef.current.length > 0) {
      const data = audioQueueRef.current.shift()!;
      await clientRef.current?.playAudio(data);
      playNextAudio();
    }
  }, []);

  // 连接服务器
  const connect = async () => {
    try {
      await clientRef.current?.connect();
      if (systemPrompt) {
        clientRef.current?.setSystemPrompt(systemPrompt);
      }
    } catch (error) {
      console.error('Connection failed:', error);
    }
  };

  // 开始录音
  const startRecording = async () => {
    setIsRecording(true);
    setMessages(prev => [...prev, { role: 'user', content: '[录音中...]' }]);
    await clientRef.current?.startTurn();
  };

  // 停止录音
  const stopRecording = async () => {
    setIsRecording(false);
    setMessages(prev => {
      const last = prev[prev.length - 1];
      if (last?.content === '[录音中...]') {
        return [...prev.slice(0, -1), { role: 'user', content: '[语音输入]' }];
      }
      return prev;
    });
    await clientRef.current?.endTurn();
  };

  // 断开连接
  const disconnect = () => {
    clientRef.current?.disconnect();
    setMessages([]);
  };

  return (
    <div className="voice-chat">
      <div className="status">
        状态: {state}
      </div>

      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <strong>{msg.role === 'user' ? '你' : 'AI'}:</strong>
            <span>{msg.content}</span>
          </div>
        ))}
      </div>

      <div className="controls">
        {state === 'disconnected' ? (
          <button onClick={connect}>连接</button>
        ) : (
          <>
            <button
              onMouseDown={startRecording}
              onMouseUp={stopRecording}
              onTouchStart={startRecording}
              onTouchEnd={stopRecording}
              disabled={state === 'processing' || state === 'responding'}
            >
              {isRecording ? '松开发送' : '按住说话'}
            </button>
            <button onClick={disconnect}>断开</button>
          </>
        )}
      </div>
    </div>
  );
}
```

### 3.6 推荐的 Opus 编解码库

```bash
# 安装 opus-recorder (编码)
npm install opus-recorder

# 安装 opus-decoder (解码)
npm install @piotr-niciak/opus-decoder

# 或使用 sphn (项目自带)
# sphn 是一个 Python/WASM 库，需要单独集成
```

---

## 4. TTS 语音合成接口

### 4.1 接口说明

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/synthesize` | POST | 返回 Base64 音频 |
| `/api/synthesize/file` | POST | 返回 WAV 文件 |
| `/health` | GET | 健康检查 |
| `/api/info` | GET | 模型信息 |

### 4.2 请求参数

```json
{
  "text": "要合成的文本",
  "speaker_id": "中文女",
  "prompt_wav": "/path/to/ref.wav",    // 可选: 参考音频路径
  "prompt_text": "参考音频对应的文本",   // 可选: 参考音频文本
  "stream": false                       // 是否流式返回
}
```

### 4.3 响应格式

**非流式响应:**
```json
{
  "audio": "<base64_wav_data>",
  "sample_rate": 24000,
  "duration": 3.5,
  "success": true
}
```

### 4.4 TypeScript 实现

```typescript
// types.ts
interface TTSResponse {
  audio: string;  // Base64 编码的 WAV
  sample_rate: number;
  duration: number;
  success: boolean;
  error?: string;
}

interface TTSOptions {
  speakerId?: string;
  promptWav?: string;
  promptText?: string;
  stream?: boolean;
}

// tts-client.ts
class TTSClient {
  private baseUrl: string;
  private audioContext: AudioContext | null = null;

  constructor(baseUrl: string = 'http://localhost:8004') {
    this.baseUrl = baseUrl;
  }

  /**
   * 合成语音 (返回 Base64)
   */
  async synthesize(text: string, options: TTSOptions = {}): Promise<TTSResponse> {
    const response = await fetch(`${this.baseUrl}/api/synthesize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        speaker_id: options.speakerId || '中文女',
        prompt_wav: options.promptWav,
        prompt_text: options.promptText,
        stream: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`TTS request failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * 合成语音并直接获取音频 Blob
   */
  async synthesizeToBlob(text: string, options: TTSOptions = {}): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/synthesize/file`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        speaker_id: options.speakerId || '中文女',
        prompt_wav: options.promptWav,
        prompt_text: options.promptText,
      }),
    });

    if (!response.ok) {
      throw new Error(`TTS request failed: ${response.statusText}`);
    }

    return response.blob();
  }

  /**
   * 合成并播放
   */
  async synthesizeAndPlay(text: string, options: TTSOptions = {}): Promise<void> {
    const result = await this.synthesize(text, options);
    await this.playBase64Audio(result.audio);
  }

  /**
   * 播放 Base64 音频
   */
  async playBase64Audio(base64Audio: string): Promise<void> {
    // 解码 Base64
    const binaryString = atob(base64Audio);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    // 创建 AudioContext
    if (!this.audioContext) {
      this.audioContext = new AudioContext();
    }

    // 解码音频
    const audioBuffer = await this.audioContext.decodeAudioData(bytes.buffer);

    // 播放
    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);
    source.start();

    // 返回 Promise，等待播放完成
    return new Promise((resolve) => {
      source.onended = () => resolve();
    });
  }

  /**
   * 下载音频文件
   */
  async downloadAudio(text: string, filename: string = 'speech.wav', options: TTSOptions = {}): Promise<void> {
    const blob = await this.synthesizeToBlob(text, options);

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  /**
   * 获取可用的发音人列表
   */
  async getSpeakers(): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/api/info`);
    const data = await response.json();
    return data.available_speakers || [];
  }

  /**
   * 健康检查
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      const data = await response.json();
      return data.status === 'healthy';
    } catch {
      return false;
    }
  }
}

export { TTSClient, TTSResponse, TTSOptions };
```

### 4.5 React Hook 示例

```tsx
// useTTS.ts
import { useState, useCallback, useRef } from 'react';
import { TTSClient, TTSOptions } from './tts-client';

const ttsClient = new TTSClient('http://your-server:8004');

export function useTTS() {
  const [isLoading, setIsLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const speak = useCallback(async (text: string, options?: TTSOptions) => {
    setIsLoading(true);
    setError(null);

    try {
      setIsPlaying(true);
      await ttsClient.synthesizeAndPlay(text, options);
    } catch (err) {
      const message = err instanceof Error ? err.message : '合成失败';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
      setIsPlaying(false);
    }
  }, []);

  const download = useCallback(async (text: string, filename?: string, options?: TTSOptions) => {
    setIsLoading(true);
    setError(null);

    try {
      await ttsClient.downloadAudio(text, filename, options);
    } catch (err) {
      const message = err instanceof Error ? err.message : '下载失败';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { speak, download, isLoading, isPlaying, error };
}

// 使用示例
function TextToSpeech() {
  const { speak, download, isLoading, isPlaying } = useTTS();
  const [text, setText] = useState('');

  return (
    <div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="输入要合成的文本..."
      />
      <div>
        <button
          onClick={() => speak(text)}
          disabled={isLoading || !text}
        >
          {isPlaying ? '播放中...' : '播放'}
        </button>
        <button
          onClick={() => download(text, 'speech.wav')}
          disabled={isLoading || !text}
        >
          下载
        </button>
      </div>
    </div>
  );
}
```

### 4.6 cURL 测试命令

```bash
# 合成并返回 Base64
curl -X POST http://localhost:8004/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，欢迎使用语音合成服务", "speaker_id": "中文女"}'

# 合成并下载 WAV 文件
curl -X POST http://localhost:8004/api/synthesize/file \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界"}' \
  -o output.wav

# 健康检查
curl http://localhost:8004/health

# 获取模型信息
curl http://localhost:8004/api/info
```

---

## 5. 通用工具函数

### 5.1 音频录制工具

```typescript
// audio-recorder.ts
export class AudioRecorder {
  private mediaRecorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private stream: MediaStream | null = null;

  async start(options: MediaRecorderOptions = {}): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      }
    });

    this.chunks = [];
    this.mediaRecorder = new MediaRecorder(this.stream, {
      mimeType: 'audio/webm;codecs=opus',
      ...options
    });

    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        this.chunks.push(e.data);
      }
    };

    this.mediaRecorder.start(100); // 每 100ms 一个 chunk
  }

  async stop(): Promise<Blob> {
    return new Promise((resolve) => {
      if (!this.mediaRecorder) {
        throw new Error('Recorder not started');
      }

      this.mediaRecorder.onstop = () => {
        const blob = new Blob(this.chunks, { type: 'audio/webm' });
        this.cleanup();
        resolve(blob);
      };

      this.mediaRecorder.stop();
    });
  }

  private cleanup(): void {
    this.stream?.getTracks().forEach(track => track.stop());
    this.stream = null;
    this.mediaRecorder = null;
    this.chunks = [];
  }

  isRecording(): boolean {
    return this.mediaRecorder?.state === 'recording';
  }
}
```

### 5.2 音频格式转换

```typescript
// audio-utils.ts

/**
 * 将 WebM 转换为 WAV (用于 ASR)
 */
export async function webmToWav(webmBlob: Blob): Promise<Blob> {
  const audioContext = new AudioContext();
  const arrayBuffer = await webmBlob.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  const wavBuffer = audioBufferToWav(audioBuffer);
  return new Blob([wavBuffer], { type: 'audio/wav' });
}

/**
 * AudioBuffer 转 WAV
 */
function audioBufferToWav(buffer: AudioBuffer): ArrayBuffer {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;

  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;

  const dataLength = buffer.length * blockAlign;
  const headerLength = 44;
  const totalLength = headerLength + dataLength;

  const arrayBuffer = new ArrayBuffer(totalLength);
  const view = new DataView(arrayBuffer);

  // WAV header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, totalLength - 8, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataLength, true);

  // Audio data
  const channelData = buffer.getChannelData(0);
  let offset = 44;
  for (let i = 0; i < buffer.length; i++) {
    const sample = Math.max(-1, Math.min(1, channelData[i]));
    view.setInt16(offset, sample * 0x7FFF, true);
    offset += 2;
  }

  return arrayBuffer;
}

function writeString(view: DataView, offset: number, string: string): void {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}
```

---

## 6. 错误处理

### 6.1 通用错误类型

```typescript
// errors.ts
export class AudioServiceError extends Error {
  constructor(
    message: string,
    public code: string,
    public service: 'ASR' | 'S2S' | 'TTS'
  ) {
    super(message);
    this.name = 'AudioServiceError';
  }
}

export const ErrorCodes = {
  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT: 'TIMEOUT',
  INVALID_AUDIO: 'INVALID_AUDIO',
  MODEL_NOT_READY: 'MODEL_NOT_READY',
  PERMISSION_DENIED: 'PERMISSION_DENIED',
  UNSUPPORTED_FORMAT: 'UNSUPPORTED_FORMAT',
} as const;
```

### 6.2 错误处理示例

```typescript
// error-handler.ts
export async function withErrorHandling<T>(
  fn: () => Promise<T>,
  options: {
    service: 'ASR' | 'S2S' | 'TTS';
    retries?: number;
    onRetry?: (attempt: number, error: Error) => void;
  }
): Promise<T> {
  const { service, retries = 3, onRetry } = options;

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      const isLastAttempt = attempt === retries;

      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new AudioServiceError(
          '网络连接失败',
          ErrorCodes.NETWORK_ERROR,
          service
        );
      }

      if (!isLastAttempt) {
        onRetry?.(attempt, error as Error);
        await new Promise(r => setTimeout(r, 1000 * attempt));
        continue;
      }

      throw error;
    }
  }

  throw new Error('Unexpected error');
}
```

---

## 7. 完整示例项目

### 7.1 项目结构

```
src/
├── clients/
│   ├── asr-client.ts
│   ├── s2s-client.ts
│   └── tts-client.ts
├── hooks/
│   ├── useASR.ts
│   ├── useS2S.ts
│   └── useTTS.ts
├── utils/
│   ├── audio-recorder.ts
│   ├── audio-utils.ts
│   └── errors.ts
├── components/
│   ├── VoiceInput.tsx
│   ├── VoiceChat.tsx
│   └── TextToSpeech.tsx
└── App.tsx
```

### 7.2 环境配置

```typescript
// config.ts
export const config = {
  ASR_URL: process.env.REACT_APP_ASR_URL || 'http://localhost:8003',
  S2S_URL: process.env.REACT_APP_S2S_URL || 'ws://localhost:8002',
  TTS_URL: process.env.REACT_APP_TTS_URL || 'http://localhost:8004',
};
```

### 7.3 package.json 依赖

```json
{
  "dependencies": {
    "opus-recorder": "^8.0.5",
    "@piotr-niciak/opus-decoder": "^1.0.0"
  }
}
```

---

## 附录: 服务端点快速参考

```
# ASR 语音识别
POST http://host:8003/api/transcribe  (multipart/form-data)
GET  http://host:8003/health
GET  http://host:8003/api/info

# S2S 语音对话
WS   ws://host:8002/api/chat          (Binary WebSocket)

# TTS 语音合成
POST http://host:8004/api/synthesize       (JSON → Base64 音频)
POST http://host:8004/api/synthesize/file  (JSON → WAV 文件)
GET  http://host:8004/health
GET  http://host:8004/api/info
```

---

*文档版本: v1.0 | 最后更新: 2026-01-08*
