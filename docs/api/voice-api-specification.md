# Fun-Audio-Chat 接口规范

> 语音服务前后端接口文档 v1.0
> 适配 DeepAgent v0.21 架构

---

## 一、接口总览

### 1.1 服务架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Voice Service Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          Frontend (App/Web)                          │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │   │
│  │  │ VoiceRecorder │  │ AudioPlayer   │  │ VoiceSessionManager   │   │   │
│  │  │ (Opus Encoder)│  │ (Opus Decoder)│  │ (WebSocket Client)    │   │   │
│  │  └───────────────┘  └───────────────┘  └───────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    │ wss://                                 │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Voice Gateway Service                          │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │   │
│  │  │ /api/chat     │  │ /api/call     │  │ /api/session          │   │   │
│  │  │ (WebSocket)   │  │ (REST)        │  │ (REST)                │   │   │
│  │  └───────────────┘  └───────────────┘  └───────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    │ gRPC/HTTP                              │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Fun-Audio-Chat S2S Engine                         │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  Fun-Audio-Chat-8B (S2S) + CosyVoice3-0.5B (TTS)              │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    │ HTTP/gRPC                              │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DeepAgent Router (Fido v4.21)                     │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐    │   │
│  │  │ Router  │→│  Plan   │→│ Executor │→│ Memory (mem0/langmem)│    │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 端点清单

| 端点 | 协议 | 方法 | 说明 |
|------|------|------|------|
| `/api/chat` | WebSocket | - | 全双工语音对话 |
| `/api/session` | REST | POST/GET/DELETE | 会话管理 |
| `/api/call/initiate` | REST | POST | 发起主动呼叫 |
| `/api/call/accept` | REST | POST | 接听呼叫 |
| `/api/health` | REST | GET | 健康检查 |

---

## 二、WebSocket 接口 (`/api/chat`)

### 2.1 连接建立

**URL格式**:
```
wss://{host}:{port}/api/chat?user_id={user_id}&session_id={session_id}
```

**Query Parameters**:
| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `user_id` | 是 | string | 用户唯一标识 |
| `session_id` | 否 | string | 会话ID，不传则自动生成 |
| `trigger_context` | 否 | string | Base64编码的触发上下文JSON |

**连接示例**:
```javascript
const ws = new WebSocket(
  'wss://voice.vibe.finance/api/chat?user_id=u_123&session_id=s_456'
);

ws.onopen = () => {
  console.log('Voice session connected');
};
```

### 2.2 消息协议

#### 二进制消息格式

```
┌─────────┬──────────────────────────────────────┐
│ 1 byte  │           Variable length            │
├─────────┼──────────────────────────────────────┤
│  Type   │              Payload                 │
└─────────┴──────────────────────────────────────┘

Type:
  0x01 = Audio (Opus encoded)
  0x02 = Text (UTF-8 encoded)
  0x03 = Metadata (JSON)
```

#### 消息类型定义

**1. 音频消息 (0x01)**

```typescript
// 发送格式
type AudioMessage = {
  type: 0x01;
  payload: Uint8Array;  // Opus encoded audio
};

// 音频参数
const AUDIO_CONFIG = {
  sampleRate: 24000,      // 24kHz
  channels: 1,            // Mono
  codec: 'opus',
  frameSize: 960,         // 40ms frames
  bitrate: 24000          // 24kbps
};
```

**2. 文本消息 (0x02)**

```typescript
// 发送/接收格式
type TextMessage = {
  type: 0x02;
  payload: string;  // UTF-8 text
};
```

**3. 元数据消息 (0x03)**

```typescript
type MetadataMessage = {
  type: 0x03;
  payload: {
    // 会话配置
    system_prompt?: string;
    voice_id?: string;
    language?: string;

    // 触发上下文 (主动呼叫时)
    trigger_context?: TriggerContext;
  };
};
```

#### JSON 控制消息

```typescript
// 客户端 → 服务端
type ClientControlMessage =
  | { type: 'start' }           // 开始说话
  | { type: 'pause' }           // 暂停说话
  | { type: 'endTurn' }         // 结束本轮
  | { type: 'interrupt' }       // 打断AI
  | { type: 'confirm', action: string }  // 确认操作
  | { type: 'cancel' }          // 取消操作

// 服务端 → 客户端
type ServerControlMessage =
  | { type: 'listening' }       // 进入监听
  | { type: 'processing' }      // 正在处理
  | { type: 'speaking' }        // 开始播放
  | { type: 'endTurn' }         // 本轮结束
  | { type: 'needConfirm', action: PendingAction }  // 需要确认
  | { type: 'error', code: string, message: string }
```

### 2.3 交互流程

#### 2.3.1 用户主动对话

```
Client                              Server
  │                                    │
  │──── WebSocket Connect ────────────▶│
  │◀─── { type: 'listening' } ────────│
  │                                    │
  │──── { type: 'start' } ────────────▶│
  │──── 0x01 + opus_audio ────────────▶│
  │──── 0x01 + opus_audio ────────────▶│
  │──── { type: 'pause' } ─────────────▶│
  │                                    │
  │◀─── { type: 'processing' } ───────│
  │                                    │
  │◀─── { type: 'speaking' } ─────────│
  │◀─── 0x02 + "正在分析..." ──────────│
  │◀─── 0x01 + opus_audio ────────────│
  │◀─── 0x01 + opus_audio ────────────│
  │◀─── 0x02 + "分析完成..." ──────────│
  │◀─── { type: 'endTurn' } ──────────│
  │                                    │
  │◀─── { type: 'listening' } ────────│
  │                                    │
```

#### 2.3.2 服务端主动呼叫

```
Server                              Client
  │                                    │
  │── Push Notification (APNs/FCM) ──▶│ (用户收到来电)
  │                                    │
  │◀── POST /api/call/accept ─────────│ (用户接听)
  │                                    │
  │◀─── WebSocket Connect ────────────│
  │                                    │
  │──── { type: 'speaking' } ─────────▶│
  │──── 0x02 + "紧急提醒..." ─────────▶│
  │──── 0x01 + opus_audio ────────────▶│
  │──── { type: 'endTurn' } ──────────▶│
  │                                    │
  │──── { type: 'listening' } ────────▶│
  │◀─── { type: 'start' } ────────────│
  │◀─── 0x01 + opus_audio ────────────│
  │                                    │
```

#### 2.3.3 交易确认流程

```
Client                              Server
  │                                    │
  │◀─── 0x02 + "确认买入50股..." ─────│
  │◀─── 0x01 + opus_audio ────────────│
  │◀─── { type: 'needConfirm',        │
  │       action: {                    │
  │         type: 'trade',             │
  │         symbol: 'AAPL',            │
  │         side: 'buy',               │
  │         quantity: 50,              │
  │         estimated_amount: 9750     │
  │       }                            │
  │     } ────────────────────────────│
  │                                    │
  │──── { type: 'start' } ────────────▶│
  │──── 0x01 + "确认" ────────────────▶│ (语音确认)
  │──── { type: 'pause' } ─────────────▶│
  │                                    │
  │◀─── 0x02 + "订单已提交" ──────────│
  │◀─── { type: 'endTurn' } ──────────│
  │                                    │
```

### 2.4 错误码

| Code | Message | 说明 |
|------|---------|------|
| `E001` | `session_expired` | 会话已过期 |
| `E002` | `auth_failed` | 认证失败 |
| `E003` | `rate_limited` | 请求频率限制 |
| `E004` | `model_unavailable` | 模型服务不可用 |
| `E005` | `audio_decode_error` | 音频解码失败 |
| `E006` | `confirmation_timeout` | 确认超时 |
| `E007` | `trade_failed` | 交易执行失败 |

---

## 三、REST 接口

### 3.1 会话管理

#### 创建会话

```http
POST /api/session
Content-Type: application/json
Authorization: Bearer {jwt_token}

{
  "user_id": "u_123456",
  "voice_id": "zh-CN-XiaoxiaoNeural",
  "language": "zh-CN",
  "system_prompt": "你是Vibe Finance智能助手...",
  "trigger_context": {
    "trigger_type": "stop_loss",
    "symbol": "AAPL",
    "current_price": 185.50,
    "threshold": 190.00
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "session_id": "s_789012",
    "websocket_url": "wss://voice.vibe.finance/api/chat?session_id=s_789012",
    "expires_at": "2026-01-07T10:30:00Z"
  }
}
```

#### 获取会话状态

```http
GET /api/session/{session_id}
Authorization: Bearer {jwt_token}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "session_id": "s_789012",
    "user_id": "u_123456",
    "status": "active",
    "created_at": "2026-01-07T09:30:00Z",
    "last_activity": "2026-01-07T09:45:00Z",
    "turn_count": 5,
    "pending_action": null
  }
}
```

#### 结束会话

```http
DELETE /api/session/{session_id}
Authorization: Bearer {jwt_token}
```

### 3.2 主动呼叫

#### 发起呼叫

```http
POST /api/call/initiate
Content-Type: application/json
Authorization: Bearer {service_token}

{
  "user_id": "u_123456",
  "trigger_context": {
    "trigger_id": "tr_001",
    "trigger_type": "stop_loss",
    "priority": "high",
    "symbol": "AAPL",
    "symbol_name": "苹果",
    "current_price": 185.50,
    "threshold": 190.00,
    "user_holdings": [
      {
        "symbol": "AAPL",
        "quantity": 100,
        "cost_basis": 180.00
      }
    ],
    "user_pnl": -450.00,
    "market_context": {
      "index_change": -0.02,
      "sector_change": -0.015
    }
  },
  "notification_options": {
    "title": "止损提醒",
    "body": "苹果触发止损线",
    "sound": "urgent"
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "call_id": "c_345678",
    "session_id": "s_789012",
    "status": "pending",
    "notification_sent": true,
    "expires_at": "2026-01-07T09:35:00Z"
  }
}
```

#### 接听呼叫

```http
POST /api/call/accept
Content-Type: application/json
Authorization: Bearer {jwt_token}

{
  "call_id": "c_345678"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "session_id": "s_789012",
    "websocket_url": "wss://voice.vibe.finance/api/chat?session_id=s_789012",
    "opening_message": "紧急提醒，你的苹果持仓触发了止损..."
  }
}
```

### 3.3 健康检查

```http
GET /api/health
```

**Response**:
```json
{
  "status": "healthy",
  "components": {
    "voice_engine": "healthy",
    "websocket": "healthy",
    "agent_service": "healthy",
    "redis": "healthy"
  },
  "version": "1.0.0",
  "uptime": 86400
}
```

---

## 四、与 DeepAgent 集成接口

### 4.1 Voice-Agent Bridge API

Voice Gateway 调用 DeepAgent 的内部接口：

```http
POST /internal/agent/invoke
Content-Type: application/json
X-Internal-Token: {internal_token}

{
  "user_id": "u_123456",
  "thread_id": "voice_s_789012",
  "query": "为什么苹果股票今天跌了",
  "intent": "analyze",
  "entities": {
    "symbol": "AAPL",
    "market": "us-stock"
  },
  "emotion": "anxious",
  "context": {
    "trigger_context": {
      "trigger_type": "stop_loss",
      "current_price": 185.50
    },
    "conversation_history": [
      {"role": "assistant", "content": "紧急提醒..."},
      {"role": "user", "content": "为什么跌这么多"}
    ]
  },
  "stream": true
}
```

**Streaming Response** (SSE):
```
event: chunk
data: {"type": "text", "content": "今天大盘下跌"}

event: chunk
data: {"type": "text", "content": "2%，苹果跟随..."}

event: tool_call
data: {"tool": "market_data", "input": {"symbol": "AAPL"}}

event: tool_result
data: {"tool": "market_data", "output": {"price": 185.50, "change": -0.03}}

event: chunk
data: {"type": "text", "content": "当前股价185.50..."}

event: done
data: {"finish_reason": "stop"}
```

### 4.2 DeepAgent SubAgent 调用规范

Voice Gateway 作为特殊的 SubAgent 接入：

```python
# 在 DeepAgent Router 中注册
VOICE_SUBAGENT_CONFIG = {
    "voice-input": {
        "description": "处理语音输入，转换为结构化查询",
        "input_schema": VoiceInputSchema,
        "output_schema": StructuredQuerySchema
    },
    "voice-output": {
        "description": "将 Agent 输出转换为语音响应",
        "input_schema": AgentOutputSchema,
        "output_schema": VoiceOutputSchema
    }
}

# Router 调用示例
task(
    subagent_type="voice-output",
    description="将分析结果转换为语音",
    content="苹果今天下跌3%，主要原因是...",
    emotion_hint="reassuring",
    include_action=True
)
```

### 4.3 触发器回调接口

Celery Worker 调用 Voice Gateway 发起呼叫：

```python
# celery_tasks.py
from voice_gateway_client import VoiceGatewayClient

@app.task
def trigger_voice_alert(user_id: str, trigger_context: dict):
    """触发语音提醒"""

    client = VoiceGatewayClient()

    response = client.initiate_call(
        user_id=user_id,
        trigger_context=trigger_context,
        notification_options={
            "title": trigger_context.get("alert_title"),
            "body": trigger_context.get("alert_summary"),
            "sound": "urgent" if trigger_context.get("priority") == "high" else "default"
        }
    )

    return response.call_id
```

---

## 五、前端 SDK

### 5.1 TypeScript SDK

```typescript
// voice-client.ts

interface VoiceClientConfig {
  baseUrl: string;
  userId: string;
  authToken: string;
  onMessage?: (message: VoiceMessage) => void;
  onStateChange?: (state: VoiceState) => void;
  onError?: (error: VoiceError) => void;
}

class VoiceClient {
  private ws: WebSocket | null = null;
  private audioContext: AudioContext;
  private mediaRecorder: MediaRecorder;
  private opusEncoder: OpusEncoder;
  private opusDecoder: OpusDecoder;

  constructor(private config: VoiceClientConfig) {
    this.audioContext = new AudioContext({ sampleRate: 24000 });
    this.opusEncoder = new OpusEncoder({ sampleRate: 24000, channels: 1 });
    this.opusDecoder = new OpusDecoder({ sampleRate: 24000, channels: 1 });
  }

  /**
   * 创建新会话并连接
   */
  async connect(options?: SessionOptions): Promise<void> {
    // 创建会话
    const session = await this.createSession(options);

    // 建立 WebSocket
    this.ws = new WebSocket(session.websocket_url);

    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.config.onStateChange?.('connected');
    };

    this.ws.onmessage = (event) => {
      this.handleMessage(event.data);
    };

    this.ws.onerror = (error) => {
      this.config.onError?.({ code: 'WS_ERROR', message: error.toString() });
    };

    this.ws.onclose = () => {
      this.config.onStateChange?.('disconnected');
    };
  }

  /**
   * 开始录音
   */
  async startRecording(): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    this.mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus'
    });

    this.mediaRecorder.ondataavailable = async (event) => {
      if (event.data.size > 0) {
        const opusData = await this.encodeAudio(event.data);
        this.sendAudio(opusData);
      }
    };

    // 发送开始信号
    this.sendControl({ type: 'start' });

    // 开始录音
    this.mediaRecorder.start(100); // 100ms chunks
  }

  /**
   * 停止录音
   */
  stopRecording(): void {
    if (this.mediaRecorder) {
      this.mediaRecorder.stop();
      this.sendControl({ type: 'pause' });
    }
  }

  /**
   * 打断AI播放
   */
  interrupt(): void {
    this.sendControl({ type: 'interrupt' });
  }

  /**
   * 确认操作
   */
  confirmAction(action: string): void {
    this.sendControl({ type: 'confirm', action });
  }

  /**
   * 取消操作
   */
  cancelAction(): void {
    this.sendControl({ type: 'cancel' });
  }

  private sendAudio(data: Uint8Array): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const message = new Uint8Array(1 + data.length);
      message[0] = 0x01;
      message.set(data, 1);
      this.ws.send(message);
    }
  }

  private sendControl(control: ControlMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(control));
    }
  }

  private handleMessage(data: ArrayBuffer | string): void {
    if (typeof data === 'string') {
      // JSON 控制消息
      const control = JSON.parse(data);
      this.handleControlMessage(control);
    } else {
      // 二进制消息
      const bytes = new Uint8Array(data);
      const type = bytes[0];
      const payload = bytes.slice(1);

      if (type === 0x01) {
        // 音频
        this.playAudio(payload);
      } else if (type === 0x02) {
        // 文本
        const text = new TextDecoder().decode(payload);
        this.config.onMessage?.({ type: 'text', content: text });
      }
    }
  }

  private async playAudio(opusData: Uint8Array): Promise<void> {
    const pcm = await this.opusDecoder.decode(opusData);
    const audioBuffer = this.audioContext.createBuffer(1, pcm.length, 24000);
    audioBuffer.getChannelData(0).set(pcm);

    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);
    source.start();
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
  }
}

export { VoiceClient, VoiceClientConfig };
```

### 5.2 React Hook

```typescript
// useVoiceChat.ts
import { useState, useCallback, useEffect, useRef } from 'react';
import { VoiceClient } from './voice-client';

type VoiceState = 'idle' | 'connecting' | 'listening' | 'processing' | 'speaking';

interface UseVoiceChatOptions {
  baseUrl: string;
  userId: string;
  authToken: string;
  onTranscript?: (text: string) => void;
  onConfirmNeeded?: (action: PendingAction) => void;
}

function useVoiceChat(options: UseVoiceChatOptions) {
  const [state, setState] = useState<VoiceState>('idle');
  const [isRecording, setIsRecording] = useState(false);
  const [transcripts, setTranscripts] = useState<string[]>([]);
  const [pendingAction, setPendingAction] = useState<PendingAction | null>(null);

  const clientRef = useRef<VoiceClient | null>(null);

  useEffect(() => {
    clientRef.current = new VoiceClient({
      ...options,
      onStateChange: setState,
      onMessage: (msg) => {
        if (msg.type === 'text') {
          setTranscripts(prev => [...prev, msg.content]);
          options.onTranscript?.(msg.content);
        }
      },
      onError: (error) => {
        console.error('Voice error:', error);
      }
    });

    return () => {
      clientRef.current?.disconnect();
    };
  }, [options]);

  const connect = useCallback(async () => {
    setState('connecting');
    await clientRef.current?.connect();
  }, []);

  const startRecording = useCallback(async () => {
    await clientRef.current?.startRecording();
    setIsRecording(true);
  }, []);

  const stopRecording = useCallback(() => {
    clientRef.current?.stopRecording();
    setIsRecording(false);
  }, []);

  const interrupt = useCallback(() => {
    clientRef.current?.interrupt();
  }, []);

  const confirm = useCallback(() => {
    if (pendingAction) {
      clientRef.current?.confirmAction(pendingAction.id);
      setPendingAction(null);
    }
  }, [pendingAction]);

  const cancel = useCallback(() => {
    clientRef.current?.cancelAction();
    setPendingAction(null);
  }, []);

  const disconnect = useCallback(() => {
    clientRef.current?.disconnect();
    setState('idle');
  }, []);

  return {
    state,
    isRecording,
    transcripts,
    pendingAction,
    connect,
    startRecording,
    stopRecording,
    interrupt,
    confirm,
    cancel,
    disconnect
  };
}

export { useVoiceChat };
```

### 5.3 使用示例

```tsx
// VoiceChatComponent.tsx
import { useVoiceChat } from './useVoiceChat';

function VoiceChatComponent() {
  const {
    state,
    isRecording,
    transcripts,
    pendingAction,
    connect,
    startRecording,
    stopRecording,
    interrupt,
    confirm,
    cancel,
    disconnect
  } = useVoiceChat({
    baseUrl: 'wss://voice.vibe.finance',
    userId: 'user_123',
    authToken: 'jwt_token',
    onTranscript: (text) => console.log('AI:', text)
  });

  return (
    <div className="voice-chat">
      <div className="status">状态: {state}</div>

      <div className="transcripts">
        {transcripts.map((t, i) => (
          <div key={i}>{t}</div>
        ))}
      </div>

      {state === 'idle' && (
        <button onClick={connect}>开始对话</button>
      )}

      {state === 'listening' && (
        <button
          onMouseDown={startRecording}
          onMouseUp={stopRecording}
          className={isRecording ? 'recording' : ''}
        >
          {isRecording ? '松开发送' : '按住说话'}
        </button>
      )}

      {state === 'speaking' && (
        <button onClick={interrupt}>打断</button>
      )}

      {pendingAction && (
        <div className="confirm-dialog">
          <p>{pendingAction.description}</p>
          <button onClick={confirm}>确认</button>
          <button onClick={cancel}>取消</button>
        </div>
      )}

      <button onClick={disconnect}>结束对话</button>
    </div>
  );
}
```

---

## 六、数据结构定义

### 6.1 触发上下文

```typescript
interface TriggerContext {
  // 触发信息
  trigger_id: string;
  trigger_type: TriggerType;
  trigger_time: string;  // ISO 8601
  priority: 'low' | 'medium' | 'high' | 'critical';

  // 标的信息
  symbol: string;
  symbol_name: string;
  current_price: number;
  price_change: number;

  // 规则信息
  rule_id: string;
  threshold: number;
  condition: 'above' | 'below' | 'change_pct';

  // 用户上下文
  user_id: string;
  user_holdings: Holding[];
  user_cost_basis: number;
  user_pnl: number;

  // 市场上下文
  market_sentiment: string;
  related_news: News[];
  sector_performance: number;
}

type TriggerType =
  | 'stop_loss'
  | 'price_target'
  | 'option_expiry'
  | 'earnings'
  | 'major_event'
  | 'strategy_signal'
  | 'morning_brief';
```

### 6.2 待确认操作

```typescript
interface PendingAction {
  id: string;
  type: 'trade' | 'alert_dismiss' | 'strategy_adjust';
  created_at: string;
  expires_at: string;

  // 交易类型
  trade?: {
    symbol: string;
    side: 'buy' | 'sell';
    quantity: number;
    order_type: 'market' | 'limit';
    limit_price?: number;
    estimated_amount: number;
  };

  // 其他操作类型...
}
```

### 6.3 会话状态

```typescript
interface VoiceSession {
  session_id: string;
  user_id: string;
  status: 'active' | 'paused' | 'ended';
  created_at: string;
  last_activity: string;
  turn_count: number;

  // 配置
  voice_id: string;
  language: string;
  system_prompt: string;

  // 触发上下文 (如有)
  trigger_context?: TriggerContext;

  // 待确认操作
  pending_action?: PendingAction;

  // 对话历史
  conversation: ConversationTurn[];
}

interface ConversationTurn {
  role: 'user' | 'assistant';
  content: string;
  audio_duration?: number;
  timestamp: string;
}
```

---

## 七、安全规范

### 7.1 认证

- WebSocket 连接需携带 JWT Token (query param 或 Sec-WebSocket-Protocol)
- REST API 使用 Bearer Token
- 内部服务间通信使用 mTLS + Service Token

### 7.2 交易确认

- 所有交易操作需要语音确认
- 确认超时 30 秒自动取消
- 大额交易 (>$10,000) 需要二次确认

### 7.3 敏感信息

- 不在日志中记录完整音频
- 交易信息脱敏后记录
- 用户语音数据 30 天后自动清理

---

## 八、性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 首包延迟 | <500ms | 用户说完到AI开始响应 |
| 端到端延迟 | <2s | 完整一轮对话 |
| WebSocket 重连 | <3s | 断线自动重连 |
| 并发会话 | 1000/节点 | 单节点支持 |
| 可用性 | 99.9% | SLA 目标 |

---

*Voice API Specification v1.0 | 2026-01-07*
