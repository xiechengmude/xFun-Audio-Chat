# 全双工语音 Agent 架构方案

> v1.0 | 2026-01-07

---

## 一、核心定位

**本质**：语音 Agent = 用户的手 + 用户的耳

| 方向 | 能力 | 场景 |
|------|------|------|
| **输入** | 语音→指令 | 开车时说"查一下特斯拉" |
| **输出** | 内容→语音 | 播报刚生成的分析报告 |
| **交互** | 全双工对话 | 多轮追问、打断、确认 |

---

## 二、系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                        全双工语音 Agent                               │
│                                                                      │
│  APP (用户说话)                                                       │
│       ↓ WebSocket (Opus音频)                                         │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    Voice Gateway (xplatform)                   │  │
│  │  ┌──────────────────┐  ┌─────────────────────────────────────┐│  │
│  │  │ Fun-Audio-Chat   │  │         Voice-Agent Bridge          ││  │
│  │  │  (语音↔文本)     │  │  ┌─────────────────────────────┐   ││  │
│  │  │                  │←→│  │    LangGraph Client         │   ││  │
│  │  │ "分析苹果财报"   │  │  │  POST /threads/xxx/runs     │   ││  │
│  │  │        ↓         │  │  │         ↓                   │   ││  │
│  │  │ "苹果Q3营收..."  │←─│  │  Strategy Agent 返回分析    │   ││  │
│  │  └──────────────────┘  │  └─────────────────────────────┘   ││  │
│  │                        │                                     ││  │
│  │                        │  ┌─────────────────────────────┐   ││  │
│  │                        │  │    Context Manager          │   ││  │
│  │                        │  │  - 对话历史                  │   ││  │
│  │                        │  │  - TriggerContext           │   ││  │
│  │                        │  │  - 用户偏好                  │   ││  │
│  │                        │  └─────────────────────────────┘   ││  │
│  │                        └─────────────────────────────────────┘│  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                    ↓ HTTP                            │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │              LangGraph Server (strategy agent)                 │  │
│  │  POST /threads/{thread_id}/runs/stream                         │  │
│  │       ↓                                                        │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │  │
│  │  │smart_search │  │finance_exec │  │strategy_executor    │    │  │
│  │  │(知识增强)   │  │(金融分析)   │  │(策略回测)           │    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘    │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 三、服务组件

### 3.1 xplatform (基础服务)

| 组件 | 文件 | 功能 |
|------|------|------|
| VoiceCallNotifier | `notification_channels.py` | 触发语音呼叫 |
| VoiceCallService | `voice_call_service.py` | 会话管理、Redis 存储 |
| Voice Gateway | `voice_gateway/` | WebSocket 端点 |
| - bridge.py | | Fun-Audio-Chat 客户端 |
| - session.py | | 会话状态管理 |
| - server.py | | FastAPI WebSocket |
| - langgraph_client.py | | LangGraph API 调用 |

### 3.2 xdan-vibe-finance-strategy (LangGraph Agent)

| 组件 | 功能 |
|------|------|
| `factory.py:get_agent()` | LangGraph Server 入口 |
| smart_search | 知识增强、事实验证 |
| explorer | API/SKILL 发现 |
| finance_executor | 金融数据分析 |
| strategy_executor | 策略回测 |

---

## 四、交互协议

### 4.1 APP ↔ Voice Gateway

**WebSocket 端点**: `wss://api.xxx.com/v1/voice/chat?session_id=xxx`

**二进制消息格式**:
```
┌─────────┬──────────────────────────┐
│ 1 byte  │      Payload             │
├─────────┼──────────────────────────┤
│  0x01   │   Opus 音频数据          │
│  0x02   │   UTF-8 文本             │
└─────────┴──────────────────────────┘
```

**JSON 控制消息**:
```javascript
// 客户端 → 服务端
{ "type": "start" }      // 开始说话
{ "type": "pause" }      // 停止说话
{ "type": "end" }        // 结束对话

// 服务端 → 客户端
{ "type": "ready" }      // 连接就绪
{ "type": "listening" }  // 等待用户输入
{ "type": "processing" } // 正在处理
{ "type": "speaking" }   // AI 正在说话
{ "type": "endTurn" }    // AI 说完一段
```

### 4.2 Voice Gateway ↔ Fun-Audio-Chat

同上协议，通过 `bridge.py` 转发。

### 4.3 Voice Gateway ↔ LangGraph

**REST API**:
```http
POST /threads
→ { "thread_id": "xxx" }

POST /threads/{thread_id}/runs/stream
← SSE 流式响应
```

---

## 五、数据流

### 5.1 主动触发流程 (智能提醒)

```
1. Celery 定时任务扫描 pending 提醒
2. 检测到止损触发 → should_use_voice_call()
3. VoiceCallService 创建会话
   → Redis: voice_session:{id} = {trigger_context, ...}
4. 推送通知 APP: "语音来电"
5. 用户接听 → WebSocket 连接
6. Voice Gateway:
   ├─ 从 Redis 获取 TriggerContext
   ├─ 连接 Fun-Audio-Chat
   ├─ 注入上下文 (AI 知道为什么呼叫)
   └─ 双向转发音频
7. AI: "您的 AAPL 止损位 165 已触发，当前价 164.2，要执行卖出吗？"
```

### 5.2 用户主动流程 (点击开启)

```
1. 用户点击语音按钮
2. APP 建立 WebSocket 连接
3. Voice Gateway:
   ├─ 创建 LangGraph Thread
   ├─ 连接 Fun-Audio-Chat
   └─ 等待用户说话
4. 用户: "分析一下苹果最近的财报"
5. Fun-Audio-Chat 语音转文本
6. LangGraph Client 调用 Strategy Agent
7. Strategy Agent 返回分析结果
8. Fun-Audio-Chat 文本转语音
9. 播放给用户
```

---

## 六、环境变量

```bash
# Fun-Audio-Chat
FUN_AUDIO_CHAT_ENABLED=true
FUN_AUDIO_CHAT_HOST=194.68.245.6
FUN_AUDIO_CHAT_WS_PORT=22035

# LangGraph
LANGGRAPH_API_URL=http://localhost:8123
LANGGRAPH_API_KEY=xxx
LANGGRAPH_GRAPH_NAME=strategy

# Redis (会话存储)
REDIS_URL=redis://localhost:6379/0
```

---

## 七、API 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| WS | `/api/v1/voice/chat` | 语音对话 WebSocket |
| GET | `/api/v1/voice/sessions/{id}` | 获取会话信息 |
| GET | `/api/v1/voice/health` | 健康检查 |

---

## 八、阶段演进

| 阶段 | 功能 | 复杂度 |
|------|------|--------|
| **P0** | 语音输入+播报 (REST) | ★☆☆ |
| **V1** | 主动呼叫+简单对话 | ★★★ |
| **V2** | 全双工+工具调用 | ★★★★★ |

### 当前实现 (V2)

- [x] VoiceCallNotifier
- [x] VoiceCallService
- [x] Voice Gateway Server
- [x] Fun-Audio-Chat Bridge
- [x] Session Manager
- [x] LangGraph Client
- [ ] Voice-Agent Bridge (语音↔LangGraph 消息转换)
- [ ] Context Manager (多轮对话状态)
- [ ] Confirmation Handler (敏感操作确认)

---

## 九、安全考虑

### 9.1 敏感操作确认

交易类指令需要多重确认:

```
用户: "卖掉苹果"
AI: "确认要卖出您持有的 100 股苹果吗？当前价 172.5"
用户: "确认"
AI: "请说出确认码 8527"
用户: "8527"
AI: "已提交卖出订单..."
```

### 9.2 权限控制

- JWT Token 验证
- 操作权限检查
- 频率限制

---

## 十、部署架构

```
┌─────────────────┐     ┌─────────────────┐
│   xplatform     │     │  LangGraph      │
│   (K8s Pod)     │────→│  Server (GPU)   │
│   + Voice GW    │     │                 │
└────────┬────────┘     └─────────────────┘
         │
         ↓
┌─────────────────┐
│ Fun-Audio-Chat  │
│ (RunPod A40)    │
│ 194.68.245.6    │
└─────────────────┘
```

---

*全双工语音 Agent 架构方案 v1.0*
