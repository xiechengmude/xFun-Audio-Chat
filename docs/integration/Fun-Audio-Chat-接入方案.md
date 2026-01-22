# Fun-Audio-Chat 接入方案

> 对接 DeepAgent + xplatform 智能提醒
> Integration Plan v1.0 | 2026-01-07

---

## 一、系统现状分析

### 1.1 DeepAgent 现有架构

```
deepagents/
├── graph.py                 # create_deep_agent() 核心入口
├── factory.py               # Agent 工厂 + 模型配置
├── middleware/
│   ├── subagents.py         # SubAgent 机制 (task tool)
│   ├── memory_recall.py     # 记忆召回
│   ├── memory_processing.py # 记忆处理
│   └── plan_subagent.py     # 规划 SubAgent
└── tools/                   # 工具层
```

**关键接口**:
- `SubAgent` TypedDict - 定义SubAgent规格
- `SubAgentMiddleware` - 注册SubAgent到Agent
- `task()` tool - Router调用SubAgent

### 1.2 xplatform 智能提醒现有架构

```
xplatform/
├── services/
│   ├── notification_channels.py  # 通知渠道 (含 VoiceNotifier/CallNotifier)
│   ├── notification_service.py   # 通知服务
│   ├── reminder_agent.py         # 提醒Agent
│   └── tts_service.py            # TTS服务
└── tasks/
    └── reminder_tasks.py         # Celery定时任务
```

**关键接口**:
- `NotificationDispatcher.dispatch()` - 多渠道分发
- `VoiceNotifier` - 语音播报 (TTS)
- `CallNotifier` - 电话提醒 (未实现)
- Celery Beat - `check_condition_triggers` 条件检测

---

## 二、接入架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Fun-Audio-Chat 接入架构                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Voice Gateway (新增)                          │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │  │
│  │  │ voice_gateway/  │  │ voice_gateway/  │  │ voice_gateway/      │   │  │
│  │  │ server.py       │  │ bridge.py       │  │ session_manager.py  │   │  │
│  │  │ (WebSocket)     │  │ (Agent桥接)     │  │ (会话管理)          │   │  │
│  │  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘   │  │
│  │           │                    │                       │              │  │
│  │           └────────────────────┼───────────────────────┘              │  │
│  │                                │                                      │  │
│  │  ┌─────────────────────────────▼─────────────────────────────────┐   │  │
│  │  │                  Fun-Audio-Chat Client                         │   │  │
│  │  │              (RunPod GPU: 194.68.245.6:22035)                  │   │  │
│  │  └────────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                   │                                         │
│                                   │ gRPC/HTTP                               │
│                                   ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      DeepAgent (改造)                                 │  │
│  │                                                                       │  │
│  │  middleware/                                                          │  │
│  │  ├── voice_subagent.py      # 新增: 语音SubAgent规格                  │  │
│  │  └── subagents.py           # 注册voice-input/voice-output           │  │
│  │                                                                       │  │
│  │  tools/                                                               │  │
│  │  └── voice_tools.py         # 新增: 语音相关工具                      │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                   │                                         │
│                                   │ 触发                                    │
│                                   ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      xplatform (改造)                                 │  │
│  │                                                                       │  │
│  │  services/                                                            │  │
│  │  ├── notification_channels.py                                        │  │
│  │  │   └── VoiceCallNotifier   # 新增: 语音对话通知器                   │  │
│  │  └── voice_call_service.py   # 新增: 语音呼叫服务                     │  │
│  │                                                                       │  │
│  │  tasks/                                                               │  │
│  │  └── reminder_tasks.py       # 改造: 条件触发→语音呼叫                │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 组件职责

| 组件 | 位置 | 职责 |
|------|------|------|
| `VoiceGatewayServer` | voice_gateway/server.py | WebSocket服务、连接管理 |
| `VoiceAgentBridge` | voice_gateway/bridge.py | 语音↔DeepAgent语义转换 |
| `VoiceSessionManager` | voice_gateway/session_manager.py | 会话状态、Redis持久化 |
| `VoiceSubAgent` | deepagents/middleware/voice_subagent.py | 语音SubAgent定义 |
| `VoiceCallNotifier` | xplatform/services/notification_channels.py | 语音呼叫通知器 |
| `VoiceCallService` | xplatform/services/voice_call_service.py | 呼叫发起、会话创建 |

---

## 三、代码实现

### 3.1 xplatform: 新增 VoiceCallNotifier

**文件**: `xplatform/services/notification_channels.py`

```python
# ============================================
# Voice Call Notifier (Fun-Audio-Chat 语音对话)
# ============================================
class VoiceCallNotifier(BaseNotifier):
    """语音对话通知器 - 使用 Fun-Audio-Chat 进行实时语音对话

    适用场景:
    - 止损/风控触发 (紧急 + 需要决策)
    - 价格目标达成 (需要确认操作)
    - 策略信号触发 (需要确认执行)

    工作流程:
    1. 创建语音会话 (带触发上下文)
    2. 发送 Push 邀请用户接听
    3. 用户接听后建立 WebSocket 对话
    4. Fun-Audio-Chat 以上下文开场白开始对话
    """

    def __init__(self):
        self.voice_gateway_url = os.getenv(
            "VOICE_GATEWAY_URL",
            "http://localhost:8100"
        )
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def send(
        self,
        user_id: str,
        title: str,
        content: str,
        **kwargs
    ) -> dict:
        """发起语音呼叫

        Args:
            user_id: 用户ID
            title: 通知标题
            content: 通知内容
            **kwargs:
                trigger_context: TriggerContext 触发上下文
                priority: 优先级 (low/medium/high/critical)
                push_title: Push通知标题
                push_body: Push通知内容
        """
        trigger_context = kwargs.get("trigger_context", {})
        priority = kwargs.get("priority", "high")

        try:
            # 1. 调用 Voice Gateway 创建会话
            response = await self.http_client.post(
                f"{self.voice_gateway_url}/api/call/initiate",
                json={
                    "user_id": user_id,
                    "trigger_context": {
                        "trigger_type": trigger_context.get("trigger_type"),
                        "symbol": trigger_context.get("symbol"),
                        "symbol_name": trigger_context.get("symbol_name"),
                        "current_price": trigger_context.get("current_price"),
                        "threshold": trigger_context.get("threshold"),
                        "user_pnl": trigger_context.get("user_pnl"),
                        "priority": priority,
                        "title": title,
                        "content": content,
                        # 传递更多上下文给语音会话
                        "market_context": trigger_context.get("market_context"),
                        "user_holdings": trigger_context.get("user_holdings"),
                    },
                    "notification_options": {
                        "title": kwargs.get("push_title", f"🔔 {title}"),
                        "body": kwargs.get("push_body", content[:50]),
                        "sound": "urgent" if priority in ("high", "critical") else "default"
                    }
                }
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "success",
                    "channel": "voice_call",
                    "call_id": data.get("call_id"),
                    "session_id": data.get("session_id"),
                    "expires_at": data.get("expires_at")
                }
            else:
                logger.error(f"Voice call failed: {response.text}")
                return {
                    "status": "failed",
                    "channel": "voice_call",
                    "error": response.text
                }

        except Exception as e:
            logger.error(f"Failed to initiate voice call: {e}")
            return {"status": "failed", "channel": "voice_call", "error": str(e)}

    async def close(self):
        """清理资源"""
        await self.http_client.aclose()


# 更新 NotificationDispatcher
class NotificationDispatcher:
    """通知调度器"""

    def __init__(self):
        self.notifiers: dict[str, BaseNotifier] = {
            NotificationChannel.NOTIFICATION.value: InAppNotifier(),
            NotificationChannel.PUSH.value: PushNotifier(),
            NotificationChannel.CHAT.value: ChatNotifier(),
            NotificationChannel.POPUP.value: PopupNotifier(),
            NotificationChannel.SOUND.value: SoundNotifier(),
            NotificationChannel.SMS.value: SMSNotifier(),
            NotificationChannel.EMAIL.value: EmailNotifier(),
            NotificationChannel.VOICE.value: VoiceNotifier(),
            NotificationChannel.CALL.value: CallNotifier(),
            # 新增: 语音对话
            "voice_call": VoiceCallNotifier(),
        }
```

### 3.2 xplatform: 新增 VoiceCallService

**文件**: `xplatform/services/voice_call_service.py`

```python
"""Voice Call Service - 语音呼叫服务

管理语音呼叫的生命周期:
- 发起呼叫
- 接听呼叫
- 会话状态查询
- 呼叫历史记录
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

import httpx
import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class VoiceCall:
    """语音呼叫"""
    call_id: str
    user_id: str
    session_id: str
    trigger_context: dict
    status: str  # pending, accepted, rejected, expired, completed
    created_at: datetime
    expires_at: datetime
    accepted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class VoiceCallService:
    """语音呼叫服务"""

    def __init__(self):
        self.voice_gateway_url = os.getenv(
            "VOICE_GATEWAY_URL",
            "http://localhost:8100"
        )
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis: Optional[redis.Redis] = None
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.call_ttl = 300  # 呼叫5分钟过期

    async def _get_redis(self) -> redis.Redis:
        if self.redis is None:
            self.redis = await redis.from_url(self.redis_url)
        return self.redis

    async def initiate_call(
        self,
        user_id: str,
        trigger_context: dict,
        notification_options: dict
    ) -> VoiceCall:
        """发起语音呼叫

        Args:
            user_id: 用户ID
            trigger_context: 触发上下文
            notification_options: Push通知选项

        Returns:
            VoiceCall 对象
        """
        # 1. 调用 Voice Gateway 创建会话
        response = await self.http_client.post(
            f"{self.voice_gateway_url}/api/call/initiate",
            json={
                "user_id": user_id,
                "trigger_context": trigger_context,
                "notification_options": notification_options
            }
        )

        if response.status_code != 200:
            raise Exception(f"Failed to initiate call: {response.text}")

        data = response.json()["data"]

        # 2. 记录呼叫状态
        call = VoiceCall(
            call_id=data["call_id"],
            user_id=user_id,
            session_id=data["session_id"],
            trigger_context=trigger_context,
            status="pending",
            created_at=datetime.utcnow(),
            expires_at=datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
        )

        # 3. 存储到 Redis
        r = await self._get_redis()
        await r.setex(
            f"voice_call:{call.call_id}",
            self.call_ttl,
            call.__dict__
        )

        logger.info(f"Voice call initiated: {call.call_id} for user {user_id}")
        return call

    async def get_call(self, call_id: str) -> Optional[VoiceCall]:
        """获取呼叫状态"""
        r = await self._get_redis()
        data = await r.get(f"voice_call:{call_id}")
        if data:
            return VoiceCall(**data)
        return None

    async def accept_call(self, call_id: str) -> dict:
        """用户接听呼叫"""
        call = await self.get_call(call_id)
        if not call:
            raise Exception("Call not found or expired")

        if call.status != "pending":
            raise Exception(f"Call is not pending: {call.status}")

        # 调用 Voice Gateway 接听
        response = await self.http_client.post(
            f"{self.voice_gateway_url}/api/call/accept",
            json={"call_id": call_id}
        )

        if response.status_code != 200:
            raise Exception(f"Failed to accept call: {response.text}")

        data = response.json()["data"]

        # 更新呼叫状态
        call.status = "accepted"
        call.accepted_at = datetime.utcnow()

        r = await self._get_redis()
        await r.setex(
            f"voice_call:{call_id}",
            3600,  # 接听后延长1小时
            call.__dict__
        )

        return {
            "session_id": data["session_id"],
            "websocket_url": data["websocket_url"],
            "opening_message": data.get("opening_message")
        }

    async def close(self):
        if self.redis:
            await self.redis.close()
        await self.http_client.aclose()


# 单例
_voice_call_service: Optional[VoiceCallService] = None


def get_voice_call_service() -> VoiceCallService:
    global _voice_call_service
    if _voice_call_service is None:
        _voice_call_service = VoiceCallService()
    return _voice_call_service
```

### 3.3 xplatform: 改造 reminder_tasks.py

**文件**: `xplatform/tasks/reminder_tasks.py`

```python
# 在现有代码基础上添加

from xplatform.services.notification_channels import NotificationDispatcher, VoiceCallNotifier

# ============================================
# 路由决策: 是否使用语音对话
# ============================================
def should_use_voice_call(
    trigger_type: str,
    priority: str,
    user_preferences: dict = None
) -> bool:
    """判断是否使用语音对话

    Args:
        trigger_type: 触发类型
        priority: 优先级
        user_preferences: 用户偏好设置

    Returns:
        是否使用语音对话
    """
    # 用户关闭了语音呼叫
    if user_preferences and not user_preferences.get("voice_call_enabled", True):
        return False

    # 止损/风控 → 必须语音
    if trigger_type in ("stop_loss", "risk_alert", "margin_call"):
        return True

    # 价格目标达成 → 语音
    if trigger_type == "price_target":
        return True

    # 期权到期当天 → 语音
    if trigger_type == "option_expiry":
        return True

    # 策略信号 → 高优先级用语音
    if trigger_type == "strategy_signal" and priority in ("high", "critical"):
        return True

    # 其他高优先级 → 语音
    if priority in ("high", "critical"):
        return True

    return False


# ============================================
# 改造: 条件触发检查
# ============================================
@app.task
def check_condition_triggers():
    """检测条件触发 (价格/波动等)"""
    asyncio.run(_check_condition_triggers_async())


async def _check_condition_triggers_async():
    """异步检测条件触发"""
    engine = get_async_engine()
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        # 获取待检测的条件规则
        # ... 现有逻辑 ...

        for alert in triggered_alerts:
            # 构造触发上下文
            trigger_context = {
                "trigger_type": alert.trigger_type,
                "symbol": alert.symbol,
                "symbol_name": alert.symbol_name,
                "current_price": alert.current_price,
                "threshold": alert.threshold,
                "condition": alert.condition,
                "user_pnl": calculate_pnl(alert),
                "market_context": await get_market_context(),
                "user_holdings": await get_user_holdings(alert.user_id),
            }

            # 获取用户偏好
            user_prefs = await get_user_preferences(db, alert.user_id)

            # 决定通知渠道
            if should_use_voice_call(
                alert.trigger_type,
                alert.priority,
                user_prefs
            ):
                # 使用语音对话
                channels = ["voice_call"]
            else:
                # 使用传统渠道
                channels = ["push", "notification"]
                if alert.priority in ("high", "critical"):
                    channels.append("voice")  # TTS播报

            # 分发通知
            dispatcher = NotificationDispatcher()
            results = await dispatcher.dispatch(
                user_id=alert.user_id,
                title=alert.title,
                content=alert.content,
                channels=channels,
                trigger_context=trigger_context,
                priority=alert.priority
            )

            logger.info(f"Alert dispatched: {alert.id}, channels: {channels}, results: {results}")
```

### 3.4 DeepAgent: 新增 VoiceSubAgent

**文件**: `deepagents/middleware/voice_subagent.py`

```python
"""Voice SubAgent - 语音对话子Agent

提供语音对话能力:
- voice-router: 语音意图路由
- voice-analyzer: 语音场景分析
- voice-responder: 语音响应生成

用于 Voice Gateway 调用 DeepAgent 进行金融分析和决策。
"""

from typing import Sequence

from langchain_core.tools import BaseTool

from deepagents.middleware.subagents import SubAgent


def get_voice_subagent_specs(
    tools: Sequence[BaseTool] = None
) -> list[SubAgent]:
    """获取语音相关 SubAgent 规格

    Args:
        tools: 额外工具

    Returns:
        SubAgent 规格列表
    """
    return [
        {
            "name": "voice-router",
            "description": """语音意图路由器。
用于处理来自语音对话的用户请求，识别意图并路由到合适的处理流程。

适用场景:
- 用户语音输入的意图识别
- 带触发上下文的语音会话路由
- 情感感知的响应策略选择

输入:
- query: 用户语音转文本
- intent: 初步意图 (可选)
- emotion: 情感状态 (可选)
- trigger_context: 触发上下文 (可选)
""",
            "system_prompt": """你是语音对话路由器。

## 你的任务
分析用户的语音输入，识别意图，并决定如何处理。

## 触发上下文
如果存在 trigger_context，这是一个主动触发的语音会话：
- 止损触发: 需要分析原因 + 确认操作
- 价格目标: 需要分析后续 + 确认操作
- 策略信号: 需要解读信号 + 确认执行

## 情感感知
根据用户情感调整响应策略：
- anxious/worried: 冷静分析，提供明确建议
- excited: 理性提醒风险
- hesitant: 提供明确选项
- impatient: 简洁直接

## 输出
返回路由决策：
- route: "analyze" | "trade" | "explain" | "confirm"
- priority: "high" | "normal"
- emotion_strategy: "reassure" | "caution" | "direct"
""",
            "tools": tools or [],
        },
        {
            "name": "voice-responder",
            "description": """语音响应生成器。
生成适合语音播报的响应，简洁、口语化、有节奏。

适用场景:
- 将分析结果转换为语音响应
- 生成需要确认的操作描述
- 生成开场白和结束语

注意:
- 响应要简洁 (1-3句话)
- 使用口语化表达
- 数字要易于听懂
- 提供明确的下一步选项
""",
            "system_prompt": """你是语音响应生成器。将内容转换为适合语音播报的格式。

## 原则
1. 先结论后解释
2. 数字具体易懂 (1200美元，而不是一千二百美元)
3. 主动提供选项
4. 确认要简洁

## 格式示例
- 止损触发: "紧急提醒，你的{股票}触发止损，亏损{金额}。执行止损还是先分析？"
- 分析结果: "主要是两个原因：第一...第二...建议..."
- 交易确认: "确认{买入/卖出}{数量}股{股票}，约{金额}？"
""",
            "tools": [],
        },
    ]


# ============================================
# Voice Tools for DeepAgent
# ============================================
VOICE_CONTEXT_TOOL_DESCRIPTION = """获取当前语音会话的触发上下文。

返回:
- trigger_type: 触发类型 (stop_loss/price_target/strategy_signal/...)
- symbol: 标的代码
- current_price: 当前价格
- threshold: 触发阈值
- user_holdings: 用户持仓
- market_context: 市场上下文
"""


def create_voice_context_tool():
    """创建获取语音上下文的工具"""
    from langchain_core.tools import StructuredTool

    def get_voice_context(session_id: str) -> dict:
        """获取语音会话上下文

        Args:
            session_id: 语音会话ID

        Returns:
            触发上下文
        """
        # 从 Redis 获取会话上下文
        import redis
        import json
        import os

        r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        data = r.get(f"voice_session:{session_id}")

        if data:
            return json.loads(data)
        return {}

    return StructuredTool.from_function(
        func=get_voice_context,
        name="get_voice_context",
        description=VOICE_CONTEXT_TOOL_DESCRIPTION
    )
```

### 3.5 Voice Gateway: 核心实现

**文件**: `voice_gateway/server.py`

```python
"""Voice Gateway Server - 语音网关服务

提供:
- WebSocket 语音对话端点
- REST API 呼叫管理端点
- Fun-Audio-Chat 集成
"""

import asyncio
import json
import logging
import os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import redis.asyncio as redis

from voice_gateway.bridge import VoiceAgentBridge
from voice_gateway.session_manager import VoiceSessionManager

logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================
FUN_AUDIO_CHAT_URL = os.getenv("FUN_AUDIO_CHAT_URL", "ws://194.68.245.6:22035/api/chat")
DEEPAGENT_URL = os.getenv("DEEPAGENT_URL", "http://localhost:8000")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# ============================================
# Pydantic Models
# ============================================
class InitiateCallRequest(BaseModel):
    user_id: str
    trigger_context: dict
    notification_options: dict


class AcceptCallRequest(BaseModel):
    call_id: str


# ============================================
# Application
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.redis = await redis.from_url(REDIS_URL)
    app.state.session_manager = VoiceSessionManager(app.state.redis)
    app.state.bridge = VoiceAgentBridge(DEEPAGENT_URL)
    logger.info("Voice Gateway started")
    yield
    # Shutdown
    await app.state.redis.close()
    await app.state.bridge.close()
    logger.info("Voice Gateway stopped")


app = FastAPI(title="Voice Gateway", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# WebSocket Endpoint
# ============================================
@app.websocket("/api/chat")
async def voice_chat(websocket: WebSocket):
    """语音对话 WebSocket 端点"""
    await websocket.accept()

    # 获取参数
    user_id = websocket.query_params.get("user_id")
    session_id = websocket.query_params.get("session_id")

    if not user_id:
        await websocket.close(code=4001, reason="Missing user_id")
        return

    session_manager: VoiceSessionManager = app.state.session_manager
    bridge: VoiceAgentBridge = app.state.bridge

    # 获取或创建会话
    session = await session_manager.get_or_create(
        session_id=session_id,
        user_id=user_id
    )

    logger.info(f"Voice session started: {session.session_id}")

    # 连接到 Fun-Audio-Chat
    async with httpx.AsyncClient() as http:
        fun_audio_ws = await asyncio.open_connection(
            "194.68.245.6", 22035
        )

    try:
        # 如果有触发上下文，发送开场白
        if session.trigger_context:
            opening = bridge.generate_opening_message(session.trigger_context)
            # 通过 Fun-Audio-Chat 生成语音并发送
            await send_text_as_speech(websocket, opening, session)

        # 发送 listening 状态
        await websocket.send_json({"type": "listening"})

        # 主循环
        while True:
            try:
                message = await websocket.receive()

                if message["type"] == "websocket.disconnect":
                    break

                if "bytes" in message:
                    # 二进制消息 (音频)
                    await handle_audio_message(
                        websocket, message["bytes"], session, bridge
                    )
                elif "text" in message:
                    # JSON 控制消息
                    await handle_control_message(
                        websocket, message["text"], session
                    )

            except WebSocketDisconnect:
                break

    finally:
        # 保存会话状态
        await session_manager.save(session)
        logger.info(f"Voice session ended: {session.session_id}")


async def handle_audio_message(
    websocket: WebSocket,
    data: bytes,
    session,
    bridge: VoiceAgentBridge
):
    """处理音频消息"""
    msg_type = data[0] if data else 0
    payload = data[1:] if len(data) > 1 else b""

    if msg_type == 0x01:  # Audio
        # 转发给 Fun-Audio-Chat
        # Fun-Audio-Chat 会返回语义理解结果
        semantic_result = await bridge.process_audio(payload, session)

        if semantic_result:
            # 发送给 DeepAgent 处理
            await websocket.send_json({"type": "processing"})

            async for response_chunk in bridge.invoke_agent(
                query=semantic_result.text,
                intent=semantic_result.intent,
                emotion=semantic_result.emotion,
                session=session
            ):
                # 流式发送响应
                if response_chunk.type == "text":
                    await websocket.send_bytes(
                        b"\x02" + response_chunk.content.encode("utf-8")
                    )
                elif response_chunk.type == "audio":
                    await websocket.send_bytes(
                        b"\x01" + response_chunk.content
                    )

            await websocket.send_json({"type": "endTurn"})
            await websocket.send_json({"type": "listening"})


async def handle_control_message(
    websocket: WebSocket,
    data: str,
    session
):
    """处理控制消息"""
    try:
        msg = json.loads(data)
        msg_type = msg.get("type")

        if msg_type == "start":
            session.is_recording = True
        elif msg_type == "pause":
            session.is_recording = False
        elif msg_type == "interrupt":
            # 打断当前生成
            session.interrupted = True
        elif msg_type == "confirm":
            # 确认操作
            action = msg.get("action")
            await handle_confirmation(websocket, action, session)
        elif msg_type == "cancel":
            # 取消操作
            session.pending_action = None

    except json.JSONDecodeError:
        pass


# ============================================
# REST Endpoints
# ============================================
@app.post("/api/call/initiate")
async def initiate_call(request: InitiateCallRequest):
    """发起语音呼叫"""
    session_manager: VoiceSessionManager = app.state.session_manager

    # 创建会话
    session = await session_manager.create(
        user_id=request.user_id,
        trigger_context=request.trigger_context
    )

    # 发送 Push 通知
    await send_call_notification(
        user_id=request.user_id,
        call_id=session.call_id,
        options=request.notification_options
    )

    return {
        "success": True,
        "data": {
            "call_id": session.call_id,
            "session_id": session.session_id,
            "status": "pending",
            "expires_at": session.expires_at.isoformat()
        }
    }


@app.post("/api/call/accept")
async def accept_call(request: AcceptCallRequest):
    """接听呼叫"""
    session_manager: VoiceSessionManager = app.state.session_manager
    bridge: VoiceAgentBridge = app.state.bridge

    session = await session_manager.get_by_call_id(request.call_id)
    if not session:
        raise HTTPException(status_code=404, detail="Call not found or expired")

    session.status = "accepted"
    await session_manager.save(session)

    # 生成开场白
    opening = bridge.generate_opening_message(session.trigger_context)

    return {
        "success": True,
        "data": {
            "session_id": session.session_id,
            "websocket_url": f"wss://{os.getenv('VOICE_GATEWAY_HOST', 'localhost')}:8100/api/chat?session_id={session.session_id}",
            "opening_message": opening
        }
    }


@app.get("/api/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "fun_audio_chat": FUN_AUDIO_CHAT_URL,
        "deepagent": DEEPAGENT_URL
    }


# ============================================
# Helper Functions
# ============================================
async def send_call_notification(user_id: str, call_id: str, options: dict):
    """发送呼叫通知"""
    # 通过极光推送发送
    from xplatform.services.jpush_service import get_jpush_service

    jpush = get_jpush_service()
    if jpush.is_available:
        jpush.send_to_user(
            user_id=user_id,
            title=options.get("title", "语音来电"),
            content=options.get("body", "Vibe Finance 有紧急消息"),
            extras={
                "type": "voice_call",
                "call_id": call_id,
                "action": "accept_call"
            },
            sound=options.get("sound", "default")
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
```

### 3.6 Voice Gateway: Agent Bridge

**文件**: `voice_gateway/bridge.py`

```python
"""Voice Agent Bridge - 语音与 DeepAgent 桥接

职责:
- 语音语义转结构化输入
- DeepAgent 输出转语音响应
- 流式处理
- 会话状态管理
"""

import logging
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SemanticResult:
    """语义理解结果"""
    text: str
    intent: Optional[str] = None
    entities: Optional[dict] = None
    emotion: Optional[str] = None


@dataclass
class ResponseChunk:
    """响应块"""
    type: str  # "text" | "audio"
    content: bytes | str


class VoiceAgentBridge:
    """语音-Agent 桥接器"""

    def __init__(self, deepagent_url: str):
        self.deepagent_url = deepagent_url
        self.http_client = httpx.AsyncClient(timeout=60.0)

    async def process_audio(
        self,
        audio_data: bytes,
        session
    ) -> Optional[SemanticResult]:
        """处理音频数据，返回语义理解结果

        实际实现中，这里会调用 Fun-Audio-Chat 的语义理解能力。
        Fun-Audio-Chat 是端到端模型，会直接返回理解结果。
        """
        # TODO: 实际实现需要与 Fun-Audio-Chat 交互
        # 这里假设已经获得了语义结果
        return SemanticResult(
            text="为什么跌这么多",
            intent="analyze",
            emotion="anxious"
        )

    async def invoke_agent(
        self,
        query: str,
        intent: str,
        emotion: str,
        session
    ) -> AsyncIterator[ResponseChunk]:
        """调用 DeepAgent 并流式返回结果"""

        # 构造 Agent 输入
        agent_input = {
            "user_id": session.user_id,
            "thread_id": f"voice_{session.session_id}",
            "query": query,
            "intent": intent,
            "emotion": emotion,
            "context": {
                "trigger_context": session.trigger_context,
                "conversation_history": session.conversation_history
            }
        }

        # 调用 DeepAgent (SSE 流式)
        async with self.http_client.stream(
            "POST",
            f"{self.deepagent_url}/api/agent/invoke",
            json=agent_input,
            headers={"Accept": "text/event-stream"}
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    import json
                    chunk = json.loads(data)

                    if chunk.get("type") == "text":
                        yield ResponseChunk(
                            type="text",
                            content=chunk["content"]
                        )
                    elif chunk.get("type") == "tool_result":
                        # 工具结果可以转换为语音播报
                        summary = self.summarize_tool_result(chunk["output"])
                        if summary:
                            yield ResponseChunk(type="text", content=summary)

        # 更新会话历史
        session.conversation_history.append({
            "role": "user",
            "content": query
        })

    def generate_opening_message(self, trigger_context: dict) -> str:
        """根据触发上下文生成开场白"""
        trigger_type = trigger_context.get("trigger_type")
        symbol = trigger_context.get("symbol_name", trigger_context.get("symbol", ""))
        current_price = trigger_context.get("current_price")
        threshold = trigger_context.get("threshold")
        user_pnl = trigger_context.get("user_pnl", 0)

        if trigger_type == "stop_loss":
            pnl_text = f"亏损{abs(user_pnl):.0f}美元" if user_pnl else ""
            return (
                f"紧急提醒，你的{symbol}持仓触发了止损，"
                f"{pnl_text}。"
                f"要执行止损还是先分析一下？"
            )

        elif trigger_type == "price_target":
            return (
                f"好消息！{symbol}突破了你设置的目标价{threshold}，"
                f"现在{current_price}。"
                f"要我分析一下接下来怎么操作吗？"
            )

        elif trigger_type == "strategy_signal":
            return (
                f"你的策略刚触发了信号，标的是{symbol}，现价{current_price}。"
                f"要执行吗？"
            )

        elif trigger_type == "option_expiry":
            return (
                f"提醒你，{symbol}的期权今天到期。"
                f"需要我帮你分析一下是行权还是放弃吗？"
            )

        else:
            content = trigger_context.get("content", "有一条消息")
            return f"{content}。要详细了解吗？"

    def summarize_tool_result(self, output: dict) -> Optional[str]:
        """将工具结果转换为语音播报文本"""
        # 简化工具结果为口语化描述
        if "price" in output:
            return f"当前价格{output['price']}，涨跌{output.get('change', 0):.1%}"
        if "analysis" in output:
            return output["analysis"][:100]
        return None

    async def close(self):
        await self.http_client.aclose()
```

---

## 四、部署配置

### 4.1 环境变量

```bash
# Voice Gateway
VOICE_GATEWAY_HOST=voice.vibe.finance
VOICE_GATEWAY_PORT=8100
FUN_AUDIO_CHAT_URL=ws://194.68.245.6:22035/api/chat
DEEPAGENT_URL=http://localhost:8000
REDIS_URL=redis://localhost:6379/0

# xplatform
VOICE_GATEWAY_URL=http://localhost:8100

# DeepAgent
VOICE_SUBAGENT_ENABLED=true
```

### 4.2 Docker Compose

```yaml
version: '3.8'

services:
  voice-gateway:
    build: ./voice_gateway
    ports:
      - "8100:8100"
    environment:
      - FUN_AUDIO_CHAT_URL=ws://194.68.245.6:22035/api/chat
      - DEEPAGENT_URL=http://deepagent:8000
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - deepagent

  deepagent:
    build: ./deepagents
    ports:
      - "8000:8000"
    environment:
      - VOICE_SUBAGENT_ENABLED=true
      - REDIS_URL=redis://redis:6379/0

  xplatform:
    build: ./xplatform
    ports:
      - "8080:8080"
    environment:
      - VOICE_GATEWAY_URL=http://voice-gateway:8100
      - CELERY_BROKER_URL=redis://redis:6379/1
    depends_on:
      - redis
      - voice-gateway

  celery-worker:
    build: ./xplatform
    command: celery -A xplatform.tasks.reminder_tasks worker -l info
    environment:
      - VOICE_GATEWAY_URL=http://voice-gateway:8100
      - CELERY_BROKER_URL=redis://redis:6379/1
    depends_on:
      - redis
      - voice-gateway

  celery-beat:
    build: ./xplatform
    command: celery -A xplatform.tasks.reminder_tasks beat -l info
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/1
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

---

## 五、接入步骤

### 5.1 Phase 1: Voice Gateway (Week 1)

```
□ 创建 voice_gateway/ 目录结构
□ 实现 server.py (FastAPI + WebSocket)
□ 实现 session_manager.py (Redis)
□ 实现 bridge.py (Fun-Audio-Chat 客户端)
□ 测试 WebSocket 连接
□ 测试与 Fun-Audio-Chat 通信
```

### 5.2 Phase 2: xplatform 集成 (Week 2)

```
□ 添加 VoiceCallNotifier 到 notification_channels.py
□ 创建 voice_call_service.py
□ 修改 reminder_tasks.py 添加语音路由
□ 添加 voice_call 通知渠道枚举
□ 测试 Celery 触发语音呼叫
```

### 5.3 Phase 3: DeepAgent 集成 (Week 2)

```
□ 创建 middleware/voice_subagent.py
□ 注册 voice SubAgent
□ 添加 voice_context 工具
□ 测试 Router 调用 voice SubAgent
□ 测试流式响应
```

### 5.4 Phase 4: 端到端测试 (Week 3)

```
□ 止损触发 → 语音呼叫完整流程
□ 用户接听 → 对话 → 交易确认
□ 打断/取消流程
□ 错误处理和降级
```

---

## 六、文件清单

### 新增文件

```
voice_gateway/
├── __init__.py
├── server.py               # FastAPI 服务
├── bridge.py               # Agent 桥接
├── session_manager.py      # 会话管理
└── requirements.txt

deepagents/middleware/
└── voice_subagent.py       # 语音 SubAgent

xplatform/services/
└── voice_call_service.py   # 语音呼叫服务
```

### 修改文件

```
xplatform/services/notification_channels.py
  + VoiceCallNotifier
  + NotificationDispatcher 注册

xplatform/tasks/reminder_tasks.py
  + should_use_voice_call()
  + 修改触发逻辑

deepagents/factory.py
  + 注册 voice SubAgent
```

---

*Fun-Audio-Chat 接入方案 v1.0 | 2026-01-07*
