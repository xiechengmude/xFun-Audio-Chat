 现有服务架构

  ┌─────────────────────────────────────────────────────────────────┐
  │                    xplatform (基础服务)                          │
  │  ├─ 用户认证 (JWT/Supabase)                                     │
  │  ├─ 智能提醒 (reminder_tasks + Celery)                          │
  │  ├─ 通知渠道 (notification_channels)                            │
  │  └─ Voice Gateway (刚实现) ←─────────────────────┐               │
  └─────────────────────────────────────────────────┼───────────────┘
                                                    │
  ┌─────────────────────────────────────────────────┼───────────────┐
  │              xdan-vibe-finance-strategy                         │
  │                   (LangGraph Agent)                              │
  │  ┌──────────────────────────────────────────────┼────────────┐  │
  │  │              factory.py:get_agent()          ↓            │  │
  │  │  ┌─────────────────────────────────────────────────────┐  │  │
  │  │  │                   LeadAgent                         │  │  │
  │  │  │  ├─ smart_search (知识增强)                          │  │  │
  │  │  │  ├─ explorer (API/SKILL 发现)                        │  │  │
  │  │  │  ├─ finance_executor (金融分析)                      │  │  │
  │  │  │  └─ strategy_executor (策略回测)                     │  │  │
  │  │  └─────────────────────────────────────────────────────┘  │  │
  │  └───────────────────────────────────────────────────────────┘  │
  │                              ↓                                   │
  │                    LangGraph Server API                          │
  │                POST /threads/{id}/runs/stream                    │
  └──────────────────────────────────────────────────────────────────┘

  全双工语音 Agent 集成方案

⏺ 创建 LangGraph Client，让 Voice Gateway 可以调用 Strategy Agent：

✻ 创建 LangGraph Client… (esc to interrupt · ctrl+t to hide todos · 2m 39s · ↓ 2.5k tokens · thinking)
  ⎿  ☐ 创建 LangGraph Client (voice_gateway 调用 strategy agent)
     ☐ 实现 Voice-Agent Bridge (语音 ↔ LangGraph 消息转换)
     ☐ 实现 Context Manager (多轮对话 + TriggerContext)
     ☐ 实现 Confirmation Handler (敏感操作确认)
