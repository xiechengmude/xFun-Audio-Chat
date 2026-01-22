# PDF-AI 控制中心部署方案

> **版本**: v1.0
> **架构**: Docker Compose + RunPod GPU 云

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Remote Server (Control Center)                      │
│                  Docker Compose Stack                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ API Gateway  │  │   Deploy     │  │   Health     │              │
│  │ (FastAPI)    │  │   Manager    │  │   Monitor    │              │
│  │ Port: 8080   │  │ (Background) │  │ (Background) │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                       │
│         ▼                 ▼                 ▼                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Shared Volume                             │   │
│  │  /data: SSH Keys, Configs, Logs, State                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               │ RunPod API + SSH
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RunPod GPU Pod (Dynamic)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────┐    ┌────────────────────────────────────┐  │
│  │    vLLM Server     │    │      PDF API Server                │  │
│  │  LightOnOCR-2-1B   │───▶│      /api/parse                    │  │
│  │  Port: 8000        │    │      Port: 8006                    │  │
│  └────────────────────┘    └────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 部署流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Step 1    │───▶│   Step 2    │───▶│   Step 3    │───▶│   Step 4    │
│ 启动控制中心 │    │ 购买GPU Pod │    │ 部署服务    │    │ 验证测试    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
docker-compose      RunPod API        SSH + Setup        API Tests
    up -d           create pod       vLLM + PDF API    /health check
```

## 快速开始

### 方式一：一键部署到远程服务器（推荐）

```bash
cd deploy/control-center

# 部署到远程服务器（会自动安装 Docker）
./deploy-to-server.sh root@your-server-ip --runpod-key your_api_key

# 然后通过 API 触发 GPU 部署
curl -X POST http://your-server-ip:8080/api/deploy \
    -H "Content-Type: application/json" \
    -d '{"gpu_type": "A40", "run_benchmark": true}'
```

### 方式二：本地启动控制中心

#### 1. 配置环境变量

```bash
cd deploy/control-center
cp .env.example .env
# 编辑 .env 填入 RunPod API Key
```

#### 2. 使用快速启动脚本

```bash
# 启动并部署到 A40
./start.sh --gpu A40

# 启动并运行基准测试
./start.sh --gpu A40 --benchmark

# 查看状态
./start.sh --status

# 查看日志
./start.sh --logs

# 停止服务
./start.sh --stop
```

#### 3. 或手动使用 Docker Compose

```bash
docker-compose up -d

# 触发部署
curl -X POST http://localhost:8080/api/deploy \
    -H "Content-Type: application/json" \
    -d '{"gpu_type": "A40", "run_benchmark": true}'
```

### 查看状态

```bash
curl http://localhost:8080/api/status
```

## 完整部署流程

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Your Machine    │────▶│  Remote Server   │────▶│  RunPod GPU Pod  │
│  (Local)         │     │  (Control Center)│     │  (PDF-AI Service)│
└──────────────────┘     └──────────────────┘     └──────────────────┘
        │                         │                        │
        │ deploy-to-server.sh     │ RunPod API             │
        │                         │ SSH                    │
        ▼                         ▼                        ▼
   1. 安装 Docker           2. 创建 Pod              4. 运行测试
   2. 上传代码              3. 配置环境              5. 返回结果
   3. 启动控制中心           4. 启动服务
```

### 部署成功判定

当以下条件全部满足时，部署视为成功：

1. ✅ RunPod Pod 状态为 RUNNING
2. ✅ vLLM Server 健康检查通过
3. ✅ PDF API Server 健康检查通过
4. ✅ 单页 PDF 解析测试通过
5. ✅ 批量 PDF 解析测试通过

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/deploy` | POST | 触发新部署 |
| `/api/status` | GET | 获取当前状态 |
| `/api/pods` | GET | 列出所有 Pods |
| `/api/pods/{pod_id}` | DELETE | 终止指定 Pod |
| `/api/test` | POST | 运行端到端测试 |
| `/api/benchmark` | POST | 运行性能基准测试 |
| `/health` | GET | 控制中心健康检查 |

## 目录结构

```
deploy/control-center/
├── Dockerfile              # 控制中心镜像
├── docker-compose.yml      # 编排配置
├── .env.example            # 环境变量模板
├── requirements.txt        # Python 依赖
├── services/
│   ├── __init__.py
│   ├── api_server.py       # FastAPI 网关
│   ├── deploy_manager.py   # 部署管理器
│   ├── health_monitor.py   # 健康监控
│   └── cli.py              # CLI 工具
├── utils/
│   ├── __init__.py
│   ├── runpod_client.py    # RunPod API 封装
│   ├── ssh_client.py       # SSH 操作封装
│   └── test_runner.py      # 测试执行器
└── templates/
    └── setup_commands.sh   # 服务器设置脚本模板
```

## 配置说明

### 环境变量

| 变量 | 必需 | 描述 |
|------|------|------|
| `RUNPOD_API_KEY` | ✅ | RunPod API 密钥 |
| `DEFAULT_GPU` | ❌ | 默认 GPU 类型 (A40) |
| `VLLM_PORT` | ❌ | vLLM 端口 (8000) |
| `API_PORT` | ❌ | PDF API 端口 (8006) |
| `SSH_KEY_PATH` | ❌ | SSH 私钥路径 |

### GPU 类型支持

| GPU | 显存 | 推荐场景 |
|-----|------|----------|
| H100 | 80GB | 高吞吐生产环境 |
| A100 | 80GB | 生产环境 |
| A40 | 48GB | 开发测试 |
| RTX 4090 | 24GB | 轻量测试 |

## 部署验证

部署成功的标准：

1. ✅ vLLM Server 响应 `/health`
2. ✅ PDF API Server 响应 `/health`
3. ✅ 单页 PDF 解析测试通过
4. ✅ 批量 PDF 解析测试通过
5. ✅ 端到端延迟 < 10s/page

## 故障排查

### 查看日志

```bash
# 控制中心日志
docker-compose logs -f

# 部署管理器日志
docker-compose logs -f deploy-manager

# RunPod Pod 日志
docker-compose exec deploy-manager cat /data/logs/deploy.log
```

### 常见问题

1. **vLLM OOM**: 自动使用 `--gpu-memory-utilization 0.85`
2. **SSH 连接失败**: 检查 SSH Key 和 Pod 状态
3. **模型加载慢**: 首次需要下载 ~5GB 模型
