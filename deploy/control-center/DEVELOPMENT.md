# PDF-AI 开发与部署指南

> Mac 本地开发 + 远程服务器部署完整指南

## 目录

- [架构概览](#架构概览)
- [Mac 本地开发](#mac-本地开发)
- [远程服务器部署](#远程服务器部署)
- [部署流程详解](#部署流程详解)
- [常见问题](#常见问题)

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        完整部署架构                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐      ┌─────────────────┐      ┌─────────────────────┐  │
│  │  Mac 本地   │      │   远程服务器     │      │   RunPod GPU Pod    │  │
│  │  开发环境   │─────▶│   控制中心       │─────▶│   PDF-AI 服务       │  │
│  └─────────────┘      └─────────────────┘      └─────────────────────┘  │
│        │                      │                         │               │
│        │                      │                         │               │
│   - 代码编辑            - Docker Compose          - vLLM Server         │
│   - 本地测试            - FastAPI Gateway         - PDF API Server      │
│   - Git 操作            - 健康监控                - LightOnOCR-2-1B     │
│                         - 部署管理                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Mac 本地开发

### 环境准备

#### 1. 安装依赖

```bash
# 安装 Homebrew (如果没有)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 Python 3.11+
brew install python@3.11

# 安装 Docker Desktop
brew install --cask docker

# 启动 Docker Desktop
open -a Docker
```

#### 2. 克隆项目

```bash
git clone https://github.com/xiechengmude/xFun-Audio-Chat.git
cd xFun-Audio-Chat
```

#### 3. 创建虚拟环境

```bash
# 创建虚拟环境
python3.11 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install -r deploy/control-center/requirements.txt
```

### 本地开发控制中心

#### 方式一：直接运行 (无 Docker)

```bash
cd deploy/control-center

# 设置环境变量
export RUNPOD_API_KEY="your_api_key"
export STATE_DIR="./state"

# 运行 API 服务器
python -m services.api_server
```

服务启动后访问: http://localhost:8080

#### 方式二：Docker Compose

```bash
cd deploy/control-center

# 配置环境变量
cp .env.example .env
# 编辑 .env 设置 RUNPOD_API_KEY

# 启动服务
docker-compose up --build

# 或后台运行
docker-compose up -d --build
```

### 本地测试

#### 测试 API 端点

```bash
# 健康检查
curl http://localhost:8080/health

# 获取服务信息
curl http://localhost:8080/api/info

# 查看 GPU 配置
curl http://localhost:8080/api/info | jq '.available_gpus'
```

#### 使用 CLI 工具

```bash
cd deploy/control-center

# 设置环境变量
export RUNPOD_API_KEY="your_api_key"

# 查看可用 GPU
python -m services.cli gpus

# 列出现有 Pods
python -m services.cli pods

# 查看部署状态
python -m services.cli status
```

### 代码结构

```
deploy/control-center/
├── services/                 # 核心服务
│   ├── api_server.py        # FastAPI 网关
│   ├── deploy_manager.py    # 部署编排
│   ├── health_monitor.py    # 健康监控
│   └── cli.py               # CLI 工具
├── utils/                    # 工具类
│   ├── runpod_client.py     # RunPod API
│   ├── ssh_client.py        # SSH 操作
│   └── test_runner.py       # E2E 测试
├── templates/                # 模板
│   └── setup_commands.sh    # 服务器设置脚本
├── Dockerfile               # 容器镜像
├── docker-compose.yml       # 编排配置
├── requirements.txt         # Python 依赖
├── start.sh                 # 快速启动
└── deploy-to-server.sh      # 远程部署
```

---

## 远程服务器部署

### 前提条件

1. **远程服务器**：Linux (Ubuntu 20.04+)，至少 2GB RAM
2. **SSH 访问**：能够 SSH 到远程服务器
3. **RunPod 账号**：有 API Key 和余额

### 部署方式

#### 方式一：一键部署脚本（推荐）

```bash
cd deploy/control-center

# 一键部署到远程服务器
./deploy-to-server.sh root@your-server-ip --runpod-key YOUR_RUNPOD_API_KEY
```

脚本会自动：
1. ✅ 检测并安装 Docker
2. ✅ 上传控制中心代码
3. ✅ 配置环境变量
4. ✅ 启动 Docker Compose

#### 方式二：手动部署

**步骤 1: SSH 到远程服务器**
```bash
ssh root@your-server-ip
```

**步骤 2: 安装 Docker**
```bash
curl -fsSL https://get.docker.com | sh
systemctl enable docker
systemctl start docker
```

**步骤 3: 上传代码**
```bash
# 在本地执行
scp -r deploy/control-center root@your-server-ip:/opt/pdf-ai/
```

**步骤 4: 配置并启动**
```bash
# 在远程服务器执行
cd /opt/pdf-ai
cp .env.example .env
echo "RUNPOD_API_KEY=your_key" >> .env
docker-compose up -d
```

### 触发 GPU 部署

控制中心启动后，通过 API 触发 RunPod 部署：

```bash
# 部署到 A40 GPU
curl -X POST http://your-server-ip:8080/api/deploy \
    -H "Content-Type: application/json" \
    -d '{"gpu_type": "A40", "run_benchmark": false}'

# 部署到 A40 并运行基准测试
curl -X POST http://your-server-ip:8080/api/deploy \
    -H "Content-Type: application/json" \
    -d '{"gpu_type": "A40", "run_benchmark": true}'

# 部署到 H100 (高性能)
curl -X POST http://your-server-ip:8080/api/deploy \
    -H "Content-Type: application/json" \
    -d '{"gpu_type": "H100", "run_benchmark": true}'
```

### 监控部署进度

```bash
# 查看部署状态
curl http://your-server-ip:8080/api/status | jq

# 查看部署日志
curl http://your-server-ip:8080/api/logs

# 查看所有 Pods
curl http://your-server-ip:8080/api/pods
```

---

## 部署流程详解

### 完整流程图

```
┌──────────────────────────────────────────────────────────────────┐
│                     部署流程 (约 5-10 分钟)                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 1: Creating Pod                                            │
│  ├─ 调用 RunPod API 创建 Pod                                      │
│  └─ 分配 GPU 资源 (A40/A100/H100)                                 │
│                                                                   │
│  Phase 2: Waiting for Pod                                         │
│  ├─ 等待 Pod 状态变为 RUNNING                                     │
│  └─ 获取 SSH 连接信息 (IP + Port)                                 │
│                                                                   │
│  Phase 3: SSH Connect                                             │
│  └─ 建立 SSH 连接 (默认超时 120s)                                 │
│                                                                   │
│  Phase 4: Setup Environment                                       │
│  ├─ 安装系统依赖 (poppler-utils)                                  │
│  ├─ 克隆仓库                                                      │
│  ├─ 安装 Python 依赖 (vllm, pypdfium2)                            │
│  └─ 下载模型 (LightOnOCR-2-1B, ~5GB)                              │
│                                                                   │
│  Phase 5: Start vLLM                                              │
│  ├─ 清理残留进程                                                  │
│  └─ 启动 vLLM Server (Port 8000)                                  │
│                                                                   │
│  Phase 6: Wait vLLM Ready                                         │
│  ├─ 轮询 /health 端点 (超时 420s)                                 │
│  └─ 检测 OOM 错误                                                 │
│                                                                   │
│  Phase 7: Start PDF API                                           │
│  └─ 启动 PDF API Server (Port 8006)                               │
│                                                                   │
│  Phase 8: E2E Testing                                             │
│  ├─ vLLM 健康检查                                                 │
│  ├─ PDF API 健康检查                                              │
│  ├─ 单页 PDF 解析测试                                             │
│  └─ 批量 PDF 解析测试                                             │
│                                                                   │
│  Phase 9: Benchmark (可选)                                        │
│  └─ 运行性能基准测试                                              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 部署状态码

| Phase | 状态 | 描述 |
|-------|------|------|
| `idle` | 空闲 | 未启动部署 |
| `creating_pod` | 进行中 | 正在创建 Pod |
| `waiting_pod` | 进行中 | 等待 Pod 就绪 |
| `ssh_connect` | 进行中 | 建立 SSH |
| `setup_env` | 进行中 | 配置环境 |
| `start_vllm` | 进行中 | 启动 vLLM |
| `wait_vllm` | 进行中 | 等待 vLLM |
| `start_api` | 进行中 | 启动 API |
| `wait_api` | 进行中 | 等待 API |
| `testing` | 进行中 | E2E 测试 |
| `benchmark` | 进行中 | 性能测试 |
| `completed` | ✅ 成功 | 部署完成 |
| `failed` | ❌ 失败 | 部署失败 |

---

## 常见问题

### 1. vLLM OOM 错误

**症状**: 部署卡在 `wait_vllm` 阶段，日志显示 CUDA OOM

**原因**: GPU 显存不足

**解决**:
- 使用更大显存的 GPU (A40 48GB → A100 80GB)
- 脚本已自动配置 `--gpu-memory-utilization 0.85`

### 2. SSH 连接超时

**症状**: 部署卡在 `ssh_connect` 阶段

**原因**: Pod 未完全启动或网络问题

**解决**:
```bash
# 检查 Pod 状态
curl http://your-server:8080/api/pods

# 手动测试 SSH
ssh -p <ssh_port> root@<pod_ip>
```

### 3. 模型下载慢

**症状**: `setup_env` 阶段耗时很长

**原因**: HuggingFace 模型下载 (~5GB)

**解决**: 首次部署需要下载，后续会使用缓存

### 4. PDF 解析失败

**症状**: E2E 测试失败

**检查**:
```bash
# 查看 PDF API 日志
curl http://your-server:8080/api/logs

# 直接测试 PDF API
curl -X POST http://<pod_ip>:8006/api/parse \
    -F "file=@test.pdf"
```

### 5. 清理资源

```bash
# 终止指定 Pod
curl -X DELETE http://your-server:8080/api/pods/<pod_id>

# 停止控制中心
cd deploy/control-center
docker-compose down
```

---

## 开发工作流

### 推荐流程

```
1. Mac 本地开发
   └─ 修改代码 → 本地测试 → Git 提交

2. 部署到远程服务器
   └─ ./deploy-to-server.sh → 验证控制中心

3. 触发 GPU 部署
   └─ curl /api/deploy → 等待完成 → 验证服务

4. 调用 PDF API
   └─ curl http://<pod_ip>:8006/api/parse
```

### Git 提交规范

```bash
# 功能
git commit -m "feat(deploy): add new GPU support"

# 修复
git commit -m "fix(vllm): handle OOM gracefully"

# 文档
git commit -m "docs: update deployment guide"

# 重构
git commit -m "refactor(api): simplify health check"
```

---

## 联系与支持

- **仓库**: https://github.com/xiechengmude/xFun-Audio-Chat
- **问题**: GitHub Issues
- **RunPod**: https://runpod.io
