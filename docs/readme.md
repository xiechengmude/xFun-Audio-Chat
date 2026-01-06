新建文件

  | 文件                              | 功能             |
  |-----------------------------------|------------------|
  | .claude/commands/runpod-deploy.md | SKILL 定义文件   |
  | scripts/auto_deploy.py            | 全自动部署脚本   |
  | scripts/test_deployment.py        | 部署验证测试脚本 |

  使用方式

  1. 一键部署（完全自动化）

  # 自动购买 A40 → 部署 → 启动 → 验证
  python3 scripts/auto_deploy.py --gpu A40

  # 带测试
  python3 scripts/auto_deploy.py --gpu A40 --test

  2. 查看状态

  python3 scripts/auto_deploy.py --status

  3. Claude Code 中使用

  /runpod-deploy --gpu A40

  自动化流程

  Phase 1: 创建 Pod (RunPod API)
      ↓
  Phase 2: 等待 Pod 就绪 (轮询状态)
      ↓
  Phase 3: SSH 连接就绪
      ↓
  Phase 4: 环境配置
      - apt install ffmpeg
      - pip install pytorch + cuda
      - pip install requirements
      - 下载模型 (如果不存在)
      ↓
  Phase 5: 启动 Server
      - nohup python3 web_demo.server.server
      ↓
  Phase 6: 验证部署
      - 检查端口监听
      - 检查模型加载日志

  当前状态

  Pod: increased_green_cow (45772b7uimaxg3)
  IP: 194.68.245.123
  API: ws://194.68.245.123:22160/api/chat
  状态: RUNNING ✓
  