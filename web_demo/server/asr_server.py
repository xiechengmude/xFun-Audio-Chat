# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ASR Server for Fun-ASR-Nano-2512
Provides standalone speech-to-text API service with concurrent request support

Features:
- Async HTTP API with aiohttp
- Thread pool for non-blocking inference
- Dynamic batching support
- 100+ concurrent streams supported
"""

import argparse
import os
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from aiohttp import web
import torch


def log(level: str, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level.upper()}] {message}")


class ASRModelManager:
    """Singleton manager for ASR model with concurrent support"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, model_path: str, device: str = "cuda:0",
                   max_workers: int = 4, batch_size_s: int = 300):
        """
        Initialize ASR model with concurrent support

        Args:
            model_path: Path to ASR model
            device: CUDA device
            max_workers: Max concurrent inference threads
            batch_size_s: Dynamic batch size in seconds (for batch processing)
        """
        if self._initialized:
            log("info", "ASR model already initialized, skipping...")
            return

        self.device = device
        self.model_path = model_path
        self.batch_size_s = batch_size_s

        # Thread pool for non-blocking inference
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        log("info", f"Initialized thread pool with {max_workers} workers")

        log("info", f"Loading ASR model from {model_path} to {device}...")

        try:
            from funasr import AutoModel
            import sys

            # Use local model path with remote_code
            abs_model_path = os.path.abspath(model_path)
            remote_code_path = os.path.join(abs_model_path, "model.py")

            log("info", f"Using local model: {abs_model_path}")
            log("info", f"Remote code: {remote_code_path}")

            # Add model path to sys.path for imports
            sys.path.insert(0, abs_model_path)

            self.model = AutoModel(
                model=abs_model_path,
                trust_remote_code=True,
                remote_code=remote_code_path,
                device=device,
                disable_update=True,
            )
            log("info", "ASR model loaded successfully")
            log("info", f"Concurrent support: {max_workers} workers, batch_size_s={batch_size_s}")
            self._initialized = True

        except ImportError as e:
            log("error", f"Failed to import funasr: {e}")
            log("error", "Please install funasr: pip install funasr")
            raise

        except Exception as e:
            log("error", f"Failed to load ASR model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def transcribe(self, audio_path: str, language: str = "中文",
                   hotwords: list = None, itn: bool = True) -> str:
        """
        Transcribe audio file to text (sync version)

        Args:
            audio_path: Path to audio file
            language: Recognition language (中文/英文/日文)
            hotwords: List of hotwords for better recognition
            itn: Enable inverse text normalization

        Returns:
            Transcribed text
        """
        if not self._initialized:
            raise RuntimeError("ASR model not initialized")

        result = self.model.generate(
            input=[audio_path],
            cache={},
            batch_size_s=self.batch_size_s,  # Dynamic batching by duration
            hotwords=hotwords,
            language=language,
            itn=itn,
        )

        return result[0]["text"] if result else ""

    async def transcribe_async(self, audio_path: str, language: str = "中文",
                               hotwords: list = None, itn: bool = True) -> str:
        """
        Transcribe audio file to text (async version, non-blocking)

        Uses thread pool to avoid blocking the event loop
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.transcribe(audio_path, language, hotwords, itn)
        )


class ASRServer:
    """ASR HTTP Server"""

    def __init__(self, model_manager: ASRModelManager):
        self.model_manager = model_manager

    async def handle_transcribe(self, request):
        """
        POST /api/transcribe - Speech to text

        Request (multipart/form-data):
            - audio: Audio file (wav/mp3/flac/etc)
            - language: Recognition language (default: 中文)
            - hotwords: Comma-separated hotwords (optional)

        Response:
            {
                "text": "transcribed text",
                "language": "中文",
                "success": true
            }
        """
        try:
            # Parse multipart form data
            data = await request.post()
            audio_file = data.get('audio')
            language = data.get('language', '中文')
            hotwords_str = data.get('hotwords', '')

            if audio_file is None:
                return web.json_response(
                    {"error": "No audio file provided", "success": False},
                    status=400
                )

            # Parse hotwords
            hotwords = None
            if hotwords_str:
                hotwords = [h.strip() for h in hotwords_str.split(',') if h.strip()]

            # Save to temporary file
            suffix = '.wav'
            if hasattr(audio_file, 'filename') and audio_file.filename:
                _, ext = os.path.splitext(audio_file.filename)
                if ext:
                    suffix = ext

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                content = audio_file.file.read()
                f.write(content)
                temp_path = f.name

            log("info", f"Received audio: {len(content)} bytes, language={language}")

            try:
                # Perform ASR (async, non-blocking)
                text = await self.model_manager.transcribe_async(
                    audio_path=temp_path,
                    language=language,
                    hotwords=hotwords,
                    itn=True
                )

                log("info", f"Transcription result: {text[:100]}..." if len(text) > 100 else f"Transcription result: {text}")

                return web.json_response({
                    "text": text,
                    "language": language,
                    "success": True
                })

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            log("error", f"Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return web.json_response(
                {"error": str(e), "success": False},
                status=500
            )

    async def handle_health(self, request):
        """
        GET /health - Health check

        Response:
            {
                "status": "healthy",
                "model": "Fun-ASR-Nano-2512",
                "device": "cuda:0"
            }
        """
        return web.json_response({
            "status": "healthy",
            "model": "Fun-ASR-Nano-2512",
            "model_path": self.model_manager.model_path,
            "device": self.model_manager.device
        })

    async def handle_info(self, request):
        """
        GET /api/info - Model information

        Response:
            {
                "model": "Fun-ASR-Nano-2512",
                "supported_languages": ["中文", "英文", "日文"],
                "features": [...]
            }
        """
        return web.json_response({
            "model": "Fun-ASR-Nano-2512",
            "model_path": self.model_manager.model_path,
            "device": self.model_manager.device,
            "supported_languages": ["中文", "英文", "日文"],
            "features": [
                "End-to-end speech recognition",
                "Low-latency real-time transcription",
                "Hotword support",
                "Inverse text normalization (ITN)",
                "7 major Chinese dialects",
                "20+ regional accents"
            ]
        })


def main():
    parser = argparse.ArgumentParser(
        description="Fun-ASR-Nano-2512 Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m web_demo.server.asr_server --port 8003
  python -m web_demo.server.asr_server --model-path ./pretrained_models/Fun-ASR-Nano-2512 --device cuda:1

API Endpoints:
  POST /api/transcribe  - Speech to text
  GET  /health          - Health check
  GET  /api/info        - Model information
        """
    )

    parser.add_argument("--host", default="0.0.0.0", type=str,
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", default=8003, type=int,
                        help="Server port (default: 8003)")
    parser.add_argument("--model-path", type=str,
                        default="pretrained_models/Fun-ASR-Nano-2512",
                        help="Path to ASR model")
    parser.add_argument("--device", default="cuda:0", type=str,
                        help="Device to run model on (default: cuda:0)")
    parser.add_argument("--max-workers", default=10, type=int,
                        help="Max concurrent inference workers (default: 10)")
    parser.add_argument("--batch-size-s", default=300, type=int,
                        help="Dynamic batch size in seconds (default: 300)")

    args = parser.parse_args()

    # Initialize model
    model_manager = ASRModelManager()
    try:
        model_manager.initialize(
            args.model_path,
            args.device,
            max_workers=args.max_workers,
            batch_size_s=args.batch_size_s
        )
    except Exception as e:
        log("error", f"Failed to initialize ASR model: {e}")
        return

    # Create server
    server = ASRServer(model_manager)

    # Setup routes
    app = web.Application(client_max_size=100 * 1024 * 1024)  # 100MB max upload
    app.router.add_post("/api/transcribe", server.handle_transcribe)
    app.router.add_get("/health", server.handle_health)
    app.router.add_get("/api/info", server.handle_info)

    log("info", f"ASR Server starting at http://{args.host}:{args.port}")
    log("info", f"Model: {args.model_path}")
    log("info", f"Device: {args.device}")
    log("info", "")
    log("info", "API Endpoints:")
    log("info", f"  POST http://{args.host}:{args.port}/api/transcribe")
    log("info", f"  GET  http://{args.host}:{args.port}/health")
    log("info", f"  GET  http://{args.host}:{args.port}/api/info")

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
