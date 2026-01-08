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
TTS Server for Fun-CosyVoice3-0.5B-2512
Provides standalone text-to-speech API service with optional vLLM acceleration
"""

import argparse
import os
import sys
import io
import base64
import tempfile
from datetime import datetime
from aiohttp import web
import torch
import torchaudio
import numpy as np

# Add CosyVoice paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../..')
sys.path.insert(0, project_root)

submodule_path = os.path.join(project_root, 'third_party/CosyVoice')
sys.path.insert(0, submodule_path)
matcha_tts_path = os.path.join(project_root, 'third_party/CosyVoice/third_party/Matcha-TTS')
sys.path.insert(0, matcha_tts_path)


def log(level: str, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level.upper()}] {message}")


class TTSModelManager:
    """Singleton manager for TTS model"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, model_path: str, device: str = "cuda:0",
                   use_vllm: bool = False, use_trt: bool = False):
        if self._initialized:
            log("info", "TTS model already initialized, skipping...")
            return

        self.device = device
        self.model_path = model_path
        self.use_vllm = use_vllm
        self.sample_rate = 24000

        log("info", f"Loading TTS model from {model_path} to {device}...")
        log("info", f"vLLM: {use_vllm}, TensorRT: {use_trt}")

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice3

            # Set CUDA device
            if 'cuda' in device:
                gpu_id = int(device.split(':')[1]) if ':' in device else 0
                torch.cuda.set_device(gpu_id)

            self.model = CosyVoice3(
                model_path,
                load_trt=use_trt,
                load_vllm=use_vllm,
                fp16=False
            )

            # Load default speaker embeddings
            spk_emb_path = os.path.join(project_root, 'utils/new_spk2info.pt')
            if os.path.exists(spk_emb_path):
                self.speaker_embeddings = torch.load(spk_emb_path)
                log("info", f"Loaded speaker embeddings: {list(self.speaker_embeddings.keys())}")
            else:
                self.speaker_embeddings = {}
                log("warning", f"Speaker embeddings not found: {spk_emb_path}")

            log("info", "TTS model loaded successfully")
            self._initialized = True

        except ImportError as e:
            log("error", f"Failed to import CosyVoice: {e}")
            log("error", "Please ensure CosyVoice submodule is initialized")
            raise

        except Exception as e:
            log("error", f"Failed to load TTS model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def synthesize(self, text: str, speaker_id: str = "中文女",
                   prompt_wav: str = None, prompt_text: str = None,
                   stream: bool = False):
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize
            speaker_id: Speaker ID for preset voices
            prompt_wav: Reference audio for zero-shot cloning
            prompt_text: Text corresponding to reference audio
            stream: Whether to return streaming audio

        Yields/Returns:
            Audio data as numpy array
        """
        if not self._initialized:
            raise RuntimeError("TTS model not initialized")

        # Zero-shot mode
        if prompt_wav and prompt_text:
            log("info", f"Zero-shot synthesis: {text[:50]}...")
            for chunk in self.model.inference_zero_shot(
                tts_text=text,
                prompt_text=prompt_text,
                prompt_speech_16k=prompt_wav,
                stream=stream
            ):
                if stream:
                    yield chunk['tts_speech'].cpu().numpy()
                else:
                    return chunk['tts_speech'].cpu().numpy()

        # Preset speaker mode - use inference_sft
        else:
            log("info", f"Preset synthesis: {text[:50]}... (speaker: {speaker_id})")

            # Map speaker_id to spk_id (CosyVoice uses different naming)
            spk_id_map = {
                "中文女": "中文女",
                "中文男": "中文男",
                "英文女": "英文女",
                "英文男": "英文男",
                "日文男": "日语男",
                "粤语女": "粤语女",
                "韩语女": "韩语女",
            }
            spk_id = spk_id_map.get(speaker_id, speaker_id)

            for chunk in self.model.inference_sft(
                tts_text=text,
                spk_id=spk_id,
                stream=stream
            ):
                if stream:
                    yield chunk['tts_speech'].cpu().numpy()
                else:
                    return chunk['tts_speech'].cpu().numpy()


class TTSServer:
    """TTS HTTP Server"""

    def __init__(self, model_manager: TTSModelManager):
        self.model_manager = model_manager

    async def handle_synthesize(self, request):
        """
        POST /api/synthesize - Text to speech

        Request (JSON):
            {
                "text": "要合成的文本",
                "speaker_id": "中文女",
                "prompt_wav": "/path/to/ref.wav",  // optional
                "prompt_text": "参考音频文本",      // optional
                "stream": false
            }

        Response:
            {
                "audio": "<base64_wav>",
                "sample_rate": 24000,
                "duration": 3.5,
                "success": true
            }
        """
        try:
            data = await request.json()
            text = data.get('text')
            speaker_id = data.get('speaker_id', '中文女')
            prompt_wav = data.get('prompt_wav')
            prompt_text = data.get('prompt_text')
            stream = data.get('stream', False)

            if not text:
                return web.json_response(
                    {"error": "No text provided", "success": False},
                    status=400
                )

            log("info", f"Synthesize request: '{text[:50]}...' (stream={stream})")

            # Streaming response
            if stream:
                response = web.StreamResponse(
                    status=200,
                    reason='OK',
                    headers={'Content-Type': 'audio/wav'}
                )
                await response.prepare(request)

                for audio_chunk in self.model_manager.synthesize(
                    text=text,
                    speaker_id=speaker_id,
                    prompt_wav=prompt_wav,
                    prompt_text=prompt_text,
                    stream=True
                ):
                    # Convert to bytes and send
                    audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                    await response.write(audio_bytes)

                await response.write_eof()
                return response

            # Non-streaming response
            else:
                audio_data = None
                for chunk in self.model_manager.synthesize(
                    text=text,
                    speaker_id=speaker_id,
                    prompt_wav=prompt_wav,
                    prompt_text=prompt_text,
                    stream=False
                ):
                    audio_data = chunk
                    break  # Only need first (complete) result

                if audio_data is None:
                    return web.json_response(
                        {"error": "Synthesis failed", "success": False},
                        status=500
                    )

                # Ensure correct shape
                if audio_data.ndim == 1:
                    audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
                else:
                    audio_tensor = torch.from_numpy(audio_data)

                # Save to buffer
                buffer = io.BytesIO()
                torchaudio.save(buffer, audio_tensor, self.model_manager.sample_rate, format='wav')
                buffer.seek(0)
                audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

                duration = audio_data.shape[-1] / self.model_manager.sample_rate

                log("info", f"Synthesis complete: {duration:.2f}s")

                return web.json_response({
                    "audio": audio_base64,
                    "sample_rate": self.model_manager.sample_rate,
                    "duration": duration,
                    "success": True
                })

        except Exception as e:
            log("error", f"Synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return web.json_response(
                {"error": str(e), "success": False},
                status=500
            )

    async def handle_synthesize_file(self, request):
        """
        POST /api/synthesize/file - Return audio file directly

        Same parameters as /api/synthesize, returns WAV file
        """
        try:
            data = await request.json()
            text = data.get('text')
            speaker_id = data.get('speaker_id', '中文女')
            prompt_wav = data.get('prompt_wav')
            prompt_text = data.get('prompt_text')

            if not text:
                return web.json_response(
                    {"error": "No text provided", "success": False},
                    status=400
                )

            log("info", f"Synthesize file request: '{text[:50]}...'")

            audio_data = None
            for chunk in self.model_manager.synthesize(
                text=text,
                speaker_id=speaker_id,
                prompt_wav=prompt_wav,
                prompt_text=prompt_text,
                stream=False
            ):
                audio_data = chunk
                break

            if audio_data is None:
                return web.json_response(
                    {"error": "Synthesis failed", "success": False},
                    status=500
                )

            # Ensure correct shape
            if audio_data.ndim == 1:
                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            else:
                audio_tensor = torch.from_numpy(audio_data)

            # Save to buffer
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio_tensor, self.model_manager.sample_rate, format='wav')
            buffer.seek(0)

            return web.Response(
                body=buffer.read(),
                content_type='audio/wav',
                headers={
                    'Content-Disposition': 'attachment; filename="synthesized.wav"'
                }
            )

        except Exception as e:
            log("error", f"Synthesis file failed: {e}")
            import traceback
            traceback.print_exc()
            return web.json_response(
                {"error": str(e), "success": False},
                status=500
            )

    async def handle_health(self, request):
        """
        GET /health - Health check
        """
        return web.json_response({
            "status": "healthy",
            "model": "Fun-CosyVoice3-0.5B-2512",
            "model_path": self.model_manager.model_path,
            "device": self.model_manager.device,
            "vllm_enabled": self.model_manager.use_vllm,
            "sample_rate": self.model_manager.sample_rate
        })

    async def handle_info(self, request):
        """
        GET /api/info - Model information
        """
        speakers = list(self.model_manager.speaker_embeddings.keys()) if self.model_manager.speaker_embeddings else []

        return web.json_response({
            "model": "Fun-CosyVoice3-0.5B-2512",
            "model_path": self.model_manager.model_path,
            "device": self.model_manager.device,
            "vllm_enabled": self.model_manager.use_vllm,
            "sample_rate": self.model_manager.sample_rate,
            "available_speakers": speakers,
            "features": [
                "Text-to-speech synthesis",
                "Zero-shot voice cloning",
                "Streaming audio output",
                "Multiple language support",
                "vLLM acceleration (optional)"
            ]
        })


def main():
    parser = argparse.ArgumentParser(
        description="Fun-CosyVoice3-0.5B-2512 TTS Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m web_demo.server.tts_server --port 8004
  python -m web_demo.server.tts_server --model-path ./pretrained_models/Fun-CosyVoice3-0.5B-2512 --use-vllm

API Endpoints:
  POST /api/synthesize       - Text to speech (JSON response with base64 audio)
  POST /api/synthesize/file  - Text to speech (WAV file response)
  GET  /health               - Health check
  GET  /api/info             - Model information
        """
    )

    parser.add_argument("--host", default="0.0.0.0", type=str,
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", default=8004, type=int,
                        help="Server port (default: 8004)")
    parser.add_argument("--model-path", type=str,
                        default="pretrained_models/Fun-CosyVoice3-0.5B-2512",
                        help="Path to TTS model")
    parser.add_argument("--device", default="cuda:0", type=str,
                        help="Device to run model on (default: cuda:0)")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Enable vLLM acceleration")
    parser.add_argument("--use-trt", action="store_true",
                        help="Enable TensorRT acceleration")

    args = parser.parse_args()

    # Initialize model
    model_manager = TTSModelManager()
    try:
        model_manager.initialize(
            args.model_path,
            args.device,
            use_vllm=args.use_vllm,
            use_trt=args.use_trt
        )
    except Exception as e:
        log("error", f"Failed to initialize TTS model: {e}")
        return

    # Create server
    server = TTSServer(model_manager)

    # Setup routes
    app = web.Application(client_max_size=10 * 1024 * 1024)  # 10MB max
    app.router.add_post("/api/synthesize", server.handle_synthesize)
    app.router.add_post("/api/synthesize/file", server.handle_synthesize_file)
    app.router.add_get("/health", server.handle_health)
    app.router.add_get("/api/info", server.handle_info)

    log("info", f"TTS Server starting at http://{args.host}:{args.port}")
    log("info", f"Model: {args.model_path}")
    log("info", f"Device: {args.device}")
    log("info", f"vLLM: {args.use_vllm}, TensorRT: {args.use_trt}")
    log("info", "")
    log("info", "API Endpoints:")
    log("info", f"  POST http://{args.host}:{args.port}/api/synthesize")
    log("info", f"  POST http://{args.host}:{args.port}/api/synthesize/file")
    log("info", f"  GET  http://{args.host}:{args.port}/health")
    log("info", f"  GET  http://{args.host}:{args.port}/api/info")

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
