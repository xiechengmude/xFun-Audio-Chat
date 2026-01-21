#!/usr/bin/env python3
"""
PDF-AI Parsing Server with vLLM Backend
High-performance PDF parsing using LightOnOCR-2-1B vision model

Features:
- Async HTTP API with aiohttp
- Thread pool for PDF rendering (CPU-bound)
- vLLM backend for OCR inference
- Batch processing support
- SSE streaming for real-time results

Usage:
    # Start vLLM first:
    vllm serve lightonai/LightOnOCR-2-1B --port 8000 --limit-mm-per-prompt '{"image": 1}'

    # Then start this server:
    python3 -m web_demo.server.pdf_server --port 8006

API Endpoints:
    POST /api/parse         - Parse single PDF
    POST /api/parse/batch   - Parse multiple PDFs
    POST /api/parse/stream  - Parse with SSE streaming
    GET  /health            - Health check
    GET  /api/info          - Model information
"""

import argparse
import asyncio
import base64
import io
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import web

# Optional: pypdfium2 for PDF rendering
try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except ImportError:
    PDFIUM_AVAILABLE = False

# Optional: PIL for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def log(level: str, message: str):
    """Simple logging with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level.upper()}] {message}")


class PDFModelManager:
    """Singleton manager for PDF parsing with vLLM backend"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(
        self,
        vllm_endpoint: str = "http://localhost:8000",
        max_render_workers: int = 4,
        max_concurrent_ocr: int = 8,
        default_dpi: int = 200,
        max_image_size: int = 1540,
    ):
        """
        Initialize PDF model manager

        Args:
            vllm_endpoint: vLLM server endpoint
            max_render_workers: Max threads for PDF rendering
            max_concurrent_ocr: Max concurrent OCR requests
            default_dpi: Default PDF rendering DPI
            max_image_size: Max image dimension (longest edge)
        """
        if self._initialized:
            log("info", "PDF model manager already initialized, skipping...")
            return

        self.vllm_endpoint = vllm_endpoint.rstrip("/")
        self.default_dpi = default_dpi
        self.max_image_size = max_image_size

        # Thread pool for CPU-bound PDF rendering
        self.render_executor = ThreadPoolExecutor(max_workers=max_render_workers)
        log("info", f"Initialized render pool with {max_render_workers} workers")

        # Semaphore for concurrent OCR requests
        self.ocr_semaphore = asyncio.Semaphore(max_concurrent_ocr)
        log("info", f"Max concurrent OCR requests: {max_concurrent_ocr}")

        # HTTP session (created lazily)
        self._session: Optional[aiohttp.ClientSession] = None

        # Model info
        self.model_name = "LightOnOCR-2-1B"

        log("info", f"vLLM endpoint: {self.vllm_endpoint}")
        log("info", f"Default DPI: {default_dpi}")
        log("info", f"Max image size: {max_image_size}px")

        self._initialized = True

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=120)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def render_page_sync(
        self,
        pdf_data: bytes,
        page_num: int,
        dpi: int = None
    ) -> Tuple[str, int, int]:
        """
        Render a single PDF page to base64 PNG (sync, runs in thread pool)

        Args:
            pdf_data: Raw PDF bytes
            page_num: Page number (0-indexed)
            dpi: Rendering DPI

        Returns:
            Tuple of (base64_image, width, height)
        """
        if not PDFIUM_AVAILABLE:
            raise RuntimeError("pypdfium2 not installed: pip install pypdfium2")

        dpi = dpi or self.default_dpi

        # Open PDF
        pdf = pdfium.PdfDocument(pdf_data)

        if page_num >= len(pdf):
            raise ValueError(f"Page {page_num + 1} not found (document has {len(pdf)} pages)")

        page = pdf[page_num]

        # Calculate scale for target DPI
        scale = dpi / 72.0

        # Render to bitmap
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()

        # Resize if needed (maintain aspect ratio, max dimension = max_image_size)
        w, h = pil_image.size
        if max(w, h) > self.max_image_size:
            ratio = self.max_image_size / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            w, h = new_w, new_h

        # Convert to RGB if needed (remove alpha channel)
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")

        # Encode to PNG base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG", optimize=True)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        pdf.close()

        return base64_image, w, h

    async def render_page(
        self,
        pdf_data: bytes,
        page_num: int,
        dpi: int = None
    ) -> Tuple[str, int, int]:
        """Async wrapper for PDF page rendering"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.render_executor,
            self.render_page_sync,
            pdf_data,
            page_num,
            dpi
        )

    def get_page_count_sync(self, pdf_data: bytes) -> int:
        """Get total page count (sync)"""
        if not PDFIUM_AVAILABLE:
            raise RuntimeError("pypdfium2 not installed")

        pdf = pdfium.PdfDocument(pdf_data)
        count = len(pdf)
        pdf.close()
        return count

    async def get_page_count(self, pdf_data: bytes) -> int:
        """Get total page count (async)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.render_executor,
            self.get_page_count_sync,
            pdf_data
        )

    async def ocr_image(
        self,
        base64_image: str,
        max_tokens: int = 4096
    ) -> str:
        """
        Perform OCR on base64 image using vLLM

        Args:
            base64_image: Base64 encoded PNG image
            max_tokens: Max output tokens

        Returns:
            Extracted text
        """
        session = await self.get_session()

        # OpenAI-compatible chat completions format
        payload = {
            "model": "lightonai/LightOnOCR-2-1B",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "<|im_start|>user\nExtract full text<|im_end|>\n<|im_start|>assistant\n<|reserved_special_token_0|>"
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0
        }

        async with self.ocr_semaphore:
            async with session.post(
                f"{self.vllm_endpoint}/v1/chat/completions",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"vLLM error ({response.status}): {error_text}")

                result = await response.json()

        # Extract text from response
        text = result["choices"][0]["message"]["content"]
        return text.strip()

    async def parse_page(
        self,
        pdf_data: bytes,
        page_num: int,
        dpi: int = None,
        max_tokens: int = 4096
    ) -> Dict:
        """
        Parse a single PDF page

        Args:
            pdf_data: Raw PDF bytes
            page_num: Page number (0-indexed)
            dpi: Rendering DPI
            max_tokens: Max output tokens

        Returns:
            Dict with page_number, text, processing_time
        """
        start_time = time.time()

        # Render page
        base64_image, width, height = await self.render_page(pdf_data, page_num, dpi)

        # OCR
        text = await self.ocr_image(base64_image, max_tokens)

        processing_time = time.time() - start_time

        return {
            "page_number": page_num + 1,  # 1-indexed for display
            "text": text,
            "width": width,
            "height": height,
            "processing_time": round(processing_time, 3)
        }

    async def parse_pdf(
        self,
        pdf_data: bytes,
        pages: str = "all",
        dpi: int = None,
        max_tokens: int = 4096
    ) -> Dict:
        """
        Parse entire PDF or specified pages

        Args:
            pdf_data: Raw PDF bytes
            pages: Page specification ("all", "1-5", "1,3,5")
            dpi: Rendering DPI
            max_tokens: Max tokens per page

        Returns:
            Dict with pages, total_pages, throughput
        """
        start_time = time.time()

        # Get page count
        total_pages = await self.get_page_count(pdf_data)

        # Parse page specification
        page_nums = self._parse_page_spec(pages, total_pages)

        # Parse pages concurrently
        tasks = [
            self.parse_page(pdf_data, pn, dpi, max_tokens)
            for pn in page_nums
        ]
        page_results = await asyncio.gather(*tasks)

        # Sort by page number
        page_results = sorted(page_results, key=lambda x: x["page_number"])

        total_time = time.time() - start_time
        throughput = len(page_nums) / total_time if total_time > 0 else 0

        return {
            "success": True,
            "pages": page_results,
            "total_pages": total_pages,
            "parsed_pages": len(page_nums),
            "total_time": round(total_time, 3),
            "throughput": round(throughput, 2)  # pages/second
        }

    async def parse_pdf_stream(
        self,
        pdf_data: bytes,
        pages: str = "all",
        dpi: int = None,
        max_tokens: int = 4096
    ) -> AsyncGenerator[Dict, None]:
        """
        Parse PDF with streaming results (yields each page as completed)

        Args:
            pdf_data: Raw PDF bytes
            pages: Page specification
            dpi: Rendering DPI
            max_tokens: Max tokens per page

        Yields:
            Dict for each page as it's processed
        """
        start_time = time.time()

        # Get page count
        total_pages = await self.get_page_count(pdf_data)
        page_nums = self._parse_page_spec(pages, total_pages)

        # Yield initial metadata
        yield {
            "type": "start",
            "total_pages": total_pages,
            "pages_to_parse": len(page_nums)
        }

        # Process pages and yield results
        completed = 0
        for pn in page_nums:
            try:
                result = await self.parse_page(pdf_data, pn, dpi, max_tokens)
                result["type"] = "page"
                completed += 1
                result["progress"] = f"{completed}/{len(page_nums)}"
                yield result
            except Exception as e:
                yield {
                    "type": "error",
                    "page_number": pn + 1,
                    "error": str(e)
                }

        # Yield completion
        total_time = time.time() - start_time
        yield {
            "type": "complete",
            "total_time": round(total_time, 3),
            "throughput": round(len(page_nums) / total_time, 2) if total_time > 0 else 0
        }

    def _parse_page_spec(self, pages: str, total_pages: int) -> List[int]:
        """
        Parse page specification string

        Args:
            pages: "all", "1-5", "1,3,5", "1-3,7,9-10"
            total_pages: Total pages in document

        Returns:
            List of 0-indexed page numbers
        """
        if pages.lower() == "all":
            return list(range(total_pages))

        result = set()
        parts = pages.replace(" ", "").split(",")

        for part in parts:
            if "-" in part:
                start, end = part.split("-", 1)
                start = int(start) - 1  # Convert to 0-indexed
                end = int(end)  # End is inclusive
                for i in range(max(0, start), min(end, total_pages)):
                    result.add(i)
            else:
                page = int(part) - 1  # Convert to 0-indexed
                if 0 <= page < total_pages:
                    result.add(page)

        return sorted(result)

    async def health_check(self) -> Dict:
        """Check vLLM backend health"""
        try:
            session = await self.get_session()
            async with session.get(f"{self.vllm_endpoint}/health") as resp:
                vllm_healthy = resp.status == 200
        except Exception:
            vllm_healthy = False

        return {
            "status": "healthy" if vllm_healthy else "degraded",
            "vllm_healthy": vllm_healthy,
            "vllm_endpoint": self.vllm_endpoint,
            "model": self.model_name
        }


class PDFServer:
    """PDF Parsing HTTP Server"""

    def __init__(self, model_manager: PDFModelManager):
        self.model_manager = model_manager

    async def handle_parse(self, request):
        """
        POST /api/parse - Parse single PDF

        Request (multipart/form-data):
            - file: PDF file
            - pages: Page specification (default: "all")
            - dpi: Rendering DPI (default: 200)
            - max_tokens: Max tokens per page (default: 4096)

        Response:
            {
                "success": true,
                "pages": [...],
                "total_pages": 5,
                "throughput": 5.71
            }
        """
        try:
            # Parse multipart form data
            data = await request.post()
            pdf_file = data.get("file")

            if pdf_file is None:
                return web.json_response(
                    {"error": "No PDF file provided", "success": False},
                    status=400
                )

            # Read parameters
            pages = data.get("pages", "all")
            dpi = int(data.get("dpi", self.model_manager.default_dpi))
            max_tokens = int(data.get("max_tokens", 4096))

            # Read PDF data
            pdf_data = pdf_file.file.read()

            log("info", f"Received PDF: {len(pdf_data)} bytes, pages={pages}, dpi={dpi}")

            # Parse PDF
            result = await self.model_manager.parse_pdf(
                pdf_data=pdf_data,
                pages=pages,
                dpi=dpi,
                max_tokens=max_tokens
            )

            log("info", f"Parsed {result['parsed_pages']} pages @ {result['throughput']} pages/s")

            return web.json_response(result)

        except ValueError as e:
            return web.json_response(
                {"error": str(e), "success": False},
                status=400
            )
        except Exception as e:
            log("error", f"Parse failed: {e}")
            import traceback
            traceback.print_exc()
            return web.json_response(
                {"error": str(e), "success": False},
                status=500
            )

    async def handle_parse_batch(self, request):
        """
        POST /api/parse/batch - Parse multiple PDFs

        Request (multipart/form-data):
            - files: Multiple PDF files
            - pages: Page specification (default: "all")
            - dpi: Rendering DPI (default: 200)
            - max_tokens: Max tokens per page (default: 4096)

        Response:
            {
                "success": true,
                "results": [
                    {"filename": "...", "pages": [...], ...},
                    ...
                ]
            }
        """
        try:
            data = await request.post()

            # Collect all files
            files = []
            for key, value in data.items():
                if key.startswith("file"):
                    files.append((value.filename, value.file.read()))

            if not files:
                return web.json_response(
                    {"error": "No PDF files provided", "success": False},
                    status=400
                )

            # Read parameters
            pages = data.get("pages", "all")
            dpi = int(data.get("dpi", self.model_manager.default_dpi))
            max_tokens = int(data.get("max_tokens", 4096))

            log("info", f"Batch parse: {len(files)} files")

            # Parse all files concurrently
            tasks = []
            for filename, pdf_data in files:
                tasks.append(self._parse_with_filename(
                    filename, pdf_data, pages, dpi, max_tokens
                ))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            output = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    output.append({
                        "filename": files[i][0],
                        "success": False,
                        "error": str(result)
                    })
                else:
                    output.append(result)

            return web.json_response({
                "success": True,
                "results": output
            })

        except Exception as e:
            log("error", f"Batch parse failed: {e}")
            return web.json_response(
                {"error": str(e), "success": False},
                status=500
            )

    async def _parse_with_filename(
        self,
        filename: str,
        pdf_data: bytes,
        pages: str,
        dpi: int,
        max_tokens: int
    ) -> Dict:
        """Parse PDF and include filename in result"""
        result = await self.model_manager.parse_pdf(
            pdf_data=pdf_data,
            pages=pages,
            dpi=dpi,
            max_tokens=max_tokens
        )
        result["filename"] = filename
        return result

    async def handle_parse_stream(self, request):
        """
        POST /api/parse/stream - Parse PDF with SSE streaming

        Request (multipart/form-data):
            - file: PDF file
            - pages: Page specification (default: "all")
            - dpi: Rendering DPI (default: 200)
            - max_tokens: Max tokens per page (default: 4096)

        Response: Server-Sent Events stream
            data: {"type": "start", "total_pages": 5, ...}
            data: {"type": "page", "page_number": 1, "text": "...", ...}
            data: {"type": "complete", "throughput": 5.71}
        """
        try:
            data = await request.post()
            pdf_file = data.get("file")

            if pdf_file is None:
                return web.json_response(
                    {"error": "No PDF file provided", "success": False},
                    status=400
                )

            pages = data.get("pages", "all")
            dpi = int(data.get("dpi", self.model_manager.default_dpi))
            max_tokens = int(data.get("max_tokens", 4096))

            pdf_data = pdf_file.file.read()

            log("info", f"Stream parse: {len(pdf_data)} bytes, pages={pages}")

            # Create SSE response
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
            await response.prepare(request)

            # Stream results
            async for event in self.model_manager.parse_pdf_stream(
                pdf_data=pdf_data,
                pages=pages,
                dpi=dpi,
                max_tokens=max_tokens
            ):
                data = json.dumps(event, ensure_ascii=False)
                await response.write(f"data: {data}\n\n".encode("utf-8"))

            await response.write_eof()
            return response

        except Exception as e:
            log("error", f"Stream parse failed: {e}")
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
                "vllm_healthy": true,
                "model": "LightOnOCR-2-1B"
            }
        """
        health = await self.model_manager.health_check()
        return web.json_response(health)

    async def handle_info(self, request):
        """
        GET /api/info - Model and service information

        Response:
            {
                "model": "LightOnOCR-2-1B",
                "vllm_endpoint": "...",
                "features": [...]
            }
        """
        return web.json_response({
            "model": self.model_manager.model_name,
            "vllm_endpoint": self.model_manager.vllm_endpoint,
            "default_dpi": self.model_manager.default_dpi,
            "max_image_size": self.model_manager.max_image_size,
            "features": [
                "High-speed PDF OCR (5.71 pages/s on H100)",
                "Multi-page batch processing",
                "SSE streaming for real-time results",
                "Configurable DPI and resolution",
                "Page range selection",
                "Concurrent processing"
            ],
            "endpoints": {
                "POST /api/parse": "Parse single PDF",
                "POST /api/parse/batch": "Parse multiple PDFs",
                "POST /api/parse/stream": "Parse with SSE streaming",
                "GET /health": "Health check",
                "GET /api/info": "Service information"
            }
        })


async def on_shutdown(app):
    """Cleanup on shutdown"""
    manager = app.get("model_manager")
    if manager:
        await manager.close()


def main():
    parser = argparse.ArgumentParser(
        description="PDF-AI Parsing Server (LightOnOCR-2-1B)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default vLLM endpoint
  python3 -m web_demo.server.pdf_server --port 8006

  # Custom vLLM endpoint
  python3 -m web_demo.server.pdf_server --vllm-endpoint http://localhost:8000 --port 8006

Prerequisites:
  # Start vLLM server first
  vllm serve lightonai/LightOnOCR-2-1B --port 8000 \\
      --limit-mm-per-prompt '{"image": 1}' \\
      --mm-processor-cache-gb 0

API Endpoints:
  POST /api/parse         - Parse single PDF
  POST /api/parse/batch   - Parse multiple PDFs
  POST /api/parse/stream  - Parse with SSE streaming
  GET  /health            - Health check
  GET  /api/info          - Model information
        """
    )

    parser.add_argument("--host", default="0.0.0.0", type=str,
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", default=8006, type=int,
                        help="Server port (default: 8006)")
    parser.add_argument("--vllm-endpoint", default="http://localhost:8000", type=str,
                        help="vLLM server endpoint (default: http://localhost:8000)")
    parser.add_argument("--max-render-workers", default=4, type=int,
                        help="Max threads for PDF rendering (default: 4)")
    parser.add_argument("--max-concurrent-ocr", default=8, type=int,
                        help="Max concurrent OCR requests (default: 8)")
    parser.add_argument("--default-dpi", default=200, type=int,
                        help="Default PDF rendering DPI (default: 200)")
    parser.add_argument("--max-image-size", default=1540, type=int,
                        help="Max image dimension in pixels (default: 1540)")

    args = parser.parse_args()

    # Check dependencies
    if not PDFIUM_AVAILABLE:
        log("error", "pypdfium2 not installed. Please install: pip install pypdfium2")
        return
    if not PIL_AVAILABLE:
        log("error", "Pillow not installed. Please install: pip install pillow")
        return

    # Initialize model manager
    model_manager = PDFModelManager()
    model_manager.initialize(
        vllm_endpoint=args.vllm_endpoint,
        max_render_workers=args.max_render_workers,
        max_concurrent_ocr=args.max_concurrent_ocr,
        default_dpi=args.default_dpi,
        max_image_size=args.max_image_size
    )

    # Create server
    server = PDFServer(model_manager)

    # Setup app
    app = web.Application(client_max_size=200 * 1024 * 1024)  # 200MB max upload
    app["model_manager"] = model_manager

    # Routes
    app.router.add_post("/api/parse", server.handle_parse)
    app.router.add_post("/api/parse/batch", server.handle_parse_batch)
    app.router.add_post("/api/parse/stream", server.handle_parse_stream)
    app.router.add_get("/health", server.handle_health)
    app.router.add_get("/api/info", server.handle_info)

    # Cleanup
    app.on_shutdown.append(on_shutdown)

    log("info", f"PDF Server starting at http://{args.host}:{args.port}")
    log("info", f"vLLM endpoint: {args.vllm_endpoint}")
    log("info", f"Model: LightOnOCR-2-1B")
    log("info", "")
    log("info", "API Endpoints:")
    log("info", f"  POST http://{args.host}:{args.port}/api/parse")
    log("info", f"  POST http://{args.host}:{args.port}/api/parse/batch")
    log("info", f"  POST http://{args.host}:{args.port}/api/parse/stream")
    log("info", f"  GET  http://{args.host}:{args.port}/health")
    log("info", f"  GET  http://{args.host}:{args.port}/api/info")

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
