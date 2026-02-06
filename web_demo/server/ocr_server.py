#!/usr/bin/env python3
"""
GLM-OCR High-Performance API Server
Full pipeline: PP-DocLayout-V3 layout analysis + GLM-OCR recognition via vLLM

Features:
- FastAPI + uvicorn (async, multi-worker)
- httpx.AsyncClient connection pool → vLLM
- paddlex PP-DocLayout-V3 layout detection (thread pool)
- pypdfium2 PDF rendering (thread pool)
- asyncio.Semaphore for GPU concurrency control
- CORS middleware for cross-origin support

Usage:
    # Start vLLM first:
    vllm serve zai-org/GLM-OCR --port 8000 \
        --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
        --served-model-name glm-ocr --gpu-memory-utilization 0.85 --max-num-seqs 64

    # Then start this server:
    python3 -m web_demo.server.ocr_server --port 8007

API Endpoints:
    POST /api/ocr/recognize      - Single image OCR (text/formula/table)
    POST /api/ocr/extract        - Information extraction (JSON schema)
    POST /api/ocr/batch          - Batch image OCR
    POST /api/ocr/document       - Full document parsing (layout + OCR)
    POST /api/ocr/document/stream - SSE streaming document parsing
    GET  /health                 - Health check
    GET  /api/info               - Service info
"""

import argparse
import asyncio
import base64
import io
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

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

# Optional: paddlex for layout detection
try:
    from paddlex import create_pipeline
    PADDLEX_AVAILABLE = True
except ImportError:
    PADDLEX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(level: str, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level.upper()}] {message}")


# ---------------------------------------------------------------------------
# Prompt templates for GLM-OCR
# ---------------------------------------------------------------------------

TASK_PROMPTS = {
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
}


# ---------------------------------------------------------------------------
# OCR Model Manager (singleton)
# ---------------------------------------------------------------------------

class OCRModelManager:
    """Singleton manager for GLM-OCR with vLLM backend + PP-DocLayout"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(
        self,
        vllm_endpoint: str = "http://localhost:8000",
        model_name: str = "glm-ocr",
        max_render_workers: int = 4,
        max_concurrent_ocr: int = 16,
        default_dpi: int = 200,
        max_image_size: int = 2048,
        enable_layout: bool = True,
    ):
        if self._initialized:
            log("info", "OCR model manager already initialized, skipping...")
            return

        self.vllm_endpoint = vllm_endpoint.rstrip("/")
        self.model_name = model_name
        self.default_dpi = default_dpi
        self.max_image_size = max_image_size
        self.enable_layout = enable_layout

        # Thread pool for CPU-bound work (PDF rendering, layout detection)
        self.cpu_executor = ThreadPoolExecutor(max_workers=max_render_workers)
        log("info", f"CPU thread pool: {max_render_workers} workers")

        # Semaphore for vLLM concurrency
        self.ocr_semaphore = asyncio.Semaphore(max_concurrent_ocr)
        log("info", f"Max concurrent OCR: {max_concurrent_ocr}")

        # httpx async client (connection pool)
        self._client: Optional[httpx.AsyncClient] = None

        # Layout model (loaded lazily)
        self._layout_pipeline = None

        log("info", f"vLLM endpoint: {self.vllm_endpoint}")
        log("info", f"Model: {self.model_name}")
        log("info", f"Default DPI: {default_dpi}")
        log("info", f"Max image size: {max_image_size}px")
        log("info", f"Layout detection: {'enabled' if enable_layout else 'disabled'}")

        self._initialized = True

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=10.0),
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ----- Layout detection -----

    def _load_layout_model(self):
        """Load PP-DocLayout-V3 pipeline (CPU-bound, call in thread pool)"""
        if self._layout_pipeline is None and PADDLEX_AVAILABLE:
            log("info", "Loading PP-DocLayout-V3 pipeline...")
            self._layout_pipeline = create_pipeline(pipeline_name="PP-DocLayout-V3")
            log("info", "PP-DocLayout-V3 loaded")
        return self._layout_pipeline

    def _detect_layout_sync(self, image_bytes: bytes) -> List[Dict]:
        """
        Run layout detection on an image (sync, runs in thread pool).
        Returns list of regions: {label, bbox, score}
        """
        pipeline = self._load_layout_model()
        if pipeline is None:
            return []

        # Save to temp file for paddlex
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name

        try:
            output = pipeline.predict(tmp_path)
            regions = []
            for result in output:
                res_dict = result.to_dict() if hasattr(result, "to_dict") else {}
                boxes = res_dict.get("boxes", [])
                for box in boxes:
                    regions.append({
                        "label": box.get("label", "unknown"),
                        "bbox": box.get("coordinate", []),
                        "score": round(box.get("score", 0.0), 4),
                    })
            return regions
        finally:
            os.unlink(tmp_path)

    async def detect_layout(self, image_bytes: bytes) -> List[Dict]:
        """Async wrapper for layout detection"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.cpu_executor, self._detect_layout_sync, image_bytes
        )

    # ----- PDF rendering -----

    def render_page_sync(
        self, pdf_data: bytes, page_num: int, dpi: int = None
    ) -> Tuple[bytes, int, int]:
        """
        Render PDF page to PNG bytes (sync, thread pool).
        Returns (png_bytes, width, height)
        """
        if not PDFIUM_AVAILABLE:
            raise RuntimeError("pypdfium2 not installed: pip install pypdfium2")

        dpi = dpi or self.default_dpi
        pdf = pdfium.PdfDocument(pdf_data)
        if page_num >= len(pdf):
            pdf.close()
            raise ValueError(f"Page {page_num + 1} not found (document has {len(pdf)} pages)")

        page = pdf[page_num]
        scale = dpi / 72.0
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()

        # Resize if needed
        w, h = pil_image.size
        if max(w, h) > self.max_image_size:
            ratio = self.max_image_size / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            w, h = new_w, new_h

        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()

        pdf.close()
        return png_bytes, w, h

    async def render_page(
        self, pdf_data: bytes, page_num: int, dpi: int = None
    ) -> Tuple[bytes, int, int]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.cpu_executor, self.render_page_sync, pdf_data, page_num, dpi
        )

    def get_page_count_sync(self, pdf_data: bytes) -> int:
        if not PDFIUM_AVAILABLE:
            raise RuntimeError("pypdfium2 not installed")
        pdf = pdfium.PdfDocument(pdf_data)
        count = len(pdf)
        pdf.close()
        return count

    async def get_page_count(self, pdf_data: bytes) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.cpu_executor, self.get_page_count_sync, pdf_data
        )

    # ----- Image helpers -----

    @staticmethod
    def _image_to_base64(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    @staticmethod
    def _crop_region(image_bytes: bytes, bbox: List[float]) -> bytes:
        """Crop a region from image bytes using bbox [x1, y1, x2, y2]"""
        img = Image.open(io.BytesIO(image_bytes))
        cropped = img.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()

    # ----- vLLM OCR calls -----

    async def ocr_image(
        self,
        image_bytes: bytes,
        task: str = "text",
        max_tokens: int = 8192,
    ) -> str:
        """
        Perform OCR on image bytes via vLLM.
        task: text / formula / table
        """
        prompt = TASK_PROMPTS.get(task, TASK_PROMPTS["text"])
        b64 = self._image_to_base64(image_bytes)

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        client = await self.get_client()
        async with self.ocr_semaphore:
            resp = await client.post(
                f"{self.vllm_endpoint}/v1/chat/completions", json=payload
            )
            if resp.status_code != 200:
                raise RuntimeError(f"vLLM error ({resp.status_code}): {resp.text}")
            result = resp.json()

        return result["choices"][0]["message"]["content"].strip()

    async def extract_info(
        self,
        image_bytes: bytes,
        schema: str,
        max_tokens: int = 8192,
    ) -> str:
        """Information extraction with user-provided JSON schema as prompt."""
        b64 = self._image_to_base64(image_bytes)

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": schema},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        client = await self.get_client()
        async with self.ocr_semaphore:
            resp = await client.post(
                f"{self.vllm_endpoint}/v1/chat/completions", json=payload
            )
            if resp.status_code != 200:
                raise RuntimeError(f"vLLM error ({resp.status_code}): {resp.text}")
            result = resp.json()

        return result["choices"][0]["message"]["content"].strip()

    # ----- Document parsing (layout + OCR) -----

    async def parse_document_page(
        self,
        page_image_bytes: bytes,
        page_num: int,
        enable_layout: bool = True,
        task: str = "text",
        max_tokens: int = 8192,
    ) -> Dict:
        """Parse a single page: layout detection + per-region OCR."""
        start = time.time()

        if enable_layout and self.enable_layout and PADDLEX_AVAILABLE:
            # Layout detection
            regions = await self.detect_layout(page_image_bytes)

            if regions:
                # OCR each region in parallel
                ocr_tasks = []
                for region in regions:
                    bbox = region.get("bbox", [])
                    label = region.get("label", "unknown")
                    if len(bbox) == 4 and (bbox[2] - bbox[0]) > 10 and (bbox[3] - bbox[1]) > 10:
                        cropped = self._crop_region(page_image_bytes, bbox)
                        # Choose task based on label
                        region_task = task
                        if "formula" in label.lower():
                            region_task = "formula"
                        elif "table" in label.lower():
                            region_task = "table"
                        ocr_tasks.append(
                            self._ocr_region(cropped, region, region_task, max_tokens)
                        )

                if ocr_tasks:
                    region_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)
                    parsed_regions = []
                    for r in region_results:
                        if isinstance(r, Exception):
                            parsed_regions.append({"error": str(r)})
                        else:
                            parsed_regions.append(r)

                    # Combine all text
                    full_text = "\n\n".join(
                        r.get("text", "") for r in parsed_regions if isinstance(r, dict) and r.get("text")
                    )

                    return {
                        "page_number": page_num + 1,
                        "text": full_text,
                        "regions": parsed_regions,
                        "region_count": len(parsed_regions),
                        "processing_time": round(time.time() - start, 3),
                    }

        # Fallback: OCR the full page
        text = await self.ocr_image(page_image_bytes, task=task, max_tokens=max_tokens)
        return {
            "page_number": page_num + 1,
            "text": text,
            "regions": [],
            "region_count": 0,
            "processing_time": round(time.time() - start, 3),
        }

    async def _ocr_region(
        self, cropped_bytes: bytes, region: Dict, task: str, max_tokens: int
    ) -> Dict:
        text = await self.ocr_image(cropped_bytes, task=task, max_tokens=max_tokens)
        return {
            "label": region.get("label", "unknown"),
            "bbox": region.get("bbox", []),
            "score": region.get("score", 0.0),
            "task": task,
            "text": text,
        }

    async def parse_document(
        self,
        file_bytes: bytes,
        filename: str,
        pages: str = "all",
        enable_layout: bool = True,
        dpi: int = None,
        task: str = "text",
        max_tokens: int = 8192,
    ) -> Dict:
        """Parse full document (PDF or image)."""
        start = time.time()
        is_pdf = filename.lower().endswith(".pdf")

        if is_pdf:
            total_pages = await self.get_page_count(file_bytes)
            page_nums = self._parse_page_spec(pages, total_pages)

            # Render + parse pages concurrently
            tasks = []
            for pn in page_nums:
                tasks.append(self._render_and_parse_page(
                    file_bytes, pn, enable_layout, dpi, task, max_tokens
                ))
            page_results = await asyncio.gather(*tasks, return_exceptions=True)

            results = []
            for r in page_results:
                if isinstance(r, Exception):
                    results.append({"error": str(r)})
                else:
                    results.append(r)

            results.sort(key=lambda x: x.get("page_number", 0))
            total_time = time.time() - start
            return {
                "pages": results,
                "total_pages": total_pages,
                "parsed_pages": len(page_nums),
                "total_time": round(total_time, 3),
                "throughput": f"{len(page_nums) / total_time:.2f} pgs/s" if total_time > 0 else "N/A",
            }
        else:
            # Single image
            result = await self.parse_document_page(
                file_bytes, 0, enable_layout, task, max_tokens
            )
            total_time = time.time() - start
            return {
                "pages": [result],
                "total_pages": 1,
                "parsed_pages": 1,
                "total_time": round(total_time, 3),
                "throughput": f"{1 / total_time:.2f} pgs/s" if total_time > 0 else "N/A",
            }

    async def _render_and_parse_page(
        self, pdf_data, page_num, enable_layout, dpi, task, max_tokens
    ) -> Dict:
        png_bytes, w, h = await self.render_page(pdf_data, page_num, dpi)
        result = await self.parse_document_page(
            png_bytes, page_num, enable_layout, task, max_tokens
        )
        result["width"] = w
        result["height"] = h
        return result

    async def parse_document_stream(
        self,
        file_bytes: bytes,
        filename: str,
        pages: str = "all",
        enable_layout: bool = True,
        dpi: int = None,
        task: str = "text",
        max_tokens: int = 8192,
    ) -> AsyncGenerator[Dict, None]:
        """Parse document with streaming (yields each page)."""
        start = time.time()
        is_pdf = filename.lower().endswith(".pdf")

        if is_pdf:
            total_pages = await self.get_page_count(file_bytes)
            page_nums = self._parse_page_spec(pages, total_pages)
        else:
            total_pages = 1
            page_nums = [0]

        yield {"type": "start", "total_pages": total_pages, "pages_to_parse": len(page_nums)}

        completed = 0
        for pn in page_nums:
            try:
                if is_pdf:
                    png_bytes, w, h = await self.render_page(file_bytes, pn, dpi)
                    result = await self.parse_document_page(
                        png_bytes, pn, enable_layout, task, max_tokens
                    )
                    result["width"] = w
                    result["height"] = h
                else:
                    result = await self.parse_document_page(
                        file_bytes, pn, enable_layout, task, max_tokens
                    )

                result["type"] = "page"
                completed += 1
                result["progress"] = f"{completed}/{len(page_nums)}"
                yield result
            except Exception as e:
                yield {"type": "error", "page_number": pn + 1, "error": str(e)}

        total_time = time.time() - start
        yield {
            "type": "complete",
            "total_time": round(total_time, 3),
            "throughput": f"{len(page_nums) / total_time:.2f} pgs/s" if total_time > 0 else "N/A",
        }

    # ----- Page spec parsing -----

    @staticmethod
    def _parse_page_spec(pages: str, total_pages: int) -> List[int]:
        """Parse page spec: 'all', '1-5', '1,3,5', '1-3,7,9-10' → 0-indexed list"""
        if pages.lower() == "all":
            return list(range(total_pages))

        result = set()
        parts = pages.replace(" ", "").split(",")
        for part in parts:
            if "-" in part:
                s, e = part.split("-", 1)
                s = int(s) - 1
                e = int(e)
                for i in range(max(0, s), min(e, total_pages)):
                    result.add(i)
            else:
                p = int(part) - 1
                if 0 <= p < total_pages:
                    result.add(p)
        return sorted(result)

    # ----- Health check -----

    async def health_check(self) -> Dict:
        if not self._initialized:
            return {"status": "not_initialized", "vllm": False, "layout_model": False, "model": "N/A"}

        try:
            client = await self.get_client()
            resp = await client.get(f"{self.vllm_endpoint}/health")
            vllm_healthy = resp.status_code == 200
        except Exception:
            vllm_healthy = False

        layout_available = PADDLEX_AVAILABLE and getattr(self, "enable_layout", False)

        return {
            "status": "healthy" if vllm_healthy else "degraded",
            "vllm": vllm_healthy,
            "layout_model": layout_available,
            "model": getattr(self, "model_name", "glm-ocr"),
        }


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

manager = OCRModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    log("info", "OCR server starting up...")
    # Initialize from env vars if not already initialized via main()
    if not manager._initialized:
        manager.initialize(
            vllm_endpoint=os.environ.get("OCR_VLLM_ENDPOINT", "http://localhost:8000"),
            model_name=os.environ.get("OCR_MODEL_NAME", "glm-ocr"),
            enable_layout=os.environ.get("OCR_ENABLE_LAYOUT", "false").lower() == "true",
        )
    yield
    log("info", "OCR server shutting down...")
    await manager.close()


app = FastAPI(
    title="GLM-OCR API",
    description="High-performance OCR service powered by GLM-OCR + PP-DocLayout-V3",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Routes: Simple OCR -----

@app.post("/api/ocr/recognize")
async def recognize(
    image: UploadFile = File(...),
    task: str = Form("text"),
    max_tokens: int = Form(8192),
):
    """Single image OCR (text / formula / table)."""
    try:
        start = time.time()
        image_bytes = await image.read()

        if task not in TASK_PROMPTS:
            return JSONResponse(
                {"error": f"Invalid task '{task}'. Choices: {list(TASK_PROMPTS.keys())}"},
                status_code=400,
            )

        text = await manager.ocr_image(image_bytes, task=task, max_tokens=max_tokens)
        elapsed = time.time() - start
        log("info", f"Recognize [{task}]: {len(text)} chars in {elapsed:.2f}s")

        return {"text": text, "task": task, "time": round(elapsed, 3)}
    except Exception as e:
        log("error", f"Recognize failed: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/ocr/extract")
async def extract(
    image: UploadFile = File(...),
    schema: str = Form(...),
    max_tokens: int = Form(8192),
):
    """Information extraction with JSON schema prompt."""
    try:
        start = time.time()
        image_bytes = await image.read()

        text = await manager.extract_info(image_bytes, schema=schema, max_tokens=max_tokens)
        elapsed = time.time() - start
        log("info", f"Extract: {len(text)} chars in {elapsed:.2f}s")

        # Try to parse as JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = text

        return {"data": data, "raw": text, "time": round(elapsed, 3)}
    except Exception as e:
        log("error", f"Extract failed: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/ocr/batch")
async def batch(
    images: List[UploadFile] = File(...),
    task: str = Form("text"),
    max_tokens: int = Form(8192),
):
    """Batch image OCR."""
    try:
        start = time.time()

        # Read all images
        image_data = []
        for img in images:
            image_data.append(await img.read())

        # OCR in parallel
        tasks = [
            manager.ocr_image(data, task=task, max_tokens=max_tokens)
            for data in image_data
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                output.append({"index": i, "error": str(r)})
            else:
                output.append({"index": i, "text": r})

        elapsed = time.time() - start
        throughput = len(images) / elapsed if elapsed > 0 else 0
        log("info", f"Batch [{task}]: {len(images)} images in {elapsed:.2f}s ({throughput:.1f} imgs/s)")

        return {
            "results": output,
            "total_time": round(elapsed, 3),
            "throughput": f"{throughput:.1f} imgs/s",
        }
    except Exception as e:
        log("error", f"Batch failed: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ----- Routes: Document parsing -----

@app.post("/api/ocr/document")
async def document(
    file: UploadFile = File(...),
    pages: str = Form("all"),
    enable_layout: bool = Form(True),
    dpi: int = Form(200),
    task: str = Form("text"),
    max_tokens: int = Form(8192),
):
    """Full document parsing (PDF or image): layout analysis + parallel OCR."""
    try:
        start = time.time()
        file_bytes = await file.read()

        # Validate PDF
        if file.filename and file.filename.lower().endswith(".pdf"):
            if not file_bytes.startswith(b"%PDF"):
                return JSONResponse(
                    {"error": "Invalid PDF format"}, status_code=400
                )

        log("info", f"Document parse: {file.filename}, {len(file_bytes)} bytes, pages={pages}")

        result = await manager.parse_document(
            file_bytes=file_bytes,
            filename=file.filename or "upload.png",
            pages=pages,
            enable_layout=enable_layout,
            dpi=dpi,
            task=task,
            max_tokens=max_tokens,
        )

        log("info", f"Document done: {result.get('parsed_pages')} pages, {result.get('throughput')}")
        return result
    except Exception as e:
        log("error", f"Document parse failed: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/ocr/document/stream")
async def document_stream(
    file: UploadFile = File(...),
    pages: str = Form("all"),
    enable_layout: bool = Form(True),
    dpi: int = Form(200),
    task: str = Form("text"),
    max_tokens: int = Form(8192),
):
    """SSE streaming document parsing."""
    try:
        file_bytes = await file.read()
        log("info", f"Stream parse: {file.filename}, {len(file_bytes)} bytes")

        async def event_generator():
            async for event in manager.parse_document_stream(
                file_bytes=file_bytes,
                filename=file.filename or "upload.png",
                pages=pages,
                enable_layout=enable_layout,
                dpi=dpi,
                task=task,
                max_tokens=max_tokens,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    except Exception as e:
        log("error", f"Stream parse failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ----- Routes: Management -----

@app.get("/health")
async def health():
    return await manager.health_check()


@app.get("/api/info")
async def info():
    return {
        "model": manager.model_name,
        "version": "0.9B",
        "vllm_endpoint": manager.vllm_endpoint,
        "default_dpi": manager.default_dpi,
        "max_image_size": manager.max_image_size,
        "layout_detection": manager.enable_layout and PADDLEX_AVAILABLE,
        "features": [
            "Text Recognition (printed/handwritten)",
            "Formula Recognition (LaTeX)",
            "Table Recognition (HTML/Markdown)",
            "Information Extraction (JSON schema)",
            "PDF Document Parsing (layout + OCR)",
            "SSE Streaming",
            "Batch Processing",
            "PP-DocLayout-V3 Layout Analysis",
            "MTP Speculative Decoding",
        ],
        "endpoints": {
            "POST /api/ocr/recognize": "Single image OCR",
            "POST /api/ocr/extract": "Information extraction",
            "POST /api/ocr/batch": "Batch image OCR",
            "POST /api/ocr/document": "Full document parsing",
            "POST /api/ocr/document/stream": "SSE streaming document parsing",
            "GET /health": "Health check",
            "GET /api/info": "Service information",
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GLM-OCR High-Performance API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m web_demo.server.ocr_server --port 8007
  python3 -m web_demo.server.ocr_server --port 8007 --vllm-endpoint http://localhost:8000 --enable-layout

Prerequisites:
  vllm serve zai-org/GLM-OCR --port 8000 \\
      --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \\
      --served-model-name glm-ocr --gpu-memory-utilization 0.85 --max-num-seqs 64
        """,
    )

    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", default=8007, type=int, help="Server port (default: 8007)")
    parser.add_argument("--vllm-endpoint", default="http://localhost:8000", help="vLLM endpoint")
    parser.add_argument("--model-name", default="glm-ocr", help="vLLM served model name")
    parser.add_argument("--max-render-workers", default=4, type=int, help="CPU thread pool size")
    parser.add_argument("--max-concurrent-ocr", default=16, type=int, help="Max concurrent vLLM requests")
    parser.add_argument("--default-dpi", default=200, type=int, help="Default PDF rendering DPI")
    parser.add_argument("--max-image-size", default=2048, type=int, help="Max image dimension (px)")
    parser.add_argument("--enable-layout", action="store_true", default=True, help="Enable layout detection")
    parser.add_argument("--no-layout", action="store_true", help="Disable layout detection")
    parser.add_argument("--workers", default=1, type=int, help="Uvicorn worker count")

    args = parser.parse_args()

    enable_layout = args.enable_layout and not args.no_layout

    # Initialize manager
    manager.initialize(
        vllm_endpoint=args.vllm_endpoint,
        model_name=args.model_name,
        max_render_workers=args.max_render_workers,
        max_concurrent_ocr=args.max_concurrent_ocr,
        default_dpi=args.default_dpi,
        max_image_size=args.max_image_size,
        enable_layout=enable_layout,
    )

    log("info", f"GLM-OCR API Server starting at http://{args.host}:{args.port}")
    log("info", f"vLLM endpoint: {args.vllm_endpoint}")
    log("info", f"Model: {args.model_name}")
    log("info", "")
    log("info", "API Endpoints:")
    log("info", f"  POST http://{args.host}:{args.port}/api/ocr/recognize")
    log("info", f"  POST http://{args.host}:{args.port}/api/ocr/extract")
    log("info", f"  POST http://{args.host}:{args.port}/api/ocr/batch")
    log("info", f"  POST http://{args.host}:{args.port}/api/ocr/document")
    log("info", f"  POST http://{args.host}:{args.port}/api/ocr/document/stream")
    log("info", f"  GET  http://{args.host}:{args.port}/health")
    log("info", f"  GET  http://{args.host}:{args.port}/api/info")

    import uvicorn
    uvicorn.run(
        "web_demo.server.ocr_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
