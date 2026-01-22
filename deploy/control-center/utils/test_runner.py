"""Test runner for E2E validation of PDF-AI service."""

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration: float
    message: str = ""
    details: Optional[dict] = None


@dataclass
class BenchmarkResult:
    """Result of benchmark test."""
    total_pages: int
    total_time: float
    throughput: float  # pages/second
    avg_latency: float  # seconds/page
    errors: int = 0


class TestRunner:
    """Runs E2E tests against PDF-AI service."""

    def __init__(
        self,
        api_host: str,
        api_port: int = 8006,
        vllm_port: int = 8000,
        timeout: int = 60,
    ):
        """Initialize test runner."""
        self.api_base = f"http://{api_host}:{api_port}"
        self.vllm_base = f"http://{api_host}:{vllm_port}"
        self.timeout = timeout

    async def check_vllm_health(self) -> TestResult:
        """Check vLLM server health."""
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.vllm_base}/health")
                duration = time.time() - start

                if resp.status_code == 200:
                    return TestResult(
                        name="vLLM Health",
                        passed=True,
                        duration=duration,
                        message="vLLM server is healthy",
                    )
                else:
                    return TestResult(
                        name="vLLM Health",
                        passed=False,
                        duration=duration,
                        message=f"vLLM returned status {resp.status_code}",
                    )
        except Exception as e:
            return TestResult(
                name="vLLM Health",
                passed=False,
                duration=time.time() - start,
                message=f"Failed to connect: {e}",
            )

    async def check_api_health(self) -> TestResult:
        """Check PDF API server health."""
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.api_base}/health")
                duration = time.time() - start

                if resp.status_code == 200:
                    data = resp.json()
                    return TestResult(
                        name="PDF API Health",
                        passed=True,
                        duration=duration,
                        message="PDF API server is healthy",
                        details=data,
                    )
                else:
                    return TestResult(
                        name="PDF API Health",
                        passed=False,
                        duration=duration,
                        message=f"API returned status {resp.status_code}",
                    )
        except Exception as e:
            return TestResult(
                name="PDF API Health",
                passed=False,
                duration=time.time() - start,
                message=f"Failed to connect: {e}",
            )

    async def test_single_page_parse(
        self,
        pdf_path: Optional[str] = None,
        pdf_content: Optional[bytes] = None,
    ) -> TestResult:
        """Test single page PDF parsing."""
        start = time.time()

        # Use provided PDF or generate a simple test PDF
        if pdf_path:
            pdf_content = Path(pdf_path).read_bytes()
        elif not pdf_content:
            # Generate minimal test PDF
            pdf_content = self._generate_test_pdf()

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                files = {"file": ("test.pdf", pdf_content, "application/pdf")}
                data = {"pages": "1", "max_tokens": 2048}

                resp = await client.post(
                    f"{self.api_base}/api/parse",
                    files=files,
                    data=data,
                )
                duration = time.time() - start

                if resp.status_code == 200:
                    result = resp.json()
                    if result.get("success"):
                        pages = result.get("pages", [])
                        text_len = len(pages[0].get("text", "")) if pages else 0
                        return TestResult(
                            name="Single Page Parse",
                            passed=True,
                            duration=duration,
                            message=f"Parsed 1 page, extracted {text_len} chars",
                            details=result,
                        )
                    else:
                        return TestResult(
                            name="Single Page Parse",
                            passed=False,
                            duration=duration,
                            message=f"Parse failed: {result.get('error')}",
                        )
                else:
                    return TestResult(
                        name="Single Page Parse",
                        passed=False,
                        duration=duration,
                        message=f"API returned status {resp.status_code}",
                    )
        except Exception as e:
            return TestResult(
                name="Single Page Parse",
                passed=False,
                duration=time.time() - start,
                message=f"Request failed: {e}",
            )

    async def test_batch_parse(
        self,
        pdf_paths: Optional[list[str]] = None,
        num_pages: int = 3,
    ) -> TestResult:
        """Test batch PDF parsing."""
        start = time.time()

        try:
            # Use provided PDFs or generate test PDFs
            if pdf_paths:
                files = [
                    ("files", (Path(p).name, Path(p).read_bytes(), "application/pdf"))
                    for p in pdf_paths[:3]
                ]
            else:
                # Generate test PDFs
                files = [
                    ("files", (f"test{i}.pdf", self._generate_test_pdf(), "application/pdf"))
                    for i in range(num_pages)
                ]

            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{self.api_base}/api/parse/batch",
                    files=files,
                )
                duration = time.time() - start

                if resp.status_code == 200:
                    result = resp.json()
                    if result.get("success"):
                        total_pages = result.get("total_pages", 0)
                        throughput = result.get("throughput", 0)
                        return TestResult(
                            name="Batch Parse",
                            passed=True,
                            duration=duration,
                            message=f"Parsed {total_pages} pages at {throughput:.2f} pages/s",
                            details=result,
                        )
                    else:
                        return TestResult(
                            name="Batch Parse",
                            passed=False,
                            duration=duration,
                            message=f"Batch parse failed: {result.get('error')}",
                        )
                else:
                    return TestResult(
                        name="Batch Parse",
                        passed=False,
                        duration=duration,
                        message=f"API returned status {resp.status_code}",
                    )
        except Exception as e:
            return TestResult(
                name="Batch Parse",
                passed=False,
                duration=time.time() - start,
                message=f"Request failed: {e}",
            )

    async def run_benchmark(
        self,
        pdf_paths: list[str],
        num_iterations: int = 3,
    ) -> BenchmarkResult:
        """Run performance benchmark."""
        logger.info(f"Running benchmark with {len(pdf_paths)} PDFs, {num_iterations} iterations")

        total_pages = 0
        total_time = 0.0
        errors = 0

        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")

            for pdf_path in pdf_paths:
                start = time.time()
                try:
                    pdf_content = Path(pdf_path).read_bytes()
                    async with httpx.AsyncClient(timeout=120) as client:
                        files = {"file": ("test.pdf", pdf_content, "application/pdf")}
                        resp = await client.post(
                            f"{self.api_base}/api/parse",
                            files=files,
                        )

                        if resp.status_code == 200:
                            result = resp.json()
                            if result.get("success"):
                                pages = len(result.get("pages", []))
                                total_pages += pages
                                total_time += time.time() - start
                            else:
                                errors += 1
                        else:
                            errors += 1
                except Exception as e:
                    logger.error(f"Benchmark error: {e}")
                    errors += 1

        throughput = total_pages / total_time if total_time > 0 else 0
        avg_latency = total_time / total_pages if total_pages > 0 else 0

        return BenchmarkResult(
            total_pages=total_pages,
            total_time=total_time,
            throughput=throughput,
            avg_latency=avg_latency,
            errors=errors,
        )

    async def run_all_tests(self) -> list[TestResult]:
        """Run all E2E tests."""
        results = []

        logger.info("Running E2E tests...")

        # Health checks
        results.append(await self.check_vllm_health())
        results.append(await self.check_api_health())

        # Functional tests (only if health checks pass)
        if all(r.passed for r in results):
            results.append(await self.test_single_page_parse())
            results.append(await self.test_batch_parse())

        # Summary
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        logger.info(f"Tests complete: {passed} passed, {failed} failed")

        return results

    def _generate_test_pdf(self) -> bytes:
        """Generate a minimal test PDF with text."""
        # Minimal PDF with "Hello World" text
        pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 24 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000359 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
434
%%EOF"""
        return pdf_content
