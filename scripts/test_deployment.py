#!/usr/bin/env python3
"""
Fun-Audio-Chat Deployment Verification Tests
Tests connectivity, WebSocket, and S2S functionality

Usage:
    python3 scripts/test_deployment.py --host 69.30.85.139 --port 22196
    python3 scripts/test_deployment.py --host 69.30.85.139 --port 22196 --full
"""

import os
import sys
import json
import time
import wave
import struct
import asyncio
import argparse
from typing import Dict, Optional, Tuple
from pathlib import Path

# Optional imports for full tests
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


class DeploymentTester:
    """Test Fun-Audio-Chat deployment"""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.ws_url = f"ws://{host}:{port}/api/chat"
        self.results = {
            "connectivity": None,
            "websocket": None,
            "s2s_basic": None,
            "s2s_full": None
        }

    def test_connectivity(self) -> bool:
        """Test basic HTTP/TCP connectivity"""
        print(f"\n[TEST] Connectivity to {self.host}:{self.port}")

        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((self.host, self.port))
            sock.close()

            if result == 0:
                print("  [PASS] TCP connection successful")
                self.results["connectivity"] = True
                return True
            else:
                print(f"  [FAIL] TCP connection failed (code: {result})")
                self.results["connectivity"] = False
                return False

        except Exception as e:
            print(f"  [FAIL] Connection error: {e}")
            self.results["connectivity"] = False
            return False

    async def test_websocket_connection(self) -> bool:
        """Test WebSocket connection"""
        print(f"\n[TEST] WebSocket connection to {self.ws_url}")

        if not HAS_WEBSOCKETS:
            print("  [SKIP] websockets library not installed")
            print("  Install with: pip install websockets")
            return False

        try:
            async with websockets.connect(
                self.ws_url,
                ping_timeout=30,
                close_timeout=10
            ) as ws:
                print("  [PASS] WebSocket connected")

                # Send a simple control message
                control_msg = json.dumps({"type": "ping"})
                await ws.send(control_msg)
                print("  [PASS] Message sent")

                # Wait briefly for any response
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    print(f"  [INFO] Received: {response[:100]}...")
                except asyncio.TimeoutError:
                    print("  [INFO] No immediate response (normal for this protocol)")

                self.results["websocket"] = True
                return True

        except Exception as e:
            print(f"  [FAIL] WebSocket error: {e}")
            self.results["websocket"] = False
            return False

    def generate_test_audio(self, duration_sec: float = 1.0,
                            sample_rate: int = 16000) -> bytes:
        """Generate a simple test audio waveform (sine wave)"""
        import math

        num_samples = int(sample_rate * duration_sec)
        frequency = 440  # A4 note

        samples = []
        for i in range(num_samples):
            t = i / sample_rate
            value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
            samples.append(struct.pack('<h', value))

        return b''.join(samples)

    async def test_s2s_basic(self) -> bool:
        """Test basic S2S flow (send audio, expect response)"""
        print(f"\n[TEST] Basic S2S functionality")

        if not HAS_WEBSOCKETS:
            print("  [SKIP] websockets library not installed")
            return False

        try:
            async with websockets.connect(
                self.ws_url,
                ping_timeout=60,
                close_timeout=30
            ) as ws:
                print("  [INFO] Connected to WebSocket")

                # Send start control message
                start_msg = json.dumps({"type": "start"})
                await ws.send(start_msg)
                print("  [INFO] Sent start signal")

                # Generate and send test audio
                # Protocol: 0x01 prefix for audio data (Opus encoded)
                test_audio = self.generate_test_audio(0.5)

                # Note: The actual server expects Opus-encoded audio
                # For basic testing, we just verify the connection flow
                audio_msg = b'\x01' + test_audio[:1000]  # Truncated for test
                await ws.send(audio_msg)
                print("  [INFO] Sent test audio")

                # Send pause to signal end of input
                pause_msg = json.dumps({"type": "pause"})
                await ws.send(pause_msg)
                print("  [INFO] Sent pause signal")

                # Wait for response
                response_received = False
                start_time = time.time()
                timeout = 30  # 30 seconds timeout

                while time.time() - start_time < timeout:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=5)

                        if isinstance(response, bytes):
                            msg_type = response[0] if response else 0
                            if msg_type == 0x01:  # Audio response
                                print(f"  [PASS] Received audio response ({len(response)} bytes)")
                                response_received = True
                            elif msg_type == 0x02:  # Text response
                                text = response[1:].decode('utf-8', errors='ignore')
                                print(f"  [PASS] Received text: {text[:50]}...")
                                response_received = True
                        else:
                            # JSON control message
                            try:
                                data = json.loads(response)
                                print(f"  [INFO] Control message: {data.get('type', 'unknown')}")
                            except json.JSONDecodeError:
                                print(f"  [INFO] Text response: {response[:50]}...")
                                response_received = True

                    except asyncio.TimeoutError:
                        if response_received:
                            break
                        print("  [INFO] Waiting for response...")

                if response_received:
                    print("  [PASS] S2S basic test passed")
                    self.results["s2s_basic"] = True
                    return True
                else:
                    print("  [WARN] No response received (server may still be loading)")
                    self.results["s2s_basic"] = False
                    return False

        except Exception as e:
            print(f"  [FAIL] S2S test error: {e}")
            self.results["s2s_basic"] = False
            return False

    async def test_s2s_full(self) -> bool:
        """Full S2S test with real audio encoding (requires sphn/opus)"""
        print(f"\n[TEST] Full S2S functionality (with Opus encoding)")

        try:
            import sphn
            HAS_SPHN = True
        except ImportError:
            HAS_SPHN = False
            print("  [SKIP] sphn library not installed")
            print("  Install with: pip install sphn")
            return False

        if not HAS_WEBSOCKETS:
            print("  [SKIP] websockets library not installed")
            return False

        try:
            async with websockets.connect(
                self.ws_url,
                ping_timeout=120,
                close_timeout=30
            ) as ws:
                print("  [INFO] Connected to WebSocket")

                # Initialize Opus encoder
                encoder = sphn.OpusStreamWriter(24000)

                # Generate test audio (16kHz)
                test_pcm = self.generate_test_audio(2.0, sample_rate=16000)

                # Convert to float32 for encoder
                import numpy as np
                audio_array = np.frombuffer(test_pcm, dtype=np.int16).astype(np.float32) / 32768.0

                # Resample to 24kHz for Opus
                # Simple linear interpolation
                target_len = int(len(audio_array) * 24000 / 16000)
                indices = np.linspace(0, len(audio_array) - 1, target_len)
                audio_24k = np.interp(indices, np.arange(len(audio_array)), audio_array)

                # Encode to Opus
                encoder.append_pcm(audio_24k)
                opus_data = encoder.read_bytes()

                print(f"  [INFO] Encoded {len(test_pcm)} bytes PCM to {len(opus_data)} bytes Opus")

                # Send start
                await ws.send(json.dumps({"type": "start"}))

                # Send Opus audio
                if opus_data:
                    audio_msg = b'\x01' + opus_data
                    await ws.send(audio_msg)
                    print("  [INFO] Sent Opus audio")

                # Send pause
                await ws.send(json.dumps({"type": "pause"}))

                # Collect responses
                audio_responses = []
                text_responses = []
                start_time = time.time()
                timeout = 60

                while time.time() - start_time < timeout:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=10)

                        if isinstance(response, bytes) and len(response) > 0:
                            msg_type = response[0]
                            if msg_type == 0x01:
                                audio_responses.append(response[1:])
                            elif msg_type == 0x02:
                                text = response[1:].decode('utf-8', errors='ignore')
                                text_responses.append(text)
                                print(f"  [INFO] Text: {text}")
                        else:
                            try:
                                data = json.loads(response)
                                if data.get("type") == "endTurn":
                                    print("  [INFO] End of turn received")
                                    break
                            except json.JSONDecodeError:
                                pass

                    except asyncio.TimeoutError:
                        if audio_responses or text_responses:
                            break
                        print("  [INFO] Waiting for response...")

                # Summarize results
                total_audio = sum(len(a) for a in audio_responses)
                print(f"\n  Results:")
                print(f"    Audio responses: {len(audio_responses)} chunks ({total_audio} bytes)")
                print(f"    Text responses: {len(text_responses)}")

                if audio_responses or text_responses:
                    print("  [PASS] Full S2S test passed")
                    self.results["s2s_full"] = True
                    return True
                else:
                    print("  [FAIL] No responses received")
                    self.results["s2s_full"] = False
                    return False

        except Exception as e:
            print(f"  [FAIL] Full S2S test error: {e}")
            import traceback
            traceback.print_exc()
            self.results["s2s_full"] = False
            return False

    def run_tests(self, full: bool = False) -> Dict:
        """Run all tests"""
        print("="*60)
        print("Fun-Audio-Chat Deployment Verification")
        print(f"Target: {self.host}:{self.port}")
        print("="*60)

        # Test 1: Connectivity
        if not self.test_connectivity():
            print("\n[ABORT] Connectivity test failed, skipping further tests")
            return self.results

        # Test 2: WebSocket
        asyncio.run(self.test_websocket_connection())

        # Test 3: Basic S2S
        asyncio.run(self.test_s2s_basic())

        # Test 4: Full S2S (optional)
        if full:
            asyncio.run(self.test_s2s_full())

        # Summary
        self._print_summary()
        return self.results

    def _print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        passed = 0
        failed = 0
        skipped = 0

        for test_name, result in self.results.items():
            if result is True:
                status = "[PASS]"
                passed += 1
            elif result is False:
                status = "[FAIL]"
                failed += 1
            else:
                status = "[SKIP]"
                skipped += 1

            print(f"  {test_name}: {status}")

        print("-"*60)
        print(f"  Passed: {passed}, Failed: {failed}, Skipped: {skipped}")

        if failed == 0 and passed > 0:
            print("\n  Overall: DEPLOYMENT VERIFIED")
        elif failed > 0:
            print("\n  Overall: DEPLOYMENT ISSUES DETECTED")
        else:
            print("\n  Overall: UNABLE TO VERIFY")


def test_deployment(host: str, port: int, full: bool = False) -> Dict:
    """Convenience function for importing from other scripts"""
    tester = DeploymentTester(host, port)
    return tester.run_tests(full)


def main():
    parser = argparse.ArgumentParser(
        description="Fun-Audio-Chat Deployment Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 test_deployment.py --host 69.30.85.139 --port 22196
  python3 test_deployment.py --host 69.30.85.139 --port 22196 --full
        """
    )

    parser.add_argument("--host", required=True, help="Server IP address")
    parser.add_argument("--port", type=int, required=True, help="Server port")
    parser.add_argument("--full", action="store_true",
                        help="Run full S2S test with Opus encoding")

    args = parser.parse_args()

    tester = DeploymentTester(args.host, args.port)
    results = tester.run_tests(args.full)

    # Exit code based on results
    if results.get("connectivity") and results.get("websocket"):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
