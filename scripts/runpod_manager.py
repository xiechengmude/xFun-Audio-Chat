#!/usr/bin/env python3
"""
RunPod Manager for Fun-Audio-Chat
- Create/manage templates
- Create/manage pods with A40/4090 GPUs
"""

import os
import requests
import json
from typing import Optional, List, Dict

class RunPodManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rest_base = "https://rest.runpod.io/v1"
        self.graphql_url = "https://api.runpod.io/graphql"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def graphql_query(self, query: str, variables: dict = None) -> dict:
        """Execute GraphQL query"""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        resp = requests.post(self.graphql_url, headers=self.headers, json=payload)
        if resp.status_code != 200:
            print(f"GraphQL Error: {resp.status_code}")
            print(f"Response: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    def list_gpu_types(self, filter_names: List[str] = None) -> List[dict]:
        """List available GPU types"""
        query = """
        query GpuTypes {
            gpuTypes {
                id
                displayName
                memoryInGb
                secureCloud
                communityCloud
            }
        }
        """
        result = self.graphql_query(query)
        gpus = result.get("data", {}).get("gpuTypes", [])

        if filter_names:
            gpus = [g for g in gpus if any(n.lower() in g.get("displayName", "").lower() for n in filter_names)]

        return gpus

    def list_pods(self) -> List[dict]:
        """List all pods"""
        resp = requests.get(f"{self.rest_base}/pods", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def get_pod(self, pod_id: str) -> dict:
        """Get pod details"""
        resp = requests.get(f"{self.rest_base}/pods/{pod_id}", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def create_template(self, name: str, image_name: str, docker_args: str = "",
                       container_disk_gb: int = 100, volume_gb: int = 100,
                       volume_mount_path: str = "/workspace",
                       ports: str = "22/tcp,8000/http,8001/http,8002/http,8003/http,8004/http,8005/http",
                       env: dict = None, is_serverless: bool = False) -> dict:
        """Create a pod template using GraphQL mutation"""

        # Build env array
        env_array = [{"key": k, "value": v} for k, v in (env or {}).items()]
        env_str = ", ".join([f'{{ key: "{e["key"]}", value: "{e["value"]}" }}' for e in env_array]) if env_array else ""

        # Build the mutation directly (not using variables due to API quirks)
        # dockerArgs is required, use empty string if not provided
        docker_args_value = docker_args if docker_args else ""

        query = f"""
        mutation {{
            saveTemplate(input: {{
                name: "{name}",
                imageName: "{image_name}",
                dockerArgs: "{docker_args_value}",
                containerDiskInGb: {container_disk_gb},
                volumeInGb: {volume_gb},
                volumeMountPath: "{volume_mount_path}",
                ports: "{ports}",
                isServerless: {"true" if is_serverless else "false"}
                {f', env: [{env_str}]' if env_str else ''}
            }}) {{
                id
                name
                imageName
                containerDiskInGb
                volumeInGb
                ports
            }}
        }}
        """

        result = self.graphql_query(query)
        return result

    def create_pod(self, name: str, gpu_type_id: str, image_name: str,
                   container_disk_gb: int = 100, volume_gb: int = 100,
                   gpu_count: int = 1, cloud_type: str = "SECURE",
                   ports: List[str] = None, env: dict = None,
                   template_id: str = None) -> dict:
        """Create a new pod

        Args:
            cloud_type: 'SECURE' or 'COMMUNITY'
        """

        if ports is None:
            # Default ports for Fun-Audio-Chat (as array)
            ports = ["22/tcp", "8000/tcp", "8001/tcp", "8002/tcp", "8003/tcp",
                     "8004/tcp", "8005/tcp", "8006/tcp", "8007/tcp", "8008/tcp",
                     "8009/tcp", "8010/tcp", "8080/http", "8888/http"]

        payload = {
            "name": name,
            "imageName": image_name,
            "gpuTypeIds": [gpu_type_id],  # Array of GPU type IDs
            "gpuCount": gpu_count,
            "cloudType": cloud_type,  # Must be 'SECURE' or 'COMMUNITY'
            "containerDiskInGb": container_disk_gb,
            "volumeInGb": volume_gb,
            "volumeMountPath": "/workspace",
            "ports": ports,  # Array, not string
            "supportPublicIp": True,
        }

        if template_id:
            payload["templateId"] = template_id

        if env:
            payload["env"] = env

        resp = requests.post(f"{self.rest_base}/pods", headers=self.headers, json=payload)
        if resp.status_code not in [200, 201]:
            print(f"API Error: {resp.status_code}")
            print(f"Response: {resp.text}")
            resp.raise_for_status()
        return resp.json()

    def stop_pod(self, pod_id: str) -> dict:
        """Stop a pod"""
        resp = requests.post(f"{self.rest_base}/pods/{pod_id}/stop", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def start_pod(self, pod_id: str) -> dict:
        """Start a stopped pod"""
        resp = requests.post(f"{self.rest_base}/pods/{pod_id}/start", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def terminate_pod(self, pod_id: str) -> dict:
        """Terminate (delete) a pod"""
        resp = requests.delete(f"{self.rest_base}/pods/{pod_id}", headers=self.headers)
        resp.raise_for_status()
        return resp.json()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RunPod Manager for Fun-Audio-Chat")
    parser.add_argument("--action", choices=["list-gpus", "list-pods", "create-pod", "create-template", "stop", "start", "terminate"], required=True)
    parser.add_argument("--gpu", default="A40", help="GPU type filter (A40, 4090)")
    parser.add_argument("--name", help="Pod/Template name")
    parser.add_argument("--pod-id", help="Pod ID for stop/start/terminate")
    parser.add_argument("--disk", type=int, default=100, help="Container disk size in GB")
    parser.add_argument("--volume", type=int, default=100, help="Volume size in GB")

    args = parser.parse_args()

    # Load API key from environment or .env file
    api_key = os.environ.get("RUNPOD_KEY")
    if not api_key:
        env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    if line.startswith("RUNPOD_KEY="):
                        api_key = line.strip().split("=", 1)[1]
                        break

    if not api_key:
        print("Error: RUNPOD_KEY not found")
        return 1

    manager = RunPodManager(api_key)

    if args.action == "list-gpus":
        gpus = manager.list_gpu_types([args.gpu] if args.gpu else None)
        print(f"\n{'GPU Type':<40} {'ID':<30} {'VRAM':<10} {'Secure':<10} {'Community'}")
        print("-" * 100)
        for gpu in gpus:
            print(f"{gpu.get('displayName', 'N/A'):<40} {gpu.get('id', 'N/A'):<30} {gpu.get('memoryInGb', 'N/A'):<10} {gpu.get('secureCloud', False):<10} {gpu.get('communityCloud', False)}")

    elif args.action == "list-pods":
        pods = manager.list_pods()
        print(f"\n{'Name':<25} {'ID':<20} {'GPU':<15} {'Status':<15} {'Public IP'}")
        print("-" * 90)
        for pod in pods:
            print(f"{pod.get('name', 'N/A'):<25} {pod.get('id', 'N/A'):<20} {pod.get('gpuCount', 0):<15} {pod.get('desiredStatus', 'N/A'):<15} {pod.get('publicIp', 'N/A')}")
            if pod.get('portMappings'):
                print(f"  Port mappings: {pod['portMappings']}")

    elif args.action == "create-pod":
        if not args.name:
            args.name = "fun-audio-chat"

        # Get GPU type ID
        gpus = manager.list_gpu_types([args.gpu])
        if not gpus:
            print(f"Error: No GPU found matching '{args.gpu}'")
            return 1

        gpu_id = gpus[0]["id"]
        print(f"Using GPU: {gpus[0]['displayName']} ({gpu_id})")

        # Create pod with default ports (defined in create_pod method)
        result = manager.create_pod(
            name=args.name,
            gpu_type_id=gpu_id,
            image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            container_disk_gb=args.disk,
            volume_gb=args.volume,
        )
        print(f"\nPod created:")
        print(json.dumps(result, indent=2))

    elif args.action == "create-template":
        if not args.name:
            args.name = "Fun-Audio-Chat-Template"

        # Key TCP ports for Fun-Audio-Chat
        # 22: SSH, 8000-8010: main services, 8080/8888: web interfaces
        ports = "22/tcp,8000/tcp,8001/tcp,8002/tcp,8003/tcp,8004/tcp,8005/tcp,8006/tcp,8007/tcp,8008/tcp,8009/tcp,8010/tcp,8080/http,8888/http"

        result = manager.create_template(
            name=args.name,
            image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            container_disk_gb=args.disk,
            volume_gb=args.volume,
            ports=ports,
            env={
                "JUPYTER_PASSWORD": "funaudiochat",
            }
        )
        print(f"\nTemplate created:")
        print(json.dumps(result, indent=2))

    elif args.action in ["stop", "start", "terminate"]:
        if not args.pod_id:
            print("Error: --pod-id required")
            return 1

        if args.action == "stop":
            result = manager.stop_pod(args.pod_id)
        elif args.action == "start":
            result = manager.start_pod(args.pod_id)
        else:
            result = manager.terminate_pod(args.pod_id)

        print(f"\n{args.action.capitalize()} result:")
        print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    exit(main())
