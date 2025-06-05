#!/usr/bin/env python3
"""
Test script to verify MoE configuration for Granite models.
This script validates that the TRL provider correctly detects and configures MoE models.
"""

import requests
import json
import time
from pathlib import Path

def test_server_connection():
    """Test if the Llama Stack server is running and responding."""
    print("🔧 Testing server connection...")
    try:
        response = requests.get("http://localhost:8321/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and responding")
            return True
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def test_trl_provider():
    """Test if the TRL provider is properly registered with MoE support."""
    print("\n🔧 Testing TRL provider registration...")
    try:
        response = requests.get("http://localhost:8321/v1/providers", timeout=5)
        if response.status_code == 200:
            providers = response.json()
            trl_provider = None
            
            for provider in providers.get('data', []):
                if provider.get('provider_id') == 'trl':
                    trl_provider = provider
                    break
            
            if trl_provider:
                print("✅ TRL provider found")
                print(f"   Provider ID: {trl_provider.get('provider_id')}")
                print(f"   Provider Type: {trl_provider.get('provider_type')}")
                print(f"   API: {trl_provider.get('api')}")
                return True
            else:
                print("❌ TRL provider not found")
                return False
        else:
            print(f"❌ Failed to get providers: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error checking providers: {e}")
        return False

def register_test_dataset():
    """Register a simple test dataset for MoE training."""
    print("\n🔧 Registering test dataset...")
    
    dataset_config = {
        "dataset_id": "preference_dataset",
        "purpose": "post-training/messages",
        "dataset_type": "preference",
        "source": {
            "type": "rows",
            "rows": [
                {
                    "prompt": "What is machine learning?",
                    "chosen": "Machine learning is a branch of artificial intelligence that enables computers to learn from data and improve their performance on specific tasks without being explicitly programmed. It uses algorithms to find patterns in data and make predictions or decisions.",
                    "rejected": "Machine learning is just computers doing math stuff with data."
                },
                {
                    "prompt": "Explain neural networks briefly.",
                    "chosen": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections, enabling pattern recognition and learning from examples.",
                    "rejected": "Neural networks are just complicated math formulas."
                }
            ]
        },
        "metadata": {
            "provider_id": "localfs",
            "description": "Test dataset for Granite MoE DPO training"
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:8321/v1/datasets",
            headers={"Content-Type": "application/json"},
            json=dataset_config,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Test dataset registered successfully")
            return True
        else:
            print(f"❌ Failed to register dataset: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error registering dataset: {e}")
        return False

def test_granite_moe_job():
    """Submit a test job for Granite MoE training."""
    print("\n🔧 Testing Granite MoE job submission...")
    
    # Load job configuration
    job_file = Path("granite_moe_job.json")
    if not job_file.exists():
        print(f"❌ Job file not found: {job_file}")
        return False
    
    try:
        with open(job_file, 'r') as f:
            job_config = json.load(f)
        
        # Update job UUID to avoid conflicts
        job_config["job_uuid"] = f"granite-moe-test-{int(time.time())}"
        
        print(f"📋 Submitting job: {job_config['job_uuid']}")
        print(f"🏗️ Model: {job_config['model']}")
        
        response = requests.post(
            "http://localhost:8321/v1/post-training/jobs",
            headers={"Content-Type": "application/json"},
            json=job_config,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            job_uuid = result.get('job_uuid', 'unknown')
            print(f"✅ Job submitted successfully!")
            print(f"📋 Job UUID: {job_uuid}")
            return job_uuid
        else:
            print(f"❌ Job submission failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error submitting job: {e}")
        return None

def monitor_job_status(job_uuid, max_wait_time=300):
    """Monitor job status and look for MoE-specific log messages."""
    print(f"\n🔧 Monitoring job status: {job_uuid}")
    print("🔍 Looking for MoE configuration messages...")
    
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(
                f"http://localhost:8321/v1/post-training/job/status?job_uuid={job_uuid}",
                timeout=5
            )
            
            if response.status_code == 200:
                status_data = response.json()
                current_status = status_data.get('status', 'unknown')
                
                if current_status != last_status:
                    print(f"📊 Job status: {current_status}")
                    last_status = current_status
                
                if current_status == "completed":
                    print("✅ Job completed successfully!")
                    checkpoints = status_data.get('checkpoints', [])
                    if checkpoints:
                        print(f"💾 Checkpoints created: {len(checkpoints)}")
                        for checkpoint in checkpoints:
                            print(f"   - {checkpoint.get('identifier')}")
                    return True
                
                elif current_status == "failed":
                    print("❌ Job failed!")
                    return False
                
                # Check if job is still running
                elif current_status in ["scheduled", "running", "in_progress"]:
                    time.sleep(5)  # Wait before next check
                    continue
                
            else:
                print(f"❌ Error checking status: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error monitoring job: {e}")
            return False
    
    print("⏰ Job monitoring timeout reached")
    return False

def main():
    """Run all MoE configuration tests."""
    print("🧪 Granite MoE Configuration Test Suite")
    print("=" * 50)
    
    # Test 1: Server connection
    if not test_server_connection():
        print("\n❌ Server connection test failed. Please start the server with:")
        print("   llama stack run --image-type venv --image-name trl-post-training run_granite_moe.yaml")
        return
    
    # Test 2: TRL provider registration
    if not test_trl_provider():
        print("\n❌ TRL provider test failed. Check server configuration.")
        return
    
    # Test 3: Dataset registration
    if not register_test_dataset():
        print("\n❌ Dataset registration failed.")
        return
    
    # Test 4: Job submission
    job_uuid = test_granite_moe_job()
    if not job_uuid:
        print("\n❌ Job submission failed.")
        return
    
    # Test 5: Monitor job (optional - can take a while)
    print(f"\n🎯 Job submitted successfully: {job_uuid}")
    print("📝 You can monitor the job manually with:")
    print(f"   curl \"http://localhost:8321/v1/post-training/job/status?job_uuid={job_uuid}\"")
    
    # Optional: Monitor job progress
    user_input = input("\n❓ Monitor job progress? (y/N): ").strip().lower()
    if user_input in ['y', 'yes']:
        monitor_job_status(job_uuid)
    
    print("\n🎉 MoE configuration test completed!")
    print("\n💡 Key things to check in the logs:")
    print("   - 'Detected MoE model with X experts'")
    print("   - 'Router logits enabled: True'")
    print("   - 'Router aux loss coef: 0.001'")
    print("   - 'aux_loss': <value> in training metrics")
    print("   - No weight mapping errors during training")

if __name__ == "__main__":
    main() 