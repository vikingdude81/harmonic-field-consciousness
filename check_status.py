"""Check status of HarmonicFieldAI system"""
import requests
import socket
from datetime import datetime

print("=" * 60)
print(f"HarmonicFieldAI Diagnostic - {datetime.now()}")
print("=" * 60)

# Check Qwen server
print("\n1. Qwen Server (port 8001):")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('localhost', 8001))
    if result == 0:
        print("   ✅ Port 8001 is OPEN")
        r = requests.get("http://localhost:8001/v1/models", timeout=5)
        print(f"   ✅ Model: {r.json()['data'][0]['id']}")
    else:
        print("   ❌ Port 8001 is CLOSED - Qwen server not running")
        print("   → Run: start_qwen_service.bat")
    sock.close()
except Exception as e:
    print(f"   ❌ Error: {e}")

# Check Daemon
print("\n2. HarmonicFieldAI Daemon (port 8080):")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('localhost', 8080))
    if result == 0:
        print("   ✅ Port 8080 is OPEN")
        r = requests.get("http://localhost:8080/status", timeout=5)
        data = r.json()
        print(f"   ✅ LLM Ready: {data.get('llm_ready')}")
        print(f"   ✅ Posts: {data.get('posts_this_session')}")
        print(f"   ✅ Last heartbeat: {data.get('last_heartbeat')}")
    else:
        print("   ❌ Port 8080 is CLOSED - Daemon not running")
        print("   → Run: python harmonic_daemon.py")
    sock.close()
except Exception as e:
    print(f"   ❌ Error: {e}")

# Check Moltbook
print("\n3. Moltbook Account:")
try:
    headers = {"Authorization": "Bearer moltbook_sk_vX0tbXPZMS45Y90BvlAEa84CXgzua-v1"}
    r = requests.get("https://www.moltbook.com/api/v1/agents/me", headers=headers, timeout=10)
    if r.status_code == 200:
        data = r.json()
        print(f"   ✅ Agent: {data.get('name', 'unknown')}")
        print(f"   ✅ Claimed: {data.get('is_claimed', False)}")
        posts = data.get('recentPosts', [])
        print(f"   ✅ Recent posts: {len(posts)}")
        if posts:
            for p in posts[:3]:
                title = p.get('title', 'no title')[:50]
                created = p.get('created_at', '?')[:10]
                print(f"      - [{created}] {title}...")
    elif r.status_code == 429:
        print("   ⏳ Rate limited")
    else:
        print(f"   ❌ Status: {r.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("RECOMMENDATION:")
if result != 0:
    print("→ The daemon is NOT running. Start it with:")
    print("  cd C:\\Users\\akbon\\OneDrive\\Documents\\GitHub\\LlamaFactory")
    print("  python harmonic_daemon.py")
else:
    print("→ System is running!")
print("=" * 60)
