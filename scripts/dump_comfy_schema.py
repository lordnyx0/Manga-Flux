import urllib.request, json, time

url = "http://127.0.0.1:8188/object_info"
for i in range(15):
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            break
    except Exception:
        time.sleep(2)
else:
    print("Failed to connect to ComfyUI.")
    exit(1)

print("CLIPLoader definition:")
print(json.dumps(data.get("CLIPLoader", {}), indent=2))

print("\nDualCLIPLoader definition:")
print(json.dumps(data.get("DualCLIPLoader", {}), indent=2))

print("\nNodes related to Qwen or Flux:")
for node_name in data:
    if "qwen" in node_name.lower() or "flux" in node_name.lower():
        print(f"- {node_name}")
