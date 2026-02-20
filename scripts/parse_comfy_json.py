import json

def parse_comfy(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print("--- ComfyUI Subgraphs/Nodes ---")
    if "nodes" in data:
        for node in data["nodes"]:
            print(f"Node ID: {node.get('id')} | Type: {node.get('type')}")
            if "widgets_values" in node:
                print(f"   Values: {node['widgets_values'][:2]}") # Print first two to see what it is
            
    # Normally ComfyUI API format is a dictionary of "NodeID": {"inputs": {...}, "class_type": "..."}
    # But this file seems to be the UI Save format (with "nodes" array and "links").
    # We need to know if it has a way to extract the API format.
    print(f"Format Keys: {list(data.keys())}")
    
parse_comfy("c:/Users/Nyx/Desktop/MANGACOLOR/Flux2_klein_image_edit_9b_base.json")
