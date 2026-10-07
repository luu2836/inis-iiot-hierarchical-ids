from tune_nn import evaluate
from config import TON_NET, EDGE_ML

configs = [
    {"tag": "noW h512 bn ep150", "hidden": (512, 256, 128), "epochs": 150, "mode": "none", "bn": True},
    {"tag": "noW h768x3 bn ep120", "hidden": (768, 384, 192), "epochs": 120, "mode": "none", "bn": True},
    {"tag": "noW h1024x3 bn ep120", "hidden": (1024, 512, 256), "epochs": 120, "mode": "none", "bn": True},
]

if __name__ == "__main__":
    evaluate("ToN-IoT(network)", TON_NET, "type", configs)
