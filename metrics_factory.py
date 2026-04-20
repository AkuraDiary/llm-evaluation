from config import METRICS_CONFIG

def create_metrics():
    return [
        {
            "instance": cfg["class"](threshold=cfg["threshold"]),
            "threshold": cfg["threshold"],
            "weight": cfg["weight"],
        }
        for cfg in METRICS_CONFIG
    ]

def create_weights_map():
    return {
        cfg["class"].__name__: cfg["weight"]
        for cfg in METRICS_CONFIG
    }
    
if __name__ == "__main__":
    print(create_metrics())