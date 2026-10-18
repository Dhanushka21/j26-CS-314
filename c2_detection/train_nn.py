"""STEP 8: train the neural networks.

  python c2_detection/train_nn.py --model mlp     single flow   (44,)     - baseline
  python c2_detection/train_nn.py --model lstm    10-flow window (10, 44)  - sequence model

Output: models/<model>.pt, results/<model>_metrics.json, _per_class.json, _confusion.csv, _history.json
"""
import argparse
import json
import time

import numpy as np
import torch

import config as C
from src.metrics import evaluate, single_event_latency_ms
from src.nets import LSTMClassifier, MLP, predict_proba, train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "lstm"], required=True)
    kind = ap.parse_args().model
    name = f"{kind}{C.RUN_TAG}"

    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)

    d = np.load(C.PROCESSED_DIR / "dataset.npz")
    idx = C.MODEL_FEATURE_IDX
    if kind == "mlp":       # newest flow only
        x_tr, x_val, x_te = (np.ascontiguousarray(d[k][:, -1, idx]) for k in ("X_train", "X_val", "X_test"))
        model = MLP(len(idx), len(C.CLASSES))
    else:                   # full window
        x_tr, x_val, x_te = (np.ascontiguousarray(d[k][:, :, idx]) for k in ("X_train", "X_val", "X_test"))
        model = LSTMClassifier(len(idx), len(C.CLASSES))
    y_tr, y_val, y_te = d["y_train"], d["y_val"], d["y_test"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"== {name.upper()}  input={x_tr.shape[1:]}  params={n_params:,}  "
          f"epochs<={C.EPOCHS}  dropped={C.DROP_FEATURES}")

    t0 = time.time()
    model, history = train_model(model, x_tr, y_tr, x_val, y_val,
                                 C.EPOCHS, C.BATCH_SIZE, C.LR, C.PATIENCE)
    print(f"trained in {time.time() - t0:.0f}s")

    proba = predict_proba(model, x_te)
    model.cpu().eval()
    one = torch.from_numpy(x_te[:1]).float()
    with torch.no_grad():
        latency = single_event_latency_ms(lambda s: model(s), one)
    evaluate(name, y_te, proba, latency)

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), C.MODELS_DIR / f"{name}.pt")
    (C.RESULTS_DIR / f"{name}_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nsaved models/{name}.pt")


if __name__ == "__main__":
    main()