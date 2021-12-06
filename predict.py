import torch
import argparse
import time
from pathlib import Path

from utils import load_setting, load_tokenizer
from models import SwinTransformerOCR
from dataset import CustomCollate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="settings/default.yaml",
                        help="Experiment settings")
    parser.add_argument("--target", "-t", type=str, required=True,
                        help="OCR target (image or directory)")
    parser.add_argument("--tokenizer", "-tk", type=str, required=True,
                        help="Load pre-built tokenizer")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="Load model weight in checkpoint")
    args = parser.parse_args()

    cfg = load_setting(args.setting)
    cfg.update(vars(args))
    print("setting:", cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load
    tokenizer = load_tokenizer(cfg.tokenizer)
    model = SwinTransformerOCR(cfg, tokenizer)
    saved = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(saved['state_dict'])
    collate = CustomCollate(cfg, tokenizer=tokenizer)

    target = Path(cfg.target)
    if target.is_dir():
        target = list(target.glob("*.jpg")) + list(target.glob("*.png"))
    else:
        target = [target]

    for image_fn in target:
        start = time.time()
        x = collate.ready_image(image_fn)
        print("[{}]sec | image_fn : {}".format(time.time()-start, model.predict(x)))
