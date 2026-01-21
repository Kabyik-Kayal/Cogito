import os
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
MISTRAL_GGUF_MODEL_PATH = MODEL_DIR / "models--mistralai--Ministral-3-8B-Reasoning-2512-GGUF/snapshots/4413bc9b7b091ddad28101650d1395f5e045a328/Ministral-3-8B-Reasoning-2512-Q8_0.gguf"
