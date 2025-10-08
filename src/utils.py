import keras
import os
import subprocess
from pathlib import Path

def get_dataset_path(root_dir, URL):
    """Downlods chorales csv dataset and confgirues files path"""
    DATASET_PATH = keras.utils.get_file(
        "jsb_chorales.zip",
        URL,
        extract= True,
        cache_dir= root_dir,
        cache_subdir= "data"
        )
    
    TRAIN_PATH = os.path.join(DATASET_PATH, "jsb_chorales/train")
    VAL_PATH = os.path.join(DATASET_PATH, "jsb_chorales/val")
    ARTIFACTS_PATH = os.path.join(root_dir, "artifacts")
    MODEL_PATH = os.path.join(root_dir, "model")
    
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    return TRAIN_PATH, VAL_PATH, ARTIFACTS_PATH, MODEL_PATH

def midi_to_wave(midi_file_path, SF2_PATH, wave_path="samples/sample.wav"):
    """Converts a MIDI file to a WAV audio file using FluidSynth."""
    if not os.path.exists(midi_file_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_file_path}")
    if not os.path.exists(SF2_PATH):
        raise FileNotFoundError(f"SoundFont file not found: {SF2_PATH}")
    
    os.makedirs(os.path.dirname(wave_path), exist_ok=True)
    cmd = ["fluidsynth", "-ni", "-F", wave_path, "-r", "44100", SF2_PATH, midi_file_path]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FluidSynth failed: {e.stderr}")
    
    print(f"WAV file saved at {wave_path}")

ASSETS_DIR = Path(__file__).parent.parent / "assets"

def load_css():
    return (ASSETS_DIR / "css/theme.css").read_text(encoding="utf-8")

def load_markdown(name):
    return (ASSETS_DIR / f"markdown/{name}.md").read_text(encoding="utf-8")