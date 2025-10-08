### IMPORTS ###
import os
os.environ["KERAS_BACKEND"] ="tensorflow"
import random
import keras
import gradio as gr
import time 
from src.inference import generate_chorale, draw_random_sample
from src.dataset import NoteEncoder
from src.metrics import Preplexity
from src.config import URL
from src.utils import get_dataset_path, midi_to_wave, load_css, load_markdown

### SETUP ###
ROOT_DIR = os.getcwd()
TRAIN_PATH, VAL_PATH, ARTIFACTS_PATH, MODEL_PATH = get_dataset_path(ROOT_DIR, URL)
AUDIO_SAMPLES_PATH = os.path.join(ROOT_DIR, "samples")
os.makedirs(AUDIO_SAMPLES_PATH, exist_ok=True)
midi_path = os.path.join(AUDIO_SAMPLES_PATH, "sample.mid")
wav_path = os.path.join(AUDIO_SAMPLES_PATH, "sample.wav")


### DOWNLOAD SF2 MUSIC FONT ###
sf2_download_path = keras.utils.get_file(
        "FluidR3_GM.zip",
        "https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip",
        extract= True,
        cache_dir= ARTIFACTS_PATH,
        cache_subdir= ""
        )
SF2_PATH = os.path.join(sf2_download_path, "FluidR3_GM.sf2")


### LOAD MODEL & ENCODERS ###

model = keras.models.load_model(os.path.join(MODEL_PATH, "bach_model.keras"),
                                custom_objects={"Preplexity": Preplexity})

note2id, id2note, vocab = NoteEncoder(vocab_path=ARTIFACTS_PATH, samples_path=None)


### GRADIO ASSETS ###
theme = gr.themes.Soft( font=[gr.themes.GoogleFont("Vazirmatn"),"Segoe UI", "system-ui"])
css = load_css()
english_summary = load_markdown("english_summary")
persian_summary = load_markdown("persian_summary")
english_help = load_markdown("english_help")
persian_help = load_markdown("persian_help")
english_title = "# BachNet: AI-Generated Bach Music"
persian_title = "# باخ‌نت: خلق موسیقی مشابه باخ با هوش مصنوعی"

### GENERATION FUNCTIONS ###
def pick_random_seed():
    path = draw_random_sample(VAL_PATH, seed=random.randint(0, 9999))
    return path, os.path.basename(path)

def generate_fn(seed_path, seed_len, gen_len, temp):
    sample_rows = slice(0, seed_len)

    print("=== Generation started ===")
    t0 = time.time()
    generate_chorale(
        model=model,
        sample_seed_path=seed_path,
        note2id=note2id,
        id2note=id2note,
        file_name=midi_path,
        max_len=gen_len,
        temperature=temp,
        sample_seed_rows=sample_rows
    )
    t1 = time.time()
    print(f"generate_chorale took {t1 - t0:.2f}s")

    midi_to_wave(midi_file_path=midi_path, SF2_PATH=SF2_PATH, wave_path=wav_path)
    return wav_path

def set_english():
    return (gr.update(value=english_title, elem_classes=[]),
            gr.update(value=english_summary, elem_classes=[]),
            gr.update(value=english_help, elem_classes=[]))
    
def set_persian():
    return (gr.update(value=persian_title, elem_classes=['persian']),
            gr.update(value=persian_summary, elem_classes=['persian']),
            gr.update(value=persian_help, elem_classes=['persian']))


### GRADIO APP ###
with gr.Blocks(css=css, title="BachNet", theme=theme) as demo:
    title_md = gr.Markdown(english_title, elem_id="title")
    
    with gr.Row():
        english_btn = gr.Button("English")
        persian_btn = gr.Button("Persian (فارسی)")
    
    summary_md = gr.Markdown(english_summary, elem_id="summary", max_height=None)
    
    with gr.Row(variant="panel"):
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("## Customize Your Chorale")
            with gr.Row():
                sample_seed_btn = gr.Button("Pick Random Seed", variant="primary")
                hidden_full_path = gr.State()
                seed_path_box = gr.Textbox(label="Selected Seed Path", interactive=False)
            
            seed_len_slider = gr.Slider(40, 80, 50, step=1, label="Seed Length")
            gen_len_slider = gr.Slider(20, 100, 30, step=1, label="Generated Length (Chords)")
            temp_slider = gr.Slider(0.5, 1.8, 0.9, step=0.1, label="Temperature")
            
            generate_btn = gr.Button("Generate", variant="primary")
        
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("## Generated Music: Listen & Download")
            gr.Markdown("⚠️ *Note: Running on CPU — generation may take ~15 seconds on default settings.*",elem_id="cpu_warning")
            audio_player = gr.Audio(label="Generated Chorale", type="filepath",
                                    interactive=False, show_download_button=True, streaming=True, autoplay=True)
            help_md = gr.Markdown(english_help, elem_id="help_text")


### EVENTS ###
    demo.load(pick_random_seed, outputs=[hidden_full_path, seed_path_box])
    sample_seed_btn.click(pick_random_seed, outputs=[hidden_full_path, seed_path_box])
    generate_btn.click(generate_fn, inputs=[hidden_full_path, seed_len_slider, gen_len_slider, temp_slider],
    outputs=audio_player)
    english_btn.click(set_english, outputs=[title_md, summary_md, help_md])
    persian_btn.click(set_persian, outputs=[title_md, summary_md, help_md])

### LAUNCH APP ###
if __name__ == "__main__":
    demo.launch()
