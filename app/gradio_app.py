"""
Gradio demo: classify a single lung-windowed CT slice.

Usage (from the repository root):
    python -m app.gradio_app
    python -m app.gradio_app --adapter path/to/local/adapter

The demo runs locally at http://127.0.0.1:7860. Example images placed in
app/examples/ are shown below the upload box.

Research use only. Not for clinical use.
"""

import argparse
import os
from pathlib import Path

# Windows: the TorchInductor cache can fail with "PermissionError: [WinError 5]"
# when it renames a folder in %TEMP%. Disabling it only costs a few seconds
# of recompilation on the first prediction.
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

from inference.predictor import ADAPTER, BASE_MODEL, FibrosisPredictor  # noqa: E402  (Unsloth must load first)

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

EXAMPLES_DIR = Path(__file__).parent / "examples"

DISCLAIMER = """
**Research use only. Not for clinical use.**
This proof-of-concept was trained on public datasets and has not been clinically validated.
Do not use it for patient care or diagnostic decisions.
"""

ABOUT = """
**Model.** MedGemma 1.5 4B with a LoRA adapter, fine-tuned on a consumer laptop (8 GB GPU).

**Input.** One lower-lobe chest CT slice, displayed with a lung window (about -1400 to 240 HU),
as PNG or JPEG. The model was trained on axial slices from the lower 15 to 40% of the lung.

**Held-out results** (81 patients, 7-slice majority vote, neutral prompt): accuracy 88.9%,
sensitivity 90.6%, specificity 87.8%. This demo classifies a single slice, which is less
reliable than the patient-level vote.

**Confidence** is the mean probability of the generated tokens. It indicates how decisive the
model was and is not a calibrated probability of disease.
"""


def build_demo(predictor: FibrosisPredictor) -> gr.Blocks:
    """Assemble the interface around a loaded predictor."""

    def analyse(image: Image.Image):
        if image is None:
            return "Please upload a CT slice.", None, ""
        prediction = predictor.predict(square_and_resize(image))
        return prediction.answer, round(prediction.confidence * 100, 1), prediction.raw_response

    with gr.Blocks(title="MedGemma CT Fibrosis") as demo:
        gr.Markdown("# MedGemma CT Fibrosis")
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="CT slice (PNG or JPEG)")
                run_button = gr.Button("Analyse", variant="primary")
                examples = sorted(str(p) for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
                if examples:
                    gr.Examples(examples=examples, inputs=image_input)
            with gr.Column():
                answer_output = gr.Textbox(label="Prediction", interactive=False)
                confidence_output = gr.Number(label="Confidence (%)", interactive=False, precision=1)
                raw_output = gr.Textbox(label="Raw model response", interactive=False)
                gr.Markdown(ABOUT)

        run_button.click(analyse, inputs=image_input,
                         outputs=[answer_output, confidence_output, raw_output])
    return demo


def square_and_resize(image: Image.Image, min_size: int = 512, max_size: int = 896) -> Image.Image:
    """
    Bring an uploaded image to a square RGB image between 512 and 896 pixels.

    Images inside that range keep their resolution. Smaller images are
    enlarged and larger ones reduced, keeping the aspect ratio. Non-square
    images are centred on a black square background.
    """
    image = image.convert("RGB")
    longest = max(image.size)
    if longest < min_size or longest > max_size:
        scale = (min_size if longest < min_size else max_size) / longest
        image = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.Resampling.LANCZOS)

    side = max(image.size)
    if image.width != image.height:
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        image = canvas
    return image


def main():
    parser = argparse.ArgumentParser(description="MedGemma CT Fibrosis demo")
    parser.add_argument("--adapter", default=ADAPTER, help="Hugging Face repo or local adapter folder")
    parser.add_argument("--base-model", default=BASE_MODEL, help="4-bit base model (Hugging Face repo or local folder)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_demo(FibrosisPredictor(adapter=args.adapter, base_model=args.base_model))
    demo.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
