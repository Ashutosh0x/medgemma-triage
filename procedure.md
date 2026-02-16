Project task prompt — build TECHNICAL PROTOTYPE then WORKING DEMO (separate folders)

Goal: produce a reproducible, safety-first MedGemma-based submission for the MedGemma Impact Challenge that first delivers an accurate technical prototype (research-quality, reproducible notebooks + model artifacts) and then a working demo app (UI + containerized inference). Place prototype and demo in separate top-level folders: /prototype and /demo_app.

Context (include in repo README / writeup)

Short summary to include verbatim in README and writeup:

Use at least one HAI-DEF model (MedGemma). Companion models: MedSigLIP (image encoder), MedASR (speech).

Target use-case: Chest X-Ray triage + explainable report (image ± prior report input → urgency label, short explanation, annotated crops + provenance).

Datasets: MIMIC-CXR / CheXpert (use public subsets, follow dataset TOU).

Key goals: offline-capable pipeline, provenance for each output, safety disclaimers, reproducibility (Docker + notebooks).

(You may paste the earlier long feature list into README_CONTEXT.md — it contains the detailed MedGemma/HAI-DEF capabilities, deployment notes, and safety checklist.)

Top-level deliverables

/prototype — accurate technical prototype (research notebooks + model fine-tuning / eval).

/demo_app — working demo app (containerized inference service + minimal UI or Gradio/Streamlit app).

writeup.pdf — ≤3 pages, ready for Kaggle Writeup (use template headings).

video.mp4 — ≤3 minutes demo (store in /video).

TERMS.md, LICENSE, CITATION.md (include HAI-DEF Terms of Use reference and dataset citations).

Folder structure (exact)
/prototype
  /data_scripts
    download_datasets.sh
    prepare_mimic_subset.py
  /notebooks
    01_quickstart_inference.ipynb
    02_fine_tune_medgemma.ipynb
    03_evaluation_and_metrics.ipynb
  /models
    README_models.md  # which weights were used and how to obtain
  /eval
    eval_metrics.py
    test_examples.jsonl
  Dockerfile.prototype
  requirements.txt

/demo_app
  /service
    app.py            # flask/fastapi server serving on /predict
    model_loader.py
    predict.py
  /ui
    gradio_app.py or streamlit_app.py
  /onnx
    medgemma_quant.onnx (placeholder / download script)
  /docker
    Dockerfile
    entrypoint.sh
  /tests
    smoke_test.sh
  README.md

/video
  demo.mp4
/writeup.pdf
/TERMS.md
/LICENSE
/CITATION.md
README.md

Prototype requirements (what to build first)

Place everything for reproduction under /prototype.

Purpose

Produce an accurate technical prototype that demonstrates:

correct MedGemma inference (image+optional text → text output),

a fine-tuning recipe for urgency classification and short explanation generation,

evaluation notebooks & scripts that reproduce the key metrics.

Mandatory files

01_quickstart_inference.ipynb: shows how to run inference on 5 examples (include code to load MedGemma 4B multimodal via HF transformers or local weights).

02_fine_tune_medgemma.ipynb: single clear fine-tuning recipe (hyperparams, training loop) for: (a) urgency classifier, (b) explanation generation head.

03_evaluation_and_metrics.ipynb: computes AUC, sensitivity@recall, PPV for triage; shows sample explanations and a small clinician-rating interface (simple CSV).

eval/eval_metrics.py: CLI to reproduce metrics on held-out test set.

data_scripts/download_datasets.sh and prepare_mimic_subset.py: scripted instructions to obtain allowed public subsets (point to dataset TOU).

Dockerfile.prototype and requirements.txt: create environment that can run the notebooks (CPU/GPU variants).

models/README_models.md: exact model names, expected size, and commands to download (no redistributed base weights).

Acceptance criteria (prototype)

Quickstart notebook runs end-to-end on 5 example images in ≤30 minutes (judges can test).

Fine-tune notebook documents hyperparams and can be re-run to produce a saved fine-tuned checkpoint.

Evaluation notebook prints: AUC, sensitivity at target recall (e.g., 95%), and shows 10 example outputs with provenance (image crops or referenced report snippet).

Include a short prototype.md summary listing limitations and next steps for demo.

Demo app requirements (after prototype)

Place production-ready demo in /demo_app.

Purpose

Ship a simple, polished demo that runs the prototype model in a container and provides a UI where a user can upload a CXR (PNG/DICOM) and optionally paste prior report text, then get:

urgency label (Urgent / Non-Urgent),

one-line textual explanation,

1–3 image crops highlighting evidence,

provenance text (if prior report used, include quoted snippet),

“This is decision support only” disclaimer.

Technical constraints

The model should be quantized and packaged (ONNX int8 recommended) or use torch + quantization to be runnable on a laptop or Jetson.

Container must expose a simple REST endpoint /predict and have a minimal UI (/ui via Gradio/Streamlit).

Include smoke_test.sh that runs sample inference against the container and verifies JSON schema.

Mandatory files

service/app.py (FastAPI or Flask) with /predict and status endpoints.

ui/gradio_app.py or streamlit_app.py for browser demo.

docker/Dockerfile: instructions to build the demo container (use base GPU or CPU image; include ONNX runtime).

onnx/ or models/ placeholders and script download_quantized_model.sh or build_onnx.sh.

tests/smoke_test.sh verifying end-to-end inference and UI reachable.

README.md with one-line quickstart, Docker build & run commands, and how to run smoke tests.

Acceptance criteria (demo)

docker/Dockerfile builds a working image that when run exposes UI on port 7860 (Gradio) or app port and responds to /predict.

The demo returns valid JSON with fields: {label, score, explanation, crops:[{x,y,w,h,thumb_path}], provenance:[{source, snippet}]}.

Smoke test passes (script included).

The UI demonstrates latency and model footprint (e.g., “quantized ONNX size: X MB; inference time: Y ms on laptop”).

Include video/demo.mp4 showing the UI flow + metrics callouts.

Code/implementation specifics (copy into tasks)

Model: start from google/medgemma-4b-it (or latest 4B multimodal variant). Use the MedSigLIP encoder for image tasks where appropriate. Integrate MedASR only if audio demo included.

Frameworks: Hugging Face Transformers + Accelerate; PyTorch for fine-tuning; ONNX Runtime for demo inference. Use torch.compile/accelerate where helpful. For small-edge, use quantization via onnxruntime.quantization (dynamic or static int8).

Data: use de-identified MIMIC-CXR / CheXpert subsets; provide download_datasets.sh that instructs judges how to obtain data (not redistribute).

Evaluation: scripts must compute AUC, sensitivity @ recall threshold, PPV, IoU for any localization. Include radgraph or RadGraph-style entity extraction if available, but make human-rating CSV possible.

Provenance: when generating explanation, include the top-k training examples or report snippets used to support the claim (or annotate an image crop). Save crop thumbnails to /demo_app/static/crops/.

Safety: include TERMS.md with HAI-DEF TOU and an explicit disclaimer: “For clinical decision support only — not a diagnostic tool.”

Reproducible commands (examples to include in README)

Prototype quickstart:

# build prototype environment and run example inference
cd prototype
docker build -f Dockerfile.prototype -t medproto .
docker run --rm -v $(pwd)/notebooks:/work/notebooks medproto jupyter nbconvert --execute notebooks/01_quickstart_inference.ipynb --to html


Demo quickstart:

cd demo_app/docker
docker build -t medgemma-demo .
docker run --gpus all -p 7860:7860 medgemma-demo
# smoke test
./tests/smoke_test.sh


Evaluation:

python prototype/eval/eval_metrics.py --preds demo_outputs.jsonl --gold test_labels.jsonl

QA / acceptance checklist (attach to PR)

 Prototype notebooks run end-to-end and reproduce metrics.

 Fine-tuned checkpoint saved and documented.

 Demo container builds and smoke_test.sh passes.

 README includes quickstart, dataset TOU, and HAI-DEF Terms reference.

 TERMS.md, LICENSE, CITATION.md present.

 writeup.pdf (≤3 pages) prepared using the Kaggle template.

 video/demo.mp4 ≤3 minutes stored in /video.

Extra deliverables (bonus; include if time permits)

/demo_app/live_space — link + instructions for a Hugging Face Space (optional).

onnx/medgemma_quant.tracing.txt — trace showing model conversion steps and quantization logs.

Small clinician evaluation CSV with 10 rated samples.

Project timeline & priority (recommended)

Prototype (priority 1) — produce reproducible notebooks + evaluation (this is required for writeup).

Model quantization & API (priority 2) — create ONNX and simple service.

UI + video (priority 3) — polish UI, capture 3-min video.

Finalize writeup and attach repo + video on Kaggle.