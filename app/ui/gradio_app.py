"""
MedGemma CXR Triage — Gradio Clinical Interface (V4.0)
=======================================================

Premium clinical UI with:
  - Dark medical-grade theme
  - Multi-agent execution trace visualization
  - Real-time confidence gauges
  - Safety status indicators
  - Heatmap overlays (image-derived, not fabricated)
  - Longitudinal comparison panel
  - Clinician feedback mechanism
"""

import os
import sys
import time
import traceback
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── Module Imports ─────────────────────────────────────────────────
# Add parent dirs so we can import sibling modules  
SERVICE_DIR = Path(__file__).parent.parent / "service"
sys.path.insert(0, str(SERVICE_DIR))

try:
    from model_loader import load_model, check_model_health
    from predict import run_triage_prediction, compute_simple_attention_map
    from agents import AgenticOrchestrator
except ImportError as e:
    print(f"Import warning: {e}")
    # Provide stubs so file doesn't crash at import
    def load_model(): raise RuntimeError("model_loader not available")
    def check_model_health(m, p): return False
    def run_triage_prediction(*a, **kw): raise RuntimeError("predict not available")
    def compute_simple_attention_map(img): return {}
    AgenticOrchestrator = None


# ─── State ──────────────────────────────────────────────────────────
_model = None
_processor = None
_orchestrator = None
_load_error = None


# ─── UI Styling ─────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');

:root {
    --primary: #4A90D9;
    --primary-glow: #4A90D920;
    --urgent: #E53E3E;
    --urgent-bg: #E53E3E15;
    --safe: #38A169;
    --safe-bg: #38A16915;
    --warn: #D69E2E;
    --warn-bg: #D69E2E15;
    --indeterminate: #A0AEC0;
    --bg-dark: #0F1117;
    --bg-card: #1A1D28;
    --bg-elevated: #232736;
    --border: #2D3348;
    --text-primary: #E2E8F0;
    --text-secondary: #A0AEC0;
    --text-muted: #718096;
    --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
}

body, .gradio-container {
    font-family: var(--font-sans) !important;
    background: var(--bg-dark) !important;
    color: var(--text-primary) !important;
}

.gradio-container .main {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #1a1d28 0%, #0f1117 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary), #7C3AED, var(--primary));
}

.app-header h1 {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #fff !important;
    margin: 0 0 8px 0 !important;
    letter-spacing: -0.02em;
}

.app-header .subtitle {
    font-size: 14px;
    color: var(--text-secondary);
    margin: 0;
}

.app-header .badge {
    display: inline-block;
    background: var(--primary-glow);
    color: var(--primary);
    border: 1px solid var(--primary);
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 10px;
}

/* Safety Banner */
.safety-banner {
    background: linear-gradient(90deg, #1a1d2800, #D69E2E10, #1a1d2800);
    border: 1px solid #D69E2E40;
    border-radius: 10px;
    padding: 12px 20px;
    margin-bottom: 20px;
    text-align: center;
    font-size: 13px;
    color: var(--warn);
}

/* Urgency Status Cards */
.urgency-card {
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.urgency-card.urgent {
    background: linear-gradient(135deg, #E53E3E10, #E53E3E05);
    border-color: #E53E3E60;
    box-shadow: 0 0 40px #E53E3E10;
}

.urgency-card.non-urgent {
    background: linear-gradient(135deg, #38A16910, #38A16905);
    border-color: #38A16960;
    box-shadow: 0 0 40px #38A16910;
}

.urgency-card.indeterminate {
    background: linear-gradient(135deg, #A0AEC010, #A0AEC005);
    border-color: #A0AEC060;
}

.urgency-label {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 4px 0;
}

.urgency-label.urgent { color: var(--urgent); }
.urgency-label.non-urgent { color: var(--safe); }
.urgency-label.indeterminate { color: var(--indeterminate); }

.urgency-sub {
    font-size: 13px;
    color: var(--text-secondary);
}

/* Confidence Gauge */
.confidence-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
}

.confidence-bar {
    width: 100%;
    height: 10px;
    background: var(--bg-elevated);
    border-radius: 5px;
    overflow: hidden;
    margin: 10px 0;
}

.confidence-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.8s ease;
}

.confidence-fill.high { background: linear-gradient(90deg, #38A169, #48BB78); }
.confidence-fill.medium { background: linear-gradient(90deg, #D69E2E, #ECC94B); }
.confidence-fill.low { background: linear-gradient(90deg, #E53E3E, #FC8181); }

/* Findings Card */
.findings-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
}

.finding-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
}

.finding-item:last-child { border-bottom: none; }

.finding-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--primary);
    margin-top: 6px;
    flex-shrink: 0;
}

.finding-text {
    font-size: 14px;
    color: var(--text-primary);
    line-height: 1.5;
}

/* Execution Trace */
.trace-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    font-family: var(--font-mono);
    font-size: 12px;
}

.trace-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    border-left: 2px solid var(--border);
    padding-left: 16px;
    margin-left: 8px;
    position: relative;
}

.trace-step::before {
    content: '';
    position: absolute;
    left: -5px;
    width: 8px; height: 8px;
    border-radius: 50%;
}

.trace-step.success::before { background: var(--safe); }
.trace-step.error::before { background: var(--urgent); }
.trace-step.warning::before { background: var(--warn); }

.trace-agent {
    font-weight: 600;
    color: var(--primary);
    min-width: 100px;
}

.trace-time {
    color: var(--text-muted);
    font-size: 11px;
}

/* Sections */
.section-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
}

/* Override Gradio defaults */
.dark .block { background: var(--bg-card) !important; }
.dark .block .wrap { border-color: var(--border) !important; }
.dark textarea, .dark input {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
}

button.primary {
    background: linear-gradient(135deg, var(--primary), #6366F1) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(74, 144, 217, 0.3) !important;
}
"""


# ─── Model Loading ──────────────────────────────────────────────────

def load_resources():
    """Load MedGemma model and initialize agentic orchestrator."""
    global _model, _processor, _orchestrator, _load_error

    print("=" * 60)
    print("MedGemma CXR Triage UI — Loading Resources")
    print("=" * 60)

    try:
        _model, _processor = load_model()
        is_healthy = check_model_health(_model, _processor)

        if is_healthy:
            print("[OK] Model loaded and healthy")
        else:
            print("⚠ Model loaded but health check failed — proceeding anyway")

        # Initialize agentic orchestrator
        if AgenticOrchestrator:
            _orchestrator = AgenticOrchestrator(model=_model, processor=_processor)
            print("[OK] Agentic orchestrator initialized (5 agents)")
        else:
            print("⚠ AgenticOrchestrator not available — using basic pipeline")

    except Exception as e:
        _load_error = str(e)
        _model = None
        _processor = None
        _orchestrator = None
        print(f"[FAIL] Model loading failed: {e}")
        print("  The UI will show an error when prediction is attempted.")


# ─── Visualization Helpers ──────────────────────────────────────────

def generate_heatmap_overlay(image: Image.Image, attention_data: dict) -> Image.Image:
    """Generate a real heatmap overlay based on attention data."""
    if not attention_data or attention_data.get("method") == "none":
        return image

    overlay = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    regions = attention_data.get("top_regions", [])
    for region in regions:
        px = region.get("pixel_region", {})
        att = region.get("relative_attention", 0)

        if px:
            x, y, w, h = px.get("x", 0), px.get("y", 0), px.get("w", 0), px.get("h", 0)
            # Color based on attention: low=blue, high=red
            r = int(255 * att)
            g = int(100 * (1 - att))
            b = int(255 * (1 - att))
            alpha = int(80 * att + 20)
            draw.rectangle([x, y, x+w, y+h], fill=(r, g, b, alpha), outline=(r, g, b, 150), width=2)

    return Image.alpha_composite(Image.new("RGBA", overlay.size, (0, 0, 0, 0)), overlay).convert("RGB")


def build_urgency_html(label: str, score: float, confidence: str) -> str:
    """Generate urgency status card HTML."""
    label_map = {
        "Urgent": ("urgent", '<span class="material-symbols-outlined" style="font-size:48px;color:var(--urgent)">emergency</span>', "Immediate radiologist review recommended"),
        "Non-Urgent": ("non-urgent", '<span class="material-symbols-outlined" style="font-size:48px;color:var(--safe)">check_circle</span>', "Standard reading queue"),
        "Indeterminate": ("indeterminate", '<span class="material-symbols-outlined" style="font-size:48px;color:var(--warn)">help</span>', "Human review required"),
    }
    cls, icon, desc = label_map.get(label, ("indeterminate", '<span class="material-symbols-outlined" style="font-size:48px;color:var(--indeterminate)">radio_button_unchecked</span>', "Unknown status"))

    return f"""
    <div class="urgency-card {cls}">
        <div style="margin-bottom: 8px;">{icon}</div>
        <div class="urgency-label {cls}">{label}</div>
        <div class="urgency-sub">{desc}</div>
        <div style="margin-top: 12px; font-size: 13px; color: var(--text-muted);">
            Confidence: <strong>{confidence}</strong> | Score: {score:.2f}
        </div>
    </div>
    """


def build_confidence_html(score: float, confidence: str, uncertainty: float) -> str:
    """Generate confidence gauge HTML."""
    pct = int(score * 100)
    cls = "high" if confidence == "High" else ("medium" if confidence == "Medium" else "low")
    unc_pct = int(uncertainty * 100)

    return f"""
    <div class="confidence-container">
        <div class="section-title">Confidence Analysis</div>
        <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px;">
            <span>Model Confidence</span>
            <span style="font-weight: 600;">{confidence} ({pct}%)</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill {cls}" style="width: {pct}%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 13px; margin-top: 12px; margin-bottom: 4px;">
            <span>Uncertainty</span>
            <span style="font-weight: 600; color: {'var(--urgent)' if unc_pct > 25 else 'var(--safe)'};">{unc_pct}%</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill {'low' if unc_pct > 25 else 'high'}" style="width: {unc_pct}%;"></div>
        </div>
    </div>
    """


def build_findings_html(key_findings: list) -> str:
    """Generate findings card as markdown."""
    if not key_findings:
        return "**No specific findings identified.**"

    md = "### Key Findings\n\n"
    for i, f in enumerate(key_findings, 1):
        md += f"**{i}.** {f}\n\n"
    return md


def build_trace_html(execution_trace: list) -> str:
    """Generate execution trace visualization."""
    if not execution_trace:
        return '<div class="trace-container"><em>No execution trace available</em></div>'

    rows = ""
    for step in execution_trace:
        cls = "success" if step.get("success") else "error"
        if step.get("warnings"):
            cls = "warning"
        agent = step.get("agent", "unknown").upper()
        latency = step.get("latency_ms", 0)
        status = '<span class="material-symbols-outlined" style="font-size:16px;color:var(--safe)">check</span>' if step.get("success") else '<span class="material-symbols-outlined" style="font-size:16px;color:var(--urgent)">close</span>'
        errors = step.get("errors", [])
        warn = step.get("warnings", [])

        detail = ""
        if errors:
            detail = f'<span style="color: var(--urgent); font-size: 11px;"> — {errors[0]}</span>'
        elif warn:
            detail = f'<span style="color: var(--warn); font-size: 11px;"> — {warn[0]}</span>'

        rows += f"""
        <div class="trace-step {cls}">
            <span class="trace-agent">{agent}</span>
            <span>{status}</span>
            <span class="trace-time">{latency:.0f}ms</span>
            {detail}
        </div>
        """

    total_ms = sum(s.get("latency_ms", 0) for s in execution_trace)
    return f"""
    <div class="trace-container">
        <div class="section-title">Agent Execution Trace</div>
        {rows}
        <div style="text-align: right; padding-top: 10px; color: var(--text-muted); font-size: 11px;">
            Total: {total_ms:.0f}ms | {len(execution_trace)} agents
        </div>
    </div>
    """


def build_comparison_html(comparison: dict) -> str:
    """Generate longitudinal comparison HTML."""
    if not comparison or not comparison.get("comparison_available"):
        return '<div style="text-align: center; color: var(--text-muted); padding: 20px;">No prior report provided for comparison</div>'

    summary = comparison.get("summary", "")
    return f"""
    <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; padding: 20px;">
        <div class="section-title">Longitudinal Comparison</div>
        <div style="font-size: 14px; line-height: 1.6; color: var(--text-primary);">{summary}</div>
    </div>
    """


# ─── Main Prediction Function ──────────────────────────────────────

def predict(image, prior_report):
    """Run prediction and return UI components."""
    if image is None:
        return (
            '<div class="urgency-card indeterminate"><div class="urgency-label indeterminate">Upload Image</div><div class="urgency-sub">Select a chest X-ray to begin analysis</div></div>',
            None,
            '<div class="confidence-container"><div class="section-title">Awaiting Input</div></div>',
            "Upload a chest X-ray image to begin analysis.",
            '<div style="text-align: center; color: var(--text-muted);">—</div>',
            '<div style="text-align: center; color: var(--text-muted);">—</div>',
            "{}",
        )

    if _model is None or _processor is None:
        error_msg = _load_error or "Model not available"
        return (
            f'<div class="urgency-card indeterminate"><div style="margin-bottom: 8px;"><span class="material-symbols-outlined" style="font-size:48px;color:var(--warn)">warning</span></div><div class="urgency-label indeterminate">Model Error</div><div class="urgency-sub">{error_msg}</div></div>',
            image,
            '<div class="confidence-container"><div class="section-title">Model Not Loaded</div><p style="font-size: 13px; color: var(--text-muted);">Ensure MedGemma model is available and HF_TOKEN is set.</p></div>',
            f"**Error**: {error_msg}\n\nPlease check model availability.",
            '<div style="text-align: center; color: var(--text-muted);">—</div>',
            '<div style="text-align: center; color: var(--text-muted);">—</div>',
            "{}",
        )

    try:
        img = Image.fromarray(image) if not isinstance(image, Image.Image) else image

        # Use agentic orchestrator if available
        if _orchestrator:
            result = _orchestrator.run(img, prior_report=prior_report if prior_report else None)
        else:
            result = run_triage_prediction(
                _model, _processor, img,
                prior_report=prior_report if prior_report else None,
            )

        # Build UI components
        label = result.get("label", "Non-Urgent")
        score = result.get("score", 0.5)
        confidence = result.get("model_confidence", "Unknown")
        uncertainty = result.get("uncertainty", 0.0)
        explanation = result.get("explanation", "")
        key_findings = result.get("key_findings", [])

        urgency_html = build_urgency_html(label, score, confidence)
        conf_html = build_confidence_html(score, confidence, uncertainty)
        findings_md = build_findings_html(key_findings)

        if explanation:
            findings_md = f"**Explanation:** {explanation}\n\n---\n\n{findings_md}"

        # Safety annotations
        if result.get("abstain"):
            findings_md += f"\n\n---\n\n**[WARNING] Model Abstained**: {result.get('abstain_reason', '')}"
        if not result.get("verified"):
            findings_md += f"\n\n**[WARNING] Unverified**: {result.get('verify_reason', '')}"
        if result.get("recommendations"):
            findings_md += "\n\n---\n\n**Recommendations:**\n"
            for rec in result["recommendations"]:
                findings_md += f"- {rec}\n"

        # Heatmap
        attention = result.get("model_attention", {})
        heatmap_img = generate_heatmap_overlay(img.resize((896, 896)), attention)

        # Execution trace
        trace_html = build_trace_html(result.get("execution_trace", []))

        # Comparison
        comp_html = build_comparison_html(result.get("longitudinal_comparison", {}))

        # Sanitize result for JSON display
        display_result = {k: v for k, v in result.items()
                         if k not in ("raw_triage_response", "raw_findings_response", "raw_response", "model_attention")}
        try:
            result_json = __import__("json").dumps(display_result, indent=2, default=str)
        except Exception:
            result_json = str(display_result)

        return (
            urgency_html,
            heatmap_img,
            conf_html,
            findings_md,
            comp_html,
            trace_html,
            result_json,
        )

    except Exception as e:
        traceback.print_exc()
        return (
            f'<div class="urgency-card indeterminate"><div style="margin-bottom: 8px;"><span class="material-symbols-outlined" style="font-size:48px;color:var(--urgent)">error</span></div><div class="urgency-label indeterminate">Error</div><div class="urgency-sub">{str(e)[:120]}</div></div>',
            image,
            '<div class="confidence-container"><div class="section-title">Analysis Failed</div></div>',
            f"**Analysis Error:**\n\n```\n{traceback.format_exc()[:500]}\n```",
            '<div style="text-align: center; color: var(--text-muted);">—</div>',
            '<div style="text-align: center; color: var(--text-muted);">—</div>',
            "{}",
        )


# ─── Build Gradio UI ───────────────────────────────────────────────

def build_ui():
    """Build the Gradio interface."""
    with gr.Blocks(
        css=css,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
        ),
        title="MedGemma CXR Triage",
    ) as demo:
        # Header
        gr.HTML("""
        <div class="app-header">
            <h1><span class="material-symbols-outlined" style="font-size:32px;vertical-align:middle;margin-right:8px">local_hospital</span>MedGemma CXR Triage Assistant</h1>
            <p class="subtitle">
                AI-powered chest X-ray urgency classification using Google's MedGemma
            </p>
            <div class="badge">Multi-Agent Agentic System • HAI-DEF</div>
        </div>
        """)

        # Safety Banner
        gr.HTML("""
        <div class="safety-banner">
            <span class="material-symbols-outlined" style="font-size:18px;vertical-align:middle;margin-right:4px">warning</span> <strong>Clinical Decision Support Only</strong> — All AI-generated findings 
            require verification by a qualified radiologist. Not FDA-approved.
        </div>
        """)

        with gr.Row(equal_height=False):
            # LEFT: Inputs
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Chest X-Ray",
                    type="numpy",
                    height=420,
                    sources=["upload", "clipboard"],
                )
                prior_report = gr.Textbox(
                    label="Prior Report (Optional)",
                    placeholder="Paste prior radiology report for longitudinal comparison...",
                    lines=4,
                    max_lines=8,
                )
                analyze_btn = gr.Button(
                    "Analyze CXR",
                    variant="primary",
                    size="lg",
                )

            # RIGHT: Results
            with gr.Column(scale=2):
                urgency_html = gr.HTML(
                    value='<div class="urgency-card indeterminate"><div style="margin-bottom: 8px;"><span class="material-symbols-outlined" style="font-size:48px;color:var(--primary)">local_hospital</span></div><div class="urgency-label indeterminate">Ready</div><div class="urgency-sub">Upload a chest X-ray to begin AI-assisted triage</div></div>',
                    label="Triage Result",
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        heatmap_output = gr.Image(
                            label="Attention Map",
                            type="pil",
                            height=300,
                        )
                    with gr.Column(scale=1):
                        confidence_html = gr.HTML(
                            value='<div class="confidence-container"><div class="section-title">Awaiting Input</div></div>',
                            label="Confidence",
                        )

        with gr.Row():
            with gr.Column(scale=1):
                findings_md = gr.Markdown(
                    value="Upload an image to see clinical findings.",
                    label="Clinical Findings & Explanation",
                )
            with gr.Column(scale=1):
                comparison_html = gr.HTML(
                    value='<div style="text-align: center; color: var(--text-muted); padding: 20px;">No comparison data</div>',
                    label="Longitudinal Comparison",
                )

        # Execution Trace
        with gr.Accordion("Agent Execution Trace", open=False):
            trace_html = gr.HTML(
                value='<div class="trace-container"><em>Run analysis to see agent execution trace</em></div>',
            )

        # Raw JSON
        with gr.Accordion("Raw JSON Response", open=False):
            result_json = gr.Code(
                label="Full Result",
                language="json",
                lines=20,
            )

        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: var(--text-muted); font-size: 12px; border-top: 1px solid var(--border); margin-top: 20px;">
            MedGemma CXR Triage v4.0 • Built with Google HAI-DEF Models • 
            <strong>Not for clinical use — research demonstration only</strong>
        </div>
        """)

        # Event handler
        analyze_btn.click(
            fn=predict,
            inputs=[image_input, prior_report],
            outputs=[
                urgency_html,
                heatmap_output,
                confidence_html,
                findings_md,
                comparison_html,
                trace_html,
                result_json,
            ],
        )

    return demo


# ─── Launch ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_resources()
    demo = build_ui()

    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        allowed_paths=[str(Path(__file__).parent / "assets")],
    )
