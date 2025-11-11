# app.py
"""
Prompt Guard AI (Apple Silicon Local Version)
🛡️ Guard: Llama-3-1B-Instruct
💬 Generator: Llama-3-3B-Instruct
Runs natively on M1/M2/M3 without CUDA.
"""

import os, traceback
import gradio as gr
from transformers import pipeline, logging as hf_logging
from guardprompt import produce_improved_prompt, guarded_llm_call, _fake_llm_response, JSONAuditLogger
from semantic_guard import semantic_guard_check

hf_logging.set_verbosity_error()

# Models
GUARD_MODEL_ID = os.environ.get("GUARD_MODEL_ID", "meta-llama/Meta-Llama-3-1B-Instruct")
RESPONSE_MODEL_ID = os.environ.get("RESPONSE_MODEL_ID", "meta-llama/Meta-Llama-3-3B-Instruct")

try:
    gen_pipe = pipeline("text-generation", model=RESPONSE_MODEL_ID, device_map="auto")
except Exception:
    gen_pipe = None

def llama_generate(prompt: str) -> str:
    if gen_pipe is None:
        return _fake_llm_response(prompt)
    try:
        out = gen_pipe(prompt, max_new_tokens=300, temperature=0.7)
        if isinstance(out, list) and len(out) > 0:
            return out[0].get("generated_text") or str(out[0])
        return str(out)
    except Exception:
        traceback.print_exc()
        return _fake_llm_response(prompt)

logger = JSONAuditLogger("local_audit.log")

def handle_prompt(user_prompt: str):
    try:
        if not user_prompt.strip():
            return "❌ Empty prompt.", "", ""
        # Step 1: Sanitize
        report = produce_improved_prompt(user_prompt)
        improved = report["improved_prompt"]
        # Step 2: Guard LLM
        verdict = semantic_guard_check(improved)
        label = verdict.get("label", "OTHER")
        raw = verdict.get("raw", "")
        if label != "SAFE":
            return f"⚠️ Guard flagged prompt as **{label}**\nReason: {raw}", improved, report["remediation"]
        # Step 3: Generate
        result = guarded_llm_call(improved, llama_generate, audit_logger=logger)
        if result.get("status") in ("blocked", "review", "leak_detected"):
            return f"⛔ {result['status']} | {result.get('reason','')}", improved, report["remediation"]
        return f"✅ SAFE | Risk Score: {result.get('risk_score','N/A')}", improved, result.get("response","")
    except Exception:
        traceback.print_exc()
        return "❌ Internal error.", "", ""

# UI 
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("<h1 style='text-align:center;'>🛡️ Prompt Guard AI</h1><p style='text-align:center;'>Llama-3 Guard + Llama-3 Generator (Apple Silicon Edition)</p>")
    with gr.Row():
        with gr.Column(scale=1):
            prompt_box = gr.Textbox(label="Enter Prompt", lines=8, placeholder="Type your prompt here…")
            submit_btn = gr.Button("Analyze & Generate", variant="primary")
        with gr.Column(scale=1):
            status_box = gr.Markdown("Awaiting input…")
            improved_box = gr.Textbox(label="Improved Prompt (after redaction)", lines=6)
            output_box = gr.Textbox(label="LLM Response", lines=10)
    submit_btn.click(fn=handle_prompt, inputs=prompt_box, outputs=[status_box, improved_box, output_box], concurrency_limit=2)
    gr.Markdown("<hr><center><small>Audit log → local_audit.log</small></center>")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
