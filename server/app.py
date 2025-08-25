from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json, orjson, os
import numpy as np
import faiss

BASE = "mistralai/Mistral-7B-v0.1"
LOAD_IN_4BIT = True

app = FastAPI(title="Dental LLM (Drafting Assistant)")

# Load base + LoRA
tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if LOAD_IN_4BIT:
    base_model = AutoModelForCausalLM.from_pretrained(BASE, load_in_4bit=True, device_map="auto")
else:
    base_model = AutoModelForCausalLM.from_pretrained(BASE)

# If LoRA not trained yet, run base model (not ideal). Otherwise load adapter.
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "..", "adapters", "dental-lora")
if os.path.isdir(ADAPTER_PATH):
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
else:
    model = base_model

# Load KB index
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
KB_INDEX = os.path.join(MODELS_DIR, "kb.index")
KB_DOCS = os.path.join(MODELS_DIR, "kb_docs.json")

kb_index = faiss.read_index(KB_INDEX) if os.path.exists(KB_INDEX) else None
kb_docs = json.load(open(KB_DOCS)) if os.path.exists(KB_DOCS) else []

try:
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    emb_model = None

class Inp(BaseModel):
    payload: dict

def retrieve(payload_text, k=3):
    if kb_index is None or not kb_docs or emb_model is None:
        return []
    q = emb_model.encode([payload_text], normalize_embeddings=True)
    D, I = kb_index.search(np.array(q, dtype="float32"), k)
    return [kb_docs[i] for i in I[0]]

@app.post("/infer")
def infer(inp: Inp):
    p = inp.payload

    # Basic safety gate
    if not p.get("confirmed_by_dentist", False):
        return {"error": "Not confirmed by dentist", "status": "blocked"}

    snippets = retrieve(json.dumps(p))
    kb_txt = "\n\n".join([f"[{s['id']}]\n{s['text']}" for s in snippets])

    sys = ("You are a dental clinical drafting assistant. "
           "Always return VALID JSON that matches the schema. "
           "Use retrieved text as ground truth. "
           "If info is missing or uncertain, set safety.needs_human_review=true. "
           "All prescriptions must have status='draft'.")
    prompt = f"SYSTEM:\n{sys}\n\nRETRIEVED:\n{kb_txt}\n\nINPUT_JSON:\n{json.dumps(p)}\n\nOUTPUT_JSON:\n"

    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(**input_ids, max_new_tokens=700, do_sample=False)
    raw = tokenizer.decode(gen[0], skip_special_tokens=True)
    out = raw.split("OUTPUT_JSON:")[-1].strip()

    try:
        resp = orjson.loads(out)
    except Exception:
        resp = {"safety": {"needs_human_review": True, "reasons": ["invalid_json"]}, "raw": out}

    # Enforce draft status for any Rx
    for r in resp.get("prescription_draft", []):
        r["status"] = "draft"

    return resp

@app.get("/health")
def health():
    return {"ok": True}
