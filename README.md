# dental-llm-starter

A minimal starter to build a **dental clinical drafting assistant**. It turns structured inputs
(chief complaint, exam, tooth number, etc.) into **draft** problems (SNODENT/ICD-10 placeholders),
recommended procedures (CDT placeholder), and a **draft** prescription for a licensed dentist
to review/sign. This is **not** a substitute for clinical judgement.

> ⚠️ Safety: Always keep prescriptions in `status: "draft"` and require dentist approval.
> Do not use with real patient PHI; use only de-identified or synthetic data during development.

## 0) Prereqs
- Python 3.10+
- (Optional) 1 GPU with 12–24 GB VRAM for faster LoRA fine-tuning. CPU works for learning/prototyping.
- macOS/Linux shell

## 1) Create & activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

## 2) Install dependencies
```bash
pip install -r requirements.txt
```
If you run into GPU/driver issues, set `LOAD_IN_4BIT=False` later and use CPU.

## 3) Add your knowledge base (RAG)
- Put PDF/HTML/TXT guidelines and formulary docs into `knowledge_base/`.
- Then build the vector index:
```bash
python scripts/build_index.py
```

## 4) Create instruction-tuning data
- Edit `instructions/train.jsonl` and add more examples (100–1000+). Each line is a JSON object:
  - `prompt`: input JSON (your schema)
  - `response`: expected output JSON
- Keep codes as placeholders unless you have valid licensing for SNODENT/CDT and RxNorm access.

## 5) Train a small LoRA adapter
```bash
python scripts/train_lora.py
```
Outputs LoRA adapter in `adapters/dental-lora/`.

## 6) Run the inference API (with RAG + safety)
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## 7) Test with a sample request
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## 8) Next steps
- Add more realistic examples covering: caries, pulpitis, abscess, pericoronitis, periodontitis, dry socket, pediatric, pregnancy, anticoagulants.
- With real data: de-identify first (remove all PHI). Keep prescriptions **draft** only.
- Add JSON Schema validation to the server for stricter checks.
- If commercial use: obtain licenses for SNODENT/CDT; consult legal/regulatory experts.

