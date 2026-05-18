# LLM — Fine-Tuning Efisien & Serving

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

Folder ini berisi tiga jalur hands-on terkait **Large Language Model (LLM)** dan adaptasi parameter efisien:

| Modul | Fokus | Panduan |
|-------|--------|---------|
| [accelerate_example/](accelerate_example/) | Pelatihan terdistribusi / mixed precision dengan Hugging Face **Accelerate** | [README](accelerate_example/README.md) |
| [peft_example/sequence_cls/bert/](peft_example/sequence_cls/bert/) | **PEFT** (LoRA, P-Tuning, Prefix, Prompt) pada klasifikasi sentimen BERT | [README](peft_example/sequence_cls/bert/README.md) |
| [chatglm_lora_finetuning/](chatglm_lora_finetuning/) | Fine-tune **ChatGLM** dengan LoRA / P-Tuning + API streaming Flask | [README](chatglm_lora_finetuning/README.md) |

**Sumber asli:** [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project) — folder `LLM`.

---

## Prasyarat umum

- Python 3.9+, GPU NVIDIA (disarankan untuk ChatGLM & PEFT skala penuh)
- Familiar dengan `transformers`, `torch`, dan konsep **prompt / instruction tuning**

## Alur belajar yang disarankan

1. **Accelerate** — pahami abstraksi pelatihan multi-GPU sebelum fine-tune LLM besar.  
2. **PEFT + BERT** — latih adaptasi ringan pada tugas klasifikasi (lebih ringan, cocok CPU/GPU kecil).  
3. **ChatGLM LoRA** — generasi teks + deploy streaming (butuh VRAM lebih besar).
