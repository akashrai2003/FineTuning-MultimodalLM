# Fine-Tuning Idefics2-8B for Document Understanding

Memory-efficient fine-tuning of an 8B parameter vision-language model using QLoRA. Achieves document question answering on a single 16GB GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akashrai2003/FineTuning-MultimodalLM/blob/main/FineTuningIdefics2.ipynb)

## Overview

This project fine-tunes [Idefics2-8B](https://huggingface.co/HuggingFaceM4/idefics2-8b) using QLoRA (Quantized Low-Rank Adaptation) for Document Visual Question Answering (DocVQA).

**Key achievements:**
- 8B parameter model running on a single 16GB GPU
- 4-bit quantization with minimal accuracy loss
- Model deployed to [Hugging Face Hub](https://huggingface.co/a-k-aAiMGoD/Idefics2-8b-multimodal)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Idefics2-8B Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│  📷 Vision Encoder (Frozen)                                      │
│       ↓                                                          │
│  🔄 Perceiver Resampler ← QLoRA Adapters (r=8, α=8)             │
│       ↓                                                          │
│  📝 Text Model (LLaMA-based) ← QLoRA Adapters                   │
│       ↓                                                          │
│  🎯 Output: Answer to Document Query                            │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Optimizations

| Technique | Memory Savings | Implementation |
|-----------|---------------|----------------|
| **4-bit Quantization (NF4)** | ~75% | BitsAndBytes `load_in_4bit` |
| **QLoRA Adapters** | ~90% trainable params | Only 0.1% of weights updated |
| **Gradient Accumulation** | 8x effective batch | `gradient_accumulation_steps=8` |
| **Frozen Vision Encoder** | ~40% compute | No backprop through SigLIP |
| **FP16 Mixed Precision** | ~50% memory | Native PyTorch AMP |

**Result:** Fine-tune an 8B model with ~12GB VRAM.

## Training Configuration

```python
# QLoRA Configuration
LoraConfig(
    r=8,                    # Low-rank dimension
    lora_alpha=8,           # Scaling factor
    lora_dropout=0.1,       # Regularization
    target_modules='.*text_model|modality_projection|perceiver_resampler.*'
)

# Training Hyperparameters
TrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch size: 16
    learning_rate=1e-4,
    warmup_steps=50,
    fp16=True
)
```

## Results

Trained on **DocVQA** dataset (1,200 document-question pairs):

| Metric | Before Fine-Tuning | After Fine-Tuning |
|--------|-------------------|-------------------|
| Document Understanding | Generic responses | Context-aware answers |
| Response Format | Verbose | Brief, accurate |
| Inference Speed | Baseline | Maintained with QLoRA |

## Quick Start

### Installation

```bash
pip install transformers accelerate datasets peft bitsandbytes
```

### Inference with Fine-Tuned Model

```python
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
from peft import PeftModel

# Load the fine-tuned model
model = Idefics2ForConditionalGeneration.from_pretrained(
    "a-k-aAiMGoD/Idefics2-8b-multimodal",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")

# Process document image + question
messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "Answer briefly."},
        {"type": "image"},
        {"type": "text", "text": "What is the total amount on this invoice?"}
    ]}
]

inputs = processor(text=processor.apply_chat_template(messages), images=[image])
output = model.generate(**inputs, max_new_tokens=64)
print(processor.decode(output[0]))
```

## Project Structure

```
FineTuning-MultimodalLM/
├── FineTuningIdefics2.ipynb    # Complete training notebook
├── README.md                    # This file
└── checkpoint-124/              # Model weights (uploaded to HF Hub)
```

## Technical Details

### Why Idefics2?

| Feature | Idefics2 | LLaVA | GPT-4V |
|---------|----------|-------|--------|
| Open Source | ✅ | ✅ | ❌ |
| Fine-tunable | ✅ | ✅ | ❌ |
| Native OCR | ✅ | ❌ | ✅ |
| Document Focus | ✅ | ❌ | ✅ |

### QLoRA vs Full Fine-Tuning

```
Full Fine-Tuning:  8B params × 2 bytes (FP16) = 16GB just for weights
                   + Gradients (16GB) + Optimizer states (32GB) = 64GB+ required

QLoRA:             8B params × 0.5 bytes (4-bit) = 4GB for weights
                   + 0.1% trainable LoRA params = ~12GB total
```

## Tech Stack

- **Model:** [Idefics2-8B](https://huggingface.co/HuggingFaceM4/idefics2-8b) (Hugging Face)
- **Fine-Tuning:** [PEFT/LoRA](https://github.com/huggingface/peft) with 4-bit quantization
- **Quantization:** [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- **Dataset:** [DocVQA](https://huggingface.co/datasets/nielsr/docvqa_1200_examples)
- **Training:** Hugging Face Transformers + Trainer API
- **Deployment:** Hugging Face Hub

## References

- [Idefics2 Paper](https://huggingface.co/blog/idefics2)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [DocVQA: A Dataset for VQA on Document Images](https://arxiv.org/abs/2007.00398)

## License

MIT
