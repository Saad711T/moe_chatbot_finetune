# moe_chatbot_finetune
fine-tuning a small MoE model into a chatbot with RTX-4050

This notebook fine-tunes a small **Mixture-of-Experts (MoE)** language model into a helpful chatbot, using techniques that fit inside **8–16 GB of VRAM** (RTX 4050 mobile/desktop).

**Model choices (swap in Section 2):**
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` — 14.3B total / 2.7B active params, 60 routing + 4 shared experts.
- `microsoft/Phi-mini-MoE-instruct` — 7.6B total / 2.4B active params, distilled from Phi-3.5-MoE (recommended default — smaller footprint).

**Techniques used:**
1. **4-bit NF4 quantization** via `bitsandbytes` (with double quantization + bf16 compute).
2. **LoRA** adapters targeting attention *and* MoE expert / gate modules via `peft`.
3. **Gradient checkpointing** to trade compute for memory.
4. **Paged 8-bit AdamW** optimizer to avoid OOM on the optimizer state.
5. **MoE auxiliary load-balancing loss** (`router_aux_loss_coef`, `output_router_logits`) so experts don't collapse.
6. **Chat template** formatting + assistant-only loss masking.
7. **Gradio** `ChatInterface` with streaming for the live demo.



---
This project is for educational purposes. No proprietary or private datasets were used.

### Credits
[0xSaad](https://x.com/0xdonzdev)
[Perplexity.ai](https://www.perplexity.ai/)
