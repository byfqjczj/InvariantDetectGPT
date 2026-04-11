"""Quick test: generate one sample with one model to verify the pipeline works."""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "transformers==4.44.0",
        "accelerate==0.33.0",
        "huggingface_hub==0.24.0",
        "bitsandbytes==0.43.1",
        "sentencepiece",
        "protobuf",
    )
)

app = modal.App("test-one-sample", image=image)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(gpu="A10G", timeout=600, secrets=[hf_secret])
def test_generate():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"Loading {model_id}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    print("Model loaded successfully.")

    prompt = "Write a short news article about advances in renewable energy."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Generated text:\n{generated}")
    return generated


@app.local_entrypoint()
def main():
    result = test_generate.remote()
    print(f"\nResult: {result}")
