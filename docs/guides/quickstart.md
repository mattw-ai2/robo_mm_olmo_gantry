# Quick Start Guide

Get up and running with Molmo in minutes!

## 5-Minute Inference

### Using HuggingFace Transformers

The simplest way to use Molmo:

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests

# Load model and processor
model = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True
)

# Load an image
url = "https://picsum.photos/400/300"
image = Image.open(requests.get(url, stream=True).raw)

# Ask a question
inputs = processor(
    text="What do you see in this image?",
    images=image,
    return_tensors="pt"
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate response
output = model.generate(**inputs, max_new_tokens=200)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

### Different Tasks

**Image Captioning:**
```python
inputs = processor(
    text="Describe this image in detail.",
    images=image,
    return_tensors="pt"
)
```

**Visual Question Answering:**
```python
inputs = processor(
    text="How many people are in this image?",
    images=image,
    return_tensors="pt"
)
```

**Object Pointing:**
```python
inputs = processor(
    text="Point to the cat in this image.",
    images=image,
    return_tensors="pt"
)
# Response will include coordinates: "Point: x=125 y=230"
```

**Counting:**
```python
inputs = processor(
    text="Count the number of cars in this image.",
    images=image,
    return_tensors="pt"
)
```

## Using Native Molmo API

For more control and training:

```python
from olmo.models.molmo.molmo import MolmoConfig, Molmo
import torch
from PIL import Image

# Load configuration and model
config = MolmoConfig.load("path/to/config.yaml")
model = Molmo(config)
model.load_checkpoint("path/to/checkpoint")
model.eval()
model.to("cuda")

# Prepare inputs
image = Image.open("example.jpg")
preprocessor = config.build_preprocessor(for_inference=True)

# Process input
example = {
    "image": image,
    "messages": [
        {"role": "user", "content": "Describe this image."}
    ]
}
processed = preprocessor(example)

# Generate
with torch.no_grad():
    output = model.generate(
        input_ids=processed["input_ids"],
        images=processed["images"],
        max_new_tokens=200,
    )

print(output.text)
```

## Batch Processing

Process multiple images efficiently:

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

model = AutoModelForCausalLM.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "allenai/Molmo-7B-D-0924",
    trust_remote_code=True
)

# Load multiple images
images = [Image.open(f"image_{i}.jpg") for i in range(4)]
texts = ["Describe this image."] * 4

# Process batch
inputs = processor(
    text=texts,
    images=images,
    return_tensors="pt",
    padding=True
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate for all
outputs = model.generate(**inputs, max_new_tokens=200)
responses = [processor.decode(out, skip_special_tokens=True) for out in outputs]

for i, response in enumerate(responses):
    print(f"Image {i}: {response}")
```

## Video Understanding (VideoOlmo)

Process videos frame by frame:

```python
from transformers import AutoModelForCausalLM, AutoProcessor
import cv2

# Load VideoOlmo model
model = AutoModelForCausalLM.from_pretrained(
    "allenai/VideoOlmo-7B",  # Example model name
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)

# Extract frames from video
cap = cv2.VideoCapture("video.mp4")
frames = []
while len(frames) < 16:  # Sample 16 frames
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
cap.release()

# Process video
inputs = processor(
    text="What is happening in this video?",
    images=frames,
    return_tensors="pt"
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate
output = model.generate(**inputs, max_new_tokens=200)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

## Generation Options

### Greedy Decoding (Default)

Fastest, deterministic:

```python
output = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,
)
```

### Sampling

More diverse outputs:

```python
output = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,      # Lower = more focused, higher = more random
    top_p=0.9,           # Nucleus sampling
    top_k=50,            # Top-k sampling
)
```

### Beam Search

Better quality, slower:

```python
output = model.generate(
    **inputs,
    max_new_tokens=100,
    num_beams=5,
    early_stopping=True,
)
```

## Common Use Cases

### 1. Document Understanding

```python
# Load document image (high resolution)
doc_image = Image.open("document.png")

# Ask specific questions
questions = [
    "What is the title of this document?",
    "Who is the author?",
    "What is the date?",
    "Summarize the main points."
]

for question in questions:
    inputs = processor(text=question, images=doc_image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    output = model.generate(**inputs, max_new_tokens=100)
    answer = processor.decode(output[0], skip_special_tokens=True)
    print(f"Q: {question}\nA: {answer}\n")
```

### 2. Visual Grounding

```python
# Find objects in image
prompt = "Where is the red car? Provide coordinates."

inputs = processor(text=prompt, images=image, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

output = model.generate(**inputs, max_new_tokens=50)
response = processor.decode(output[0], skip_special_tokens=True)

# Parse coordinates from response
# Expected format: "Point: x=125 y=230" or "Box: x1=100 y1=200 x2=300 y2=400"
print(response)
```

### 3. Image Comparison

```python
# Compare two images
images = [Image.open("image1.jpg"), Image.open("image2.jpg")]

prompt = "Compare these two images. What are the differences?"

inputs = processor(text=prompt, images=images, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

output = model.generate(**inputs, max_new_tokens=200)
comparison = processor.decode(output[0], skip_special_tokens=True)
print(comparison)
```

## Memory Management

For large models or multiple images:

```python
import torch

# Enable mixed precision
torch.set_default_dtype(torch.float16)

# Clear cache between runs
torch.cuda.empty_cache()

# Use context manager for inference
with torch.inference_mode():
    output = model.generate(**inputs)

# Offload to CPU when not in use
model.to("cpu")
torch.cuda.empty_cache()
```

## Performance Tips

### 1. Use Appropriate Data Types

```python
# BF16 for better quality (Ampere+ GPUs)
model = model.to(torch.bfloat16)

# FP16 for speed (most GPUs)
model = model.to(torch.float16)
```

### 2. Batch When Possible

Process multiple images together instead of one at a time.

### 3. Cache Image Features

```python
# Process image once
with torch.no_grad():
    image_features = model.encode_images(images)

# Use cached features for multiple generations
for prompt in prompts:
    output = model.generate(
        prompts=prompt,
        cached_image_features=image_features
    )
```

### 4. Use vLLM for Production

For serving at scale:

```bash
pip install vllm

# Start vLLM server
vllm serve allenai/Molmo-7B-D-0924 --trust-remote-code
```

## Common Issues

### Out of Memory

```python
# Reduce image resolution
from torchvision.transforms import Resize

resize = Resize((336, 336))  # Smaller resolution
image = resize(image)

# Or use smaller model
model = AutoModelForCausalLM.from_pretrained("allenai/MolmoE-1B-0924", ...)
```

### Slow Generation

```python
# Use greedy decoding instead of sampling
do_sample=False

# Reduce max_new_tokens
max_new_tokens=50

# Use quantization
model = model.to(torch.int8)  # INT8 quantization
```

### Poor Quality Outputs

```python
# Try higher resolution
image = resize(image, (768, 768))

# Use larger model
model = "allenai/Molmo-72B-0924"

# Adjust temperature
temperature=0.7  # Balance between diversity and quality
```

## Next Steps

- **[Training Guide](training_guide.md)** - Train your own models
- **[Evaluation Guide](evaluation_guide.md)** - Evaluate model performance
- **[Deployment Guide](deployment_guide.md)** - Deploy to production
- **[API Reference](../api/models.md)** - Detailed API documentation

## Examples Repository

Check out more examples in the `examples/` directory:
- Image captioning
- Visual question answering
- Document understanding
- Video understanding
- Multi-image reasoning

