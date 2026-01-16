"""
Script for hosting a vllm mm_olmo endpoint via modal.

TEST:
modal serve src/{your_model}.py

DEPLOY:
modal deploy src/{your_model}.py
"""
import os
import json
import time
import uuid
from typing import Any, Dict, List, Optional

import modal
import modal.gpu

N_GPU = 1

MODEL_NAME = "REPLACEME_USER_MODEL_NAME"
SEQ_LEN = 4048
MAX_NEW_TOKENS = 1024
ORIGINAL_CHECKPOINT_PATH = "REPLACEME_ORIGINAL_CHECKPOINT_PATH"

VOLUME_DIR = "/robo-molmo"
MODELS_DIR = f"{VOLUME_DIR}/REPLACEME_REMOTE_MODEL_PATH_ON_VOLUME"

APP_NAME = "REPLACEME_SANITIZED_APP_NAME"
APP_LABEL = "REPLACEME_SANITIZED_APP_LABEL"
TIMEOUT = 600

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = '1'
print("Setting VLLM_ALLOW_LONG_MAX_MODEL_LEN = 1 to override the max_model_len in VLLM: ",
      os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"])

# ## Define a container image and mount the volume containing the model checkpoint
# The `HF_ACCESS_TOKEN` environment variable must be set

try:
    volume = modal.Volume.from_name("robo-molmo", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Upload checkpoint first with modal run demo_scripts.modal_upload_checkpoint --dst_dir [DIST_DIR]")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "transformers",
        "torchmetrics",
        "imageio",
        "imageio[pyav]",
        "decord",
        "pyav",
        "moviepy",
        "datasets",
        "scipy",
        "omegaconf",
        "rich",
        "boto3",
        "google-cloud-storage",
        "cached_path",
        "ninja",
        "packaging",
    ).pip_install("vllm==0.8.4")
).add_local_python_source("olmo", "scripts")

app = modal.App(APP_NAME)


# ## The model class
#
# The inference function is best represented with Modal's [class syntax](/docs/guide/lifecycle-functions) and the `@enter` decorator.
# This enables us to load the model into memory just once every time a container starts up, and keep it cached
# on the GPU for each subsequent invocation of the function.


@app.cls(
    gpu=f"H100:{N_GPU}",
    timeout=TIMEOUT,
    scaledown_window=60 * 10,
    min_containers=0,
    image=vllm_image,
    volumes={VOLUME_DIR: volume},
    max_containers=10,
)
@modal.concurrent(max_inputs=5)
class Model:
    @modal.enter()
    def start_api(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        from vllm import ModelRegistry
        from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
        from scripts.hf_molmo.vllm_molmo import MolmoForCausalLM
        ModelRegistry.register_model("MolmoForCausalLM", MolmoForCausalLM)
        _MULTIMODAL_MODELS["MolmoForCausalLM"] = ("molmo", "MolmoForCausalLM")

        print("🥶 cold starting inference")
        start = time.monotonic_ns()

        engine_args = AsyncEngineArgs(
            model=MODELS_DIR,
            tokenizer=MODELS_DIR,
            gpu_memory_utilization=0.80,
            enforce_eager=False,  # capture the graph for faster inference, but slower cold starts
            disable_log_stats=True,  # disable logging so we can stream tokens
            disable_log_requests=True,
            max_num_batched_tokens=SEQ_LEN,
            max_model_len=SEQ_LEN,
            trust_remote_code=True,
        )

        # this can take some time!
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @staticmethod
    async def logged_timed_request(url):
        import os
        import aiohttp
        t0 = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download image. Status code: {response.status}")
                data = await response.read()

                duration = time.time() - t0
                content_length_str = response.headers.get('Content-Length', "0")
                content_length = 0
                if content_length_str.isnumeric():
                    content_length = int(content_length_str)

                speed = 0
                if duration > 0:
                    speed = int(content_length / duration)

                print(json.dumps({
                    "event": "download_image",
                    "duration_seconds": duration,
                    "status_code": response.status,
                    "url": str(response.url),
                    "content_length": content_length,
                    "bytes_per_second": speed,
                    "cloud": os.getenv("MODAL_CLOUD_PROVIDER", "unknown"),
                    "region": os.getenv("MODAL_REGION", "unknown"),
                }))
                return data

    @staticmethod
    async def image_to_numpy(image_str):
        import base64
        from io import BytesIO
        import numpy as np
        from PIL import Image, ImageFile, ImageOps
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if isinstance(image_str, str) and (
                image_str.strip().startswith("http://") or image_str.strip().startswith("https://")):
            data = await Model.logged_timed_request(image_str)
            image = Image.open(BytesIO(data)).convert("RGB")
            image = ImageOps.exif_transpose(image)
        else:
            image = ImageOps.exif_transpose(
                Image.open(BytesIO(base64.b64decode(image_str.encode("utf-8")))).convert("RGB"))

        return image

    @staticmethod
    async def process_image(image_str):
        import base64
        from io import BytesIO
        import numpy as np
        from PIL import Image, ImageFile, ImageOps
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = ImageOps.exif_transpose(Image.open(BytesIO(base64.b64decode(image_str))).convert("RGB"))
        return image

    @modal.method()
    async def completion_stream(self, input_image: str, prompt: str, opts: dict):
        from vllm import SamplingParams
        from vllm.utils import random_uuid

        kwargs = {
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0,
            "seed": 123
        }
        kwargs.update(opts)
        sampling_params = SamplingParams(**kwargs)
        request_id = random_uuid()

        try:
            # image_array = await Model.process_image(input_image)
            image_array = await Model.image_to_numpy(input_image)
        except Exception as e:
            text = "An error occurred during image processing. Please try again."
            err_msg = str(e)
            status = "error"
            infer_s = 0.0
            yield text, request_id, infer_s, status, err_msg
            return

        input = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_array}
        }

        result_generator = self.engine.generate(
            input,
            sampling_params,
            request_id,
        )

        index, num_tokens = 0, 0
        start = time.monotonic_ns()

        try:
            async for output in result_generator:
                if (
                        output.outputs[0].text
                        and "\ufffd" == output.outputs[0].text[-1]
                ):
                    continue
                text_delta = output.outputs[0].text[index:]
                if index == 0:
                    text_delta = text_delta.lstrip()  # remove leading whitespace
                index = len(output.outputs[0].text)
                num_tokens = len(output.outputs[0].token_ids)

                elapsed_s = (time.monotonic_ns() - start) / 1e9
                status = "full" if num_tokens >= MAX_NEW_TOKENS else "okay"
                yield text_delta, request_id, elapsed_s, status, ""
        except AssertionError:
            modal.experimental.stop_fetching_inputs()
        except Exception as e:
            elapsed_s = (time.monotonic_ns() - start) / 1e9
            err_msg = str(e)
            if "max_model_len" in err_msg:
                text = "The conversation exceeded Molmo's max length. Please start another conversation."
            else:
                text = "An error occurred during inference. Please try again."
            yield text, request_id, elapsed_s, "error", err_msg

    @modal.exit()
    def stop_engine(self):
        if N_GPU > 1:
            import ray

            ray.shutdown()


# ## Coupling a frontend web application
#
# We can stream inference from a FastAPI backend, also deployed on Modal.

from modal import asgi_app

api_image = (
    modal.Image.debian_slim(python_version="3.11")
)


@app.function(
    image=api_image,
    min_containers=0,
    timeout=TIMEOUT,
)
@modal.concurrent(max_inputs=100)
@asgi_app(label=APP_LABEL)
def model_web():
    import fastapi
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse

    web_app = fastapi.FastAPI()

    # Add CORSMiddleware to the application
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    @web_app.post("/completion_stream")
    async def completion_stream(inp: dict, user_agent: Optional[str] = fastapi.Header(None)):
        async def generate():
            try:
                input_image = inp["input_image"][0]
                prompt = inp["input_text"][0]
                opts = inp.get("opts", {})
            except KeyError as e:
                text = "An error happened during parsing the input. Please try again."
                err_msg = str(e)
                status = "error"
                infer_s = 0.0
                request_id = str(uuid.uuid4().hex)
                output = dict(text=text, status=status, error_message=err_msg)
                result = dict(
                    output=output,
                    inferenceTime=f"{infer_s}s",
                    originalCheckpointPath=ORIGINAL_CHECKPOINT_PATH
                )
                response = dict(requestId=request_id, result=result)
                yield f"{json.dumps(response, ensure_ascii=False)}\n".encode("utf-8")
                return

            model = Model()
            async for text, request_id, infer_s, status, err_msg in model.completion_stream.remote_gen.aio(
                    input_image, prompt, opts
            ):
                output = dict(text=text, status=status, error_message=err_msg)
                result = dict(
                    output=output,
                    inferenceTime=f"{infer_s}s",
                    originalCheckpointPath=ORIGINAL_CHECKPOINT_PATH
                )
                response = dict(requestId=request_id, result=result)

                yield f"{json.dumps(response, ensure_ascii=False)}\n".encode("utf-8")

        return StreamingResponse(generate(), media_type="text/event-stream")

    return web_app


# ## Coupling a deployed function endpoint
#
# See also https://modal.com/docs/guide/trigger-deployed-functions for
# for client-side use

@app.function(
    image=api_image,
    min_containers=0,
    timeout=TIMEOUT,
)
@modal.concurrent(max_inputs=20)
async def vllm_api(inp: dict):
    try:
        input_image = inp["image"]
        prompt = inp["prompt"]
        opts = inp.get("opts", {})
    except KeyError as e:
        text = "An error happend during parsing the input. Please try again."
        err_msg = str(e)
        status = "error"
        infer_s = 0.0
        request_id = str(uuid.uuid4().hex)
        output = dict(text=text, status=status, error_message=err_msg)
        result = dict(
            output=output,
            inferenceTime=f"{infer_s}s",
            originalCheckpointPath=ORIGINAL_CHECKPOINT_PATH
        )
        response = dict(requestId=request_id, result=result)
        yield response
        return

    model = Model()
    async for text, request_id, infer_s, status, err_msg in model.completion_stream.remote_gen.aio(
            input_image, prompt, opts
    ):
        output = dict(text=text, status=status, error_message=err_msg)
        result = dict(
            output=output,
            inferenceTime=f"{infer_s}s",
            originalCheckpointPath=ORIGINAL_CHECKPOINT_PATH
        )
        response = dict(requestId=request_id, result=result)
        yield response


# This local entry point allows you to test inference without deploying the Modal app.
# It should be used by running the following command from the repository root.
# > modal run demo_scripts/modal_api_vllm_server.py::app.main
@app.local_entrypoint()
async def main():
    print("=== local entrypoint")
    print("=== lazily initializing model")
    model = Model()
    print("=== completion_stream start", flush=True)
    result = model.completion_stream.remote_gen(
        "https://www.datocms-assets.com/64837/1721697383-wildlands-trees.jpg",
        "What does the image depict?",
        {},
    )
    for res in result:
        print(res, flush=True)
    print("=== completion_stream complete", flush=True)