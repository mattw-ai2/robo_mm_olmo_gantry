import base64
import json
import os

import requests
MOLMO_TOKEN = os.getenv("MOLMO_MODAL_TOKEN")
# MODEL_ENDPOINT = "https://ai2-reviz--uber-model-v4-synthetic.modal.run/completion_stream"
MODEL_ENDPOINT= "https://ai2-reviz--robomolmo-scenemem-objpoint-roomcount-27jun2025.modal.run/completion_stream"

def get_model_prediction(image, event):

    instruction = event.get("instruction", "")

    if image.startswith("data:image"):
        image = image.split(",")[1]

    # Define payload in the same format as the working request
    payload = {
        "input_text": [instruction],
        "input_image": [image]
    }

    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {MOLMO_TOKEN}"}
        response = requests.post(
            MODEL_ENDPOINT,
            headers=headers,
            data=json.dumps(payload),
            stream=True
        )

        print(f"Molmo API Response Status Code: {response.status_code}")

        # Check if response is valid
        if response.status_code != 200:
            print("[ERROR] Molmo API failed:", response.text)
            return None

        # Process streaming response
        response_text = ""
        for chunk in response.iter_lines():
            if chunk:
                response_text += json.loads(chunk)["result"]["output"]["text"]

        return response_text

    except Exception as e:
        print(f"[ERROR] Failed to query Molmo API: {str(e)}")
        return None

def get_random_image():
    """Download a random image from Unsplash for testing"""
    # Using a placeholder image URL - replace with actual image URL if needed
    image_url = "https://picsum.photos/800/600"
    response = requests.get(image_url)
    if response.status_code == 200:
        return response.content
    return None

if __name__ == "__main__":
    image = get_random_image()
    image_base64 = base64.b64encode(image).decode('utf-8')
    event = {"instruction": "describe this image"}

    result = get_model_prediction(image_base64, event)
    print("Model Prediction:", result)