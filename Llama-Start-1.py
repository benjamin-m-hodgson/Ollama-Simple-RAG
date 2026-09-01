# ===== SETUP ===== #

# Set local python version: pyenv local 3.13.15
# Create virtual environment: python3 -m venv venv
# Activate virtual environment: source venv/bin/activate
# If needed, delete virtual environment: rm -rf venv

# Install libraries: pip install requests
# Run in Terminal: python3 Llama-Start-1.py

# To resolve import errors, ensure VS Code is using the `venv` folder:
#   1. Open Command Palette: `Cmd + Shift + P` or type ">" in the top search bar
#   2. Type: "Python: Select Interpreter"
#   3. Find venv: select the one that mentions `venv`

import requests
import json

url = "http://localhost:11434/api/generate"

data = {
    "model" : "llama3.2:3b",
    "prompt" : "Tell me a short, funny story."
}

response = requests.post(url, json = data, stream = True)

# Check the response status
if response.status_code == 200:
    print("Generated Text:", end = " ", flush = True)
    # Iterate over the streaming response
    for line in response.iter_lines():
        if line:
            # Decode the line and parse the JSON
            decoded_line = line.decode("utf-8")
            result = json.loads(decoded_line)
            # Get the text from the response
            generated_text = result.get("response", "")
            print(generated_text, end = "", flush = True)
    print()
else:
    print("Error:", response.status_code, response.text)