import ollama
import os

model = "llama3.2:3b"

# Paths to input and output files
input_file = "./Llama/data/grocery_list.txt"
output_file = "./Llama/data/categorized_grocery_list.txt"

# Check if the input file exists
if not os.path.exists(input_file):
    print(f"Input file '{input_file}' not found.")
    exit(1)

# Read the uncategorized grocery items from the input file
with open(input_file, "r") as file:
    items = file.read().strip()

# Prepare the prompt for the model
system_prompt = f"""
    You are an assistant that categorizes and sorts grocery items.

    Here is a list of grocery items:

    {items}

    Please:

    1. Categorize these items into appropriate categories such as Produce, Dairy, Meat, Bakery, Beverages, etc.
    2. Sort the items alphabetially within each category.
    3. Present the categorized list in a clear and organized manner, using bullet points and numbering.

    """

# Send the prompt and get the response
try:
    response = ollama.generate(model = model, prompt = system_prompt)
    generated_text = response.get("response", "")
    print("===== Categorized List =====\n")
    print(generated_text)

    # Write the categorized list to the output file
    with open(output_file, "w") as file:
        file.write(generated_text.strip())

    print(f"Categorized grocery list has been saved to '{output_file}'")
except Exception as e:
    print("An error occurred:", str(e))