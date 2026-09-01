# ===== SETUP ===== #

# Create virtual environment: python3 -m venv venv
# Activate virtual environment: source venv/bin/activate
# Install libraries: pip install ollama

# To resolve import errors, ensure VS Code is using the `venv` folder:
#   1. Open Command Palette: `Cmd + Shift + P` or type ">" in the top search bar
#   2. Type: "Python: Select Interpreter"
#   3. Find venv: select the one that mentions `pyenv` or `venv`

import ollama

response = ollama.list()
#print(response)

# ===== Chat Example ===== #
chatRes = ollama.chat(
    model = "llama3.2:3b",
    messages = [
        {
            "role" : "user",
            "content" : "Why is the sky blue?"
        }
    ],
    stream = True
)

#print(chatRes)
#print(chatRes["message"]["content"])
#print(chatRes.message.content)

# Chunk iteration required if stream = True in the chat invocation
for chunk in chatRes:
    print(chunk.message.content, end = "", flush = True)
print("\n-----")

# ===== Generate Example ===== #
genRes = ollama.generate(
    model = "llama3.2:3b",
    prompt = "Why is the ocean salty?",
    stream = True
)
for chunk in genRes:
    print(chunk.response, end = "", flush = True)
print("\n-----")

#print(ollama.show("llama3.2:3b"))

# ===== Create a new model with Modelfile ===== #
systemPrompt = """
    You are a very smart research assistant for an MBA graduate working as a management consultant. 
    You answer questions using excerpts from their business school course materials: lecture slides, readings, and assignments.

    Answer in two clearly separated parts. Never blend them.

    1. FROM COURSE MATERIALS
    - Answer using only the provided context.
    - Attribute each claim to its source using whatever identifier appears in the context (course, file name, slide or page number).
    - If the context does not address the question, write "Not found in the provided course materials" and nothing else in this section.
    - If excerpts conflict or are ambiguous, name the conflict rather than smoothing it over.

    2. BEYOND COURSE MATERIALS
    - Include this section only when the course materials were incomplete or absent.
    - Answer from your general knowledge and state that it is not drawn from the user's materials.
    - Do not cite slide numbers, page numbers, file names, or courses here. You have no source to cite.
    - Omit specific figures, dates, statistics, and named studies unless you are confident they are correct. A qualitative answer is better than a fabricated number.

    General rules:
    - Never invent an author, course, date, figure, or citation.
    - Be concise and structured: direct answer first, then supporting detail in short bullets.
    - If a question is outside both the materials and your knowledge, say so rather than guessing.
    """

ollama.create(
    model = "MBA-Expert", 
    from_ = "llama3.2:3b",
    system = systemPrompt)
customRes = ollama.generate(
    model = "MBA-Expert",
    prompt = "How many oceans are there on Earth?",
    options = {
        "temperature" : 0.1
    },
    stream = True
)
for chunk in customRes:
    print(chunk.response, end = "", flush = True)
print("\n-----")