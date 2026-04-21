import importlib

modules = [
    "fastapi",
    "pdfplumber", 
    "sentence_transformers",
    "streamlit",
    "google.generativeai"
]

for module in modules:
    try:
        importlib.import_module(module)
        print(f"{module} -- OK")
    except ImportError:
        print(f"{module} -- FAILED")