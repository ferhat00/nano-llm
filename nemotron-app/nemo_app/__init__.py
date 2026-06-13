"""nemo_app — supporting package for the Nemotron Nano 4B Streamlit app.

All real logic lives here (server lifecycle, prompt building, RAG, web search,
prompt assembly, cached resources); `app.py` is a thin Streamlit entry point that
wires these together. Mirrors the repo convention of keeping logic in modules.
"""
