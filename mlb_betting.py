"""
mlb_betting.py — Streamlit Cloud entry point
============================================
Streamlit Cloud is configured to run this file.
All app logic lives in mlb_app.py; this just delegates to it.

To switch the entry point permanently:
  Streamlit Cloud → your app → Settings → Main file path → mlb_app.py
"""
import importlib.util
import os
import sys

_dir  = os.path.dirname(os.path.abspath(__file__))
_path = os.path.join(_dir, "mlb_app.py")

_spec = importlib.util.spec_from_file_location("mlb_app", _path)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["mlb_app"] = _mod
_spec.loader.exec_module(_mod)   # runs page config, CSS, and all defs
_mod.main()                      # runs the Streamlit UI
