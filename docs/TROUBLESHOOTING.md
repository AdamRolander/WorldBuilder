# Troubleshooting

Common failures during [setup](SETUP.md) and the first runs. If yours is not here, open an [issue](https://github.com/AdamRolander/WorldBuilder/issues) with the files listed in [Contributing](../CONTRIBUTING.md#reporting-bugs).

## `CUDA out of memory` Mid-Pipeline

Usually a model failed to park on CPU,
or another process holds the GPU (`nvidia-smi`). Restart the webapp/CLI
process; if it's reproducible, check the `Moving model: cuda → cpu` lines in
the log to see which stage didn't release.

## `VLM server unreachable at http://127.0.0.1:8765`

The server isn't
running, or is still loading the model (~1 min). `curl
localhost:8765/health`; `{"ok": true}` means it's ready.

## `GEMINI_API_KEY not found`

`.env` is missing, or you started the process
from a directory other than the project root (`load_dotenv()` resolves `.env`
relative to the working directory), or you meant `--detector qwen`.

## `KeyError: 'CONDA_PREFIX'` From `notebook/inference.py`

Fixed in WorldBuilder (the path is derived from the interpreter), but if you run the
upstream notebook directly you need an activated conda shell.

## "Textured room failed … using box room"

The layout stage fell back;
`3d_models/layout.json` and the console say why (usually no floor visible
or no valid intrinsics). The scene is still complete.

## SAM 3 Checkpoint 404 or Gated Repo Error

Access wasn't approved yet, or
`hf auth login` hasn't run in this shell.

## `RuntimeError: Not compiled with GPU support` From pytorch3d

It was built
on a machine without a visible GPU. Rebuild `pip install -e ./pytorch3d` on the
GPU box.

## `pipeline.yaml` Not Found

The SAM 3D download landed in the wrong place;
the `mv` step in setup step 4 is easy to miss.

## NumPy 2.x Errors After Installing Something New

Something upgraded numpy.
`pip install 'numpy==1.26.4'` and reinstall whatever pulled it.

---

