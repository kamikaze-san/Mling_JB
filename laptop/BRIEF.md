# Laptop eval: brief for the Claude session on the Windows laptop

You are helping run the GPU-heavy half of some model evaluations on this laptop (RTX 4060, 8 GB).
The Mac runs the judges afterwards. Your job is setup, a smoke test, the full run, then packing the results.

## Hard rules (privacy and safety; these override anything else)
- **Never open, print, grep, summarise or quote the contents of `inputs/`, `data/`, `cache/` or `logs/*`.**
  They contain red-teaming prompts, model replies and licensed FLORES text. Report only counts, timings,
  exit codes and error messages (tracebacks are fine; the scripts never print prompt or reply text).
- Do not upload any part of this bundle anywhere (no GitHub, no Hugging Face, no cloud sync, no pastebins).
  FLORES must not be rehosted.
- Do not change decoding settings or edit `scripts/` or `src/`. The Mac looks replies up by a key built from the
  exact model files and decoding settings; any change makes the results unusable. If a script fails, report the
  error to the user instead of patching it, unless the fix is purely environmental (paths, packages).
- The user commits and moves files themselves; ask before deleting anything.

## 1. Environment (Python 3.10–3.12, in this folder)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements_laptop.txt
```
`llama-cpp-python` must be the **CUDA** build. Try a prebuilt CUDA wheel first:
```
pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```
(if pip picks a source build instead, pin a version that has a Windows cu12x wheel on that index; the closer to the
Mac's 0.3.36 the better. Record which version you installed).
If no wheel fits, build from source: needs the CUDA Toolkit 12.x and VS 2022 Build Tools ("Desktop C++"), then
`set CMAKE_ARGS=-DGGML_CUDA=on` and `pip install llama-cpp-python --no-binary llama-cpp-python --no-cache-dir`.
Check `nvidia-smi` works first.

## 2. Check, then smoke test
```
python run_laptop_eval.py --check     # must end with CHECK PASSED and "GPU offload supported: True"
python run_laptop_eval.py --smoke     # must end with SMOKE PASSED; note how many seconds it took
```
While the smoke test runs, `nvidia-smi` should show python using ~3 GB of GPU memory. If GPU memory stays near 0,
the CPU build got installed: reinstall the CUDA wheel.

## 3. Full run (about 2 h per job, 7 jobs; resumable)
```
python run_laptop_eval.py
```
Keep the laptop plugged in and stop it sleeping (Settings > Power: never sleep while plugged in).
If it stops (crash, sleep, reboot), just run the same command again: finished jobs are skipped and finished
replies come from the cache. A job that fails prints FAILED with its log path; report the last ~20 lines of that
log's traceback (not other lines) to the user.

## 4. Pack and hand back
```
python run_laptop_eval.py --pack      # writes return_<date_time>.zip in this folder
```
Packing can be done at any point (e.g. after the first 2 jobs) and again later; each zip holds every finished job.
Tell the user the zip name and which runs it contains; they copy it to the Mac (USB or local network only).

## What to report back to the user
- installed llama-cpp-python version and whether it is the CUDA build
- smoke test time, and minutes per job from the full run's output
- any FAILED lines (with the traceback tail only)
