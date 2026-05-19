# PyInstaller spec — bundles the engine into a self-contained folder
# `engine/dist/engine/` containing engine.exe and every dependency.
#
# Usage (from repo root):
#   .\engine\.venv\Scripts\activate.ps1
#   pyinstaller engine/build_engine.spec --noconfirm --clean
#
# Output: engine/dist/engine/engine.exe (with siblings).
# electron-builder picks this up via extraResources in ui/electron-builder.yml.

# ruff: noqa
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_all, collect_data_files, collect_dynamic_libs,
    collect_submodules, copy_metadata,
)

ENGINE_DIR = Path(SPECPATH).resolve()
REPO_ROOT = ENGINE_DIR.parent

# Hidden imports — modules PyInstaller's static scanner misses because they're
# loaded via importlib at runtime or behind feature flags.
HIDDEN_IMPORTS = [
    'MetaTrader5',
    'duckdb',
    'pandas', 'pandas_ta_classic',
    'numpy', 'scipy', 'scipy.io', 'scipy.io.wavfile',
    'sklearn', 'sklearn.utils._typedefs',
    'torch', 'torch._C',
    'stable_baselines3', 'gymnasium',
    'smartmoneyconcepts',
    'anthropic', 'httpx', 'aiohttp',
    'websockets', 'pydantic', 'loguru',
    'cryptography', 'cryptography.fernet',
    'supabase', 'postgrest', 'gotrue', 'realtime', 'storage3',
    'finnhub', 'pytz', 'schedule', 'psutil', 'dotenv',
]
HIDDEN_IMPORTS += collect_submodules('torch')
HIDDEN_IMPORTS += collect_submodules('sklearn')
HIDDEN_IMPORTS += collect_submodules('stable_baselines3')

# collect_all bundles package .py files + data + submodules. Critical for
# packages PyInstaller's static scanner skips. Each tuple is
# (datas, binaries, hiddenimports).
for _pkg in ('anthropic', 'smartmoneyconcepts', 'httpx', 'httpcore',
             'lightgbm', 'pandas_ta_classic', 'supabase', 'postgrest',
             'gotrue', 'realtime', 'storage3', 'finnhub'):
    try:
        _d, _b, _h = collect_all(_pkg)
    except Exception:
        continue
    # We append at the bottom-level lists; this is a forward declaration.
    globals().setdefault('_EXTRA_DATAS', []).extend(_d)
    globals().setdefault('_EXTRA_BINARIES', []).extend(_b)
    HIDDEN_IMPORTS += _h

# Binary DLLs that ship inside torch / mt5 / scipy wheels.
BINARIES = list(globals().get('_EXTRA_BINARIES', []))
BINARIES += collect_dynamic_libs('torch')
BINARIES += collect_dynamic_libs('MetaTrader5')
BINARIES += collect_dynamic_libs('scipy')
BINARIES += collect_dynamic_libs('numpy')

# Pure-data files (stub headers, ONNX schemas, etc.).
DATAS = list(globals().get('_EXTRA_DATAS', []))
DATAS += collect_data_files('torch')
DATAS += collect_data_files('MetaTrader5')
DATAS += copy_metadata('anthropic')
DATAS += copy_metadata('supabase')

# Project-owned assets: schemas + sounds + checkpoints.
DATAS += [
    (str(ENGINE_DIR / 'data' / 'schemas.sql'),    'engine/data'),
    (str(ENGINE_DIR / 'sounds'),                  'engine/sounds'),
    (str(ENGINE_DIR / 'models' / 'checkpoints'),  'engine/models/checkpoints'),
]

a = Analysis(
    [str(ENGINE_DIR / 'engine.py')],
    pathex=[str(REPO_ROOT)],
    binaries=BINARIES,
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Skip GUI / Jupyter cruft we never use to keep the bundle smaller.
        'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'IPython', 'jupyter', 'notebook', 'matplotlib',
    ],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='engine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                # don't UPX — false-positive AV hits
    console=True,             # keep stdout visible; UI redirects this
    disable_windowed_traceback=False,
    icon=str(REPO_ROOT / 'ui' / 'build' / 'icon.ico') if (REPO_ROOT / 'ui' / 'build' / 'icon.ico').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='engine',
)
