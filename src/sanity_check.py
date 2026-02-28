"""
Sanity check script for the MNIST CNN project.

Verifies that the Python environment, all dependencies, the model, and a
short training loop work correctly.  Designed to catch the most common
Windows issue — "ImportError: DLL load failed while importing _C" — and
print an actionable fix checklist when it occurs.

Usage (from the project root):
    python src/sanity_check.py
"""

import importlib
import platform
import struct
import sys
import os

# ---------------------------------------------------------------------------
# 1. Environment info
# ---------------------------------------------------------------------------
print("=" * 60)
print("  ENVIRONMENT")
print("=" * 60)
print(f"  Python version  : {sys.version}")
print(f"  Platform        : {platform.platform()}")
print(f"  Architecture    : {platform.machine()} ({struct.calcsize('P') * 8}-bit)")
print(f"  sys.executable  : {sys.executable}")
print(f"  sys.path[0]     : {sys.path[0]}")
print()

# ---------------------------------------------------------------------------
# 2. Dependency imports
# ---------------------------------------------------------------------------
print("=" * 60)
print("  DEPENDENCY CHECK")
print("=" * 60)

# Packages to probe — torch and torchvision are handled specially below.
_SIMPLE_DEPS = [
    ("numpy",      "numpy"),
    ("matplotlib",  "matplotlib"),
    ("sklearn",     "scikit-learn"),
]

all_ok = True

for module_name, pip_name in _SIMPLE_DEPS:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  [OK]   {pip_name:20s}  version {version}")
    except ImportError as exc:
        print(f"  [FAIL] {pip_name:20s}  --> {exc}")
        all_ok = False

# --- torch (with Windows-specific DLL guidance) ---------------------------
torch_ok = False
try:
    import torch
    print(f"  [OK]   {'torch':20s}  version {torch.__version__}")
    torch_ok = True
except ImportError as exc:
    print(f"  [FAIL] {'torch':20s}  --> {exc}")
    all_ok = False
    print()
    print("-" * 60)
    print("  PyTorch failed to import.  Common fixes (Windows):")
    print("-" * 60)
    print()
    print("  1. Install the Visual C++ Redistributable (x64):")
    print("     https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print()
    print("  2. Make sure you are using a 64-bit Python (not 32-bit).")
    print(f"     Current interpreter is {struct.calcsize('P') * 8}-bit.")
    print()
    print("  3. Reinstall PyTorch CPU-only wheels inside your venv:")
    print("     pip uninstall torch torchvision -y")
    print("     pip install torch torchvision --index-url "
          "https://download.pytorch.org/whl/cpu")
    print()
    print("  4. Use Python 3.11 — it has the best PyTorch compatibility.")
    print(f"     Current version: {platform.python_version()}")
    print()
    print("  5. If you are in a venv, confirm it is activated:")
    print("     .venv\\Scripts\\Activate.ps1   (PowerShell)")
    print("-" * 60)

# --- torchvision -----------------------------------------------------------
torchvision_ok = False
if torch_ok:
    try:
        import torchvision
        print(f"  [OK]   {'torchvision':20s}  version {torchvision.__version__}")
        torchvision_ok = True
    except ImportError as exc:
        print(f"  [FAIL] {'torchvision':20s}  --> {exc}")
        all_ok = False

print()

if not torch_ok:
    print("Stopping early — torch must import before model tests can run.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. CUDA availability
# ---------------------------------------------------------------------------
print("=" * 60)
print("  CUDA STATUS")
print("=" * 60)
cuda_available = torch.cuda.is_available()
print(f"  CUDA available  : {cuda_available}")
if cuda_available:
    print(f"  CUDA version    : {torch.version.cuda}")
    print(f"  GPU device      : {torch.cuda.get_device_name(0)}")
else:
    print("  (Training will use CPU — that is fine for MNIST.)")
print()

# ---------------------------------------------------------------------------
# 4. Model import and instantiation
# ---------------------------------------------------------------------------
print("=" * 60)
print("  MODEL CHECK")
print("=" * 60)

# Allow imports from the src/ directory regardless of where the script is run.
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from model import MNISTNet

    model = MNISTNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  [OK]   MNISTNet instantiated  ({total_params:,} parameters)")
except Exception as exc:
    print(f"  [FAIL] Could not load MNISTNet --> {exc}")
    sys.exit(1)

print()

# ---------------------------------------------------------------------------
# 5. Dataset download
# ---------------------------------------------------------------------------
print("=" * 60)
print("  DATASET CHECK")
print("=" * 60)

from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

try:
    train_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )
    print(f"  [OK]   MNIST train  : {len(train_dataset):,} samples")
    print(f"  [OK]   MNIST test   : {len(test_dataset):,} samples")
except Exception as exc:
    print(f"  [FAIL] MNIST download failed --> {exc}")
    sys.exit(1)

print()

# ---------------------------------------------------------------------------
# 6. Forward pass
# ---------------------------------------------------------------------------
print("=" * 60)
print("  FORWARD PASS CHECK")
print("=" * 60)

from torch.utils.data import DataLoader

loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
images, labels = next(iter(loader))

model.eval()
with torch.no_grad():
    logits = model(images)

print(f"  Input  shape : {list(images.shape)}")
print(f"  Output shape : {list(logits.shape)}")
preds = logits.argmax(dim=1)
print(f"  Sample preds : {preds[:10].tolist()}")
print(f"  Sample labels: {labels[:10].tolist()}")
print(f"  [OK]   Forward pass succeeded")
print()

# ---------------------------------------------------------------------------
# 7. Mini training loop (20 batches)
# ---------------------------------------------------------------------------
print("=" * 60)
print("  MINI TRAINING LOOP  (20 batches, CPU)")
print("=" * 60)

import torch.nn as nn

model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

total_loss = 0.0
correct = 0
total = 0
num_batches = 0
max_batches = 20

for images, labels in loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    _, predicted = torch.max(outputs, 1)
    correct += (predicted == labels).sum().item()
    total += labels.size(0)
    num_batches += 1

    if num_batches >= max_batches:
        break

avg_loss = total_loss / num_batches
accuracy = 100.0 * correct / total

print(f"  Batches trained : {num_batches}")
print(f"  Samples seen    : {total}")
print(f"  Average loss    : {avg_loss:.4f}")
print(f"  Accuracy        : {accuracy:.2f}%")
print(f"  [OK]   Training loop succeeded")
print()

# ---------------------------------------------------------------------------
# 8. Final verdict
# ---------------------------------------------------------------------------
print("=" * 60)
if all_ok:
    print("  ALL CHECKS PASSED — your environment is ready!")
else:
    print("  SOME CHECKS FAILED — review the [FAIL] lines above.")
print("=" * 60)

sys.exit(0 if all_ok else 1)
