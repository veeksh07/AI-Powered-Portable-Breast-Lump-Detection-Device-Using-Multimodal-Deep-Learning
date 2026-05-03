import sys
print(f"Python: {sys.version}\n")

tests = {
    'serial':     'pyserial',
    'numpy':      'numpy',
    'pandas':     'pandas',
    'cv2':        'opencv-python',
    'PIL':        'Pillow',
    'scipy':      'scipy',
    'skimage':    'scikit-image',
    'matplotlib': 'matplotlib',
    'seaborn':    'seaborn',
    'sklearn':    'scikit-learn',
    'tqdm':       'tqdm',
}

# Test basic libraries first
all_ok = True
for module, name in tests.items():
    try:
        __import__(module)
        print(f"✅ {name}")
    except Exception as e:
        print(f"❌ {name} — {e}")
        all_ok = False

# Test tensorflow separately (most likely to fail)
print("\nTesting TensorFlow...")
try:
    import tensorflow as tf
    print(f"✅ tensorflow {tf.__version__}")
except Exception as e:
    print(f"❌ tensorflow — {e}")
    all_ok = False

if all_ok:
    print("\n🎉 All libraries ready! Let's build!")
else:
    print("\n⚠️  Fix issues above then run again")