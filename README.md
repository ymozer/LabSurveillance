hf_xet caused problems for me while downloading models so I removed it.

python 13,14 tested 


GGUF won't work, WiP
if you want GGUF make sure CMake, Visual Studio Build Tools (C++), and the CUDA Toolkit installed.

llama-cpp-python install:
- Default (CPU): just run install.ps1
- Optional CUDA build (priority): set LLAMA_CPP_CUDA=1 before running install.ps1 (falls back to CPU if build fails)