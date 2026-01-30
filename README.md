# Lab Survailance 

python 13,14 tested 
use `ìnstall.ps1` to install all dependencies while in **venv**.
hf_xet caused problems for me while downloading models so I removed it.

Models will be downloaded at runtime after selecting and clicking 'Load Model' button

Video streaming (http or rtsp) supported with opencv. Local Webcam also supported.

Seperate photo video input section also exist. Not tested throughly.

# TODO

GGUF won't work, WiP
if you want GGUF make sure CMake, Visual Studio Build Tools (C++), and the CUDA Toolkit installed.

llama-cpp-python install:
- Default (CPU): just run install.ps1
- Optional CUDA build (priority): set LLAMA_CPP_CUDA=1 before running install.ps1 (falls back to CPU if build fails)