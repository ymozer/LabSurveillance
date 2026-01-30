
if ($env:VIRTUAL_ENV -or $env:CONDA_PREFIX) {
	$activeEnv = if ($env:VIRTUAL_ENV) { $env:VIRTUAL_ENV } else { $env:CONDA_PREFIX }
	Write-Host "Environment already active ($activeEnv). Skipping venv activation."
} else {
	& "$PSScriptRoot\.venv\Scripts\Activate.ps1"
}

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install git+https://github.com/huggingface/transformers.git

# llama-cpp-python install priority:
# - Default: CPU install (fast, uses wheels if available)
# - Optional: set LLAMA_CPP_CUDA=1 to try a CUDA build first, then fall back to CPU if it fails
$llamaCppVersion = "0.3.16"
$wantCuda = $env:LLAMA_CPP_CUDA

if ($wantCuda -and $wantCuda -ne "0") {
	Write-Host "LLAMA_CPP_CUDA enabled; attempting CUDA build for llama-cpp-python==$llamaCppVersion"
	$oldCmakeArgs = $env:CMAKE_ARGS
	$oldForceCmake = $env:FORCE_CMAKE
	try {
		$env:CMAKE_ARGS = "-DGGML_CUDA=on"
		$env:FORCE_CMAKE = "1"
		python -m pip install --upgrade --force-reinstall --no-cache-dir --no-binary llama-cpp-python llama-cpp-python==$llamaCppVersion
	} catch {
		Write-Warning "CUDA build failed; falling back to CPU install for llama-cpp-python==$llamaCppVersion"
		python -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python==$llamaCppVersion
	} finally {
		$env:CMAKE_ARGS = $oldCmakeArgs
		$env:FORCE_CMAKE = $oldForceCmake
	}
} else {
	python -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python==$llamaCppVersion
}

pip3 install -r requirements.txt

python -m pip uninstall -y hf-xet 2>$null

