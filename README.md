# webgpt
Running gpt4all models on web through wasm


## Directory Structure
- gpt4all-wasm
  - `STEPS.md`: includes steps for compiling and documents changes
  - gpt4all-backend
        - C/C++ (ggml) model backends
  - gpt4all-training
      - Model training/inference/eval code

Plans:
1. Compile with pthread on emsdk
2. Compile other llama.cpp