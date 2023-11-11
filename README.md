# webgpt
Running gpt4all models on web through wasm


## Directory Structure
- gpt4all-backend
    - C/C++ (ggml) model backends
- gpt4all-bindings
    - Language bindings for model backends
- gpt4all-chat
    - Chat GUI
- gpt4all-training
    - Model training/inference/eval code

## Transition Plan:
This is roughly based on what's feasible now and path of least resistance.

1. Clean up gpt4all-training.
    - Remove deprecated/unneeded files
    - Organize into separate training, inference, eval, etc. directories

2. Clean up gpt4all-chat so it roughly has same structures as above 
    - Separate into gpt4all-chat and gpt4all-backends
    - Separate model backends into separate subdirectories (e.g. llama, gptj)

