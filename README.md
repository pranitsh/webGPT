# webgpt
Running gpt4all models on web through wasm

For the switch to typescript, run the below:
1. Install npm from `https://nodejs.org/`
2. Run the below to install typescript:
   ```sh
   npm install -g typescript
   ```
3. Run the below to create javascript from typescript in prep for the website.
   ```sh
   npm install
   tsc llama2.ts -t es2022
   ```