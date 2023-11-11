The source: https://developer.mozilla.org/en-US/docs/WebAssembly/C_to_Wasm#creating_html_and_javascript

Steps:
1. Download this repo: https://github.com/emscripten-core/emsdk
2. Hopefully this download process works for you: https://emscripten.org/docs/getting_started/downloads.html
3. With the terminal now has the correct paths in place (do not delete it!): 
   ```sh
   emcc hello.c -o hello.html
   ```

I'll focus on compiling it.
You'll focus on website implementation details.

Github pages
1. I simply turned on github pages.
2. /main/index.html should exit (else rename hello.html or whatever the output was)
3. Open up the website at https://pranitsh.github.io/webgpt/