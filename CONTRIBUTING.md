The source: https://developer.mozilla.org/en-US/docs/WebAssembly/C_to_Wasm#creating_html_and_javascript
https://emscripten.org/docs/compiling/Building-Projects.html?highlight=makefile

Steps:
1. Download this repo: https://github.com/emscripten-core/emsdk
2. Hopefully this download process works for you: https://emscripten.org/docs/getting_started/downloads.html
   1. Setting up for windows involves the below statements
      ```sh
      ./emsdk.bat install latest
      ./emsdk.bat activate latest
      ./emsdk_env.bat
      ```
   2. Please put your macOS code...
3. For the next steps, you have to find the path to the cmake in emsdk. This is located at the below:
   C:\Users\ppsha\Documents\GitHub\webgpt\emsdk\upstream\emscripten\cmake\Modules\Platform\Emscripten.cmake
   It should be located at the same place following linux-styled paths.
4. You then have to install cmake
   1. For windows, this means running the below with elevated permissions.
      ```sh
      cmake -DCMAKE_TOOLCHAIN_FILE=C:\Users\ppsha\Documents\GitHub\webgpt\emsdk\upstream\emscripten\cmake\Modules\Platform\Emscripten.cmake ..
      ```

I'll focus on compiling it.
You'll focus on website implementation details.

Github pages
1. I simply turned on github pages.
2. /main/index.html should exit (else rename hello.html or whatever the output was)
3. Open up the website at https://pranitsh.github.io/webgpt/
