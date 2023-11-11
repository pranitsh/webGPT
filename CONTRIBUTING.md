The source: https://developer.mozilla.org/en-US/docs/WebAssembly/C_to_Wasm#creating_html_and_javascript
https://www.continuation-labs.com/projects/d3wasm/#source-code
https://emscripten.org/docs/compiling/Building-Projects.html?highlight=makefile


https://blog.stackademic.com/fast-and-portable-llama2-inference-on-the-heterogeneous-edge-a62508e82359

New Install Steps:
1. Install docker at https://docs.docker.com/engine/install/
2. Run the below command to get the image:
   ```sh
   docker pull emscripten/emsdk
   ```
3. Then run the commands through the interactive mode of the image:
   ```sh
   cd gpt4all-wasm/gpt4all-backend # this is the path to folder that contains the `CMakeLists.txt`
   docker run -it -v .:/src emscripten/emsdk /bin/bash
   mkdir build
   cd build
   emcmake cmake ..
   emmake make
   ```

Old Steps:
1. Before you begin, verify you have the correct version of cmake, make, and c. If your cmake is below `3.16`, this will not work.
   1. Download it from here: https://cmake.org/download/
2. Download this repo: https://github.com/emscripten-core/emsdk
3. Hopefully this download process works for you: https://emscripten.org/docs/getting_started/downloads.html
   1. Setting up for windows involves the below statements
      ```sh
      ./emsdk.bat install latest
      ./emsdk.bat activate latest
      ./emsdk_env.bat
      ```
   2. For macOS, the setup follows: 
      ```sh
      git clone https://github.com/emscripten-core/emsdk.git
      # Enter that directory
      cd emsdk
      # Fetch the latest version of the emsdk (not needed the first time you clone)
      git pull
      # Download and install the latest SDK tools.
      ./emsdk install latest
      # Make the "latest" SDK "active" for the current user. (writes .emscripten file)
      ./emsdk activate latest
      # Activate PATH and other environment variables in the current terminal
      source ./emsdk_env.sh      
      ```
4. For the next steps, you have to find the path to the cmake in emsdk. This is located at the below:
   C:\Users\ppsha\Documents\GitHub\webgpt\emsdk\upstream\emscripten\cmake\Modules\Platform\Emscripten.cmake
   It should be located at the same place following linux-styled paths.
5. You have run this from inside of the build directory for cmake to see your `CmakeLists.txt`
   ```sh
   mkdir build
   cd build
   ```
6. You then have to install cmake
   1. For windows, this means running the below with elevated permissions.
      ```sh
      cmake -DCMAKE_TOOLCHAIN_FILE=C:\Users\ppsha\Documents\GitHub\webgpt\emsdk\upstream\emscripten\cmake\Modules\Platform\Emscripten.cmake ..
      ```
   2. For mac, this means installing
      ```brew
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      and then cmake brew install
      cmake
      ``` 

Github pages
1. I simply turned on github pages.
2. /main/index.html should exit (else rename hello.html or whatever the output was)
3. Open up the website at https://pranitsh.github.io/webgpt/
