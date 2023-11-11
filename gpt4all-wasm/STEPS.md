WebAssembly does not support IPO (interprocedural optimization)
1. #!PS1 are used in the `gpt4all-setup\gpt4all-backend\CMakeLists.txt` to remove checks for this, a number of related warnings.
2. To allow the use of `pthread` with emsdk, I used #!PS2 to enable these compiler options for emsdk.
3. Since emsdk has its own form of threading, comment with #!PS3 `find_package(Threads REQUIRED)`
4. #!PS4 demonstrates the use of web assemblies' pthread
