// Llama2 transformer model inference in one TypeScript file.
// by Oleksandr Nikitin, 2023 (MIT licensed).
// Based on the Andrej Karpathy's llama2.c: https://github.com/karpathy/llama2.c
//
// Use bun or t348 to run. see params at the end of the file or in the README.
const f32bytes = 4;
const i32bytes = 4;
class BufferReader {
    view;
    position;
    constructor(arrayBuffer) {
        this.view = new DataView(arrayBuffer);
        this.position = 0;
        console.log("New buffer array");
    }
    getInt32LE() {
        let value = this.view.getInt32(this.position, true);
        this.position += 4; // size of Int32 in bytes
        return value;
    }
    getFloat32LE() {
        let value = this.view.getFloat32(this.position, true);
        this.position += 4; // size of Float32 in bytes
        return value;
    }
    getBytesInto(bytes) {
        bytes.set(new Uint8Array(this.view.buffer, this.position, bytes.length));
        this.position += bytes.length;
        return bytes;
    }
}
// class FileHandleReader {
//   handle: number;
//   position: number;
//   constructor(handle: number, offset: number) {
//     this.handle = handle;
//     this.position = offset;
//   }
//   getF32Array(...dims: number[]): Float32Array {
//     let totalFloats = dims.reduce((a, b) => a * b);
// //    console.log({offset, dims, totalBytes, bb:this.view.buffer.length})
//     let bytes = Buffer.alloc(totalFloats * f32bytes);
//     fs.readSync(this.handle, bytes, 0, bytes.length, this.position);
//     let ret = new Float32Array(bytes.buffer, bytes.byteOffset, totalFloats);
//     this.position += totalFloats * f32bytes;
//     return ret;
//   }
//   getF32Arrays(dim0: number, ...dims: number[]): Float32Array[] {
//     let array = new Array(dim0);
//     for (let i = 0; i < dim0; ++i) {
//       array[i] = this.getF32Array(...dims);
//     }
//     return array;
//   }
// }
class ArrayBufferReader {
    buffer;
    position;
    view;
    constructor(arrayBuffer, offset = 0) {
        this.buffer = arrayBuffer;
        this.position = offset;
        this.view = new DataView(arrayBuffer);
        console.log("New array buffer array");
    }
    getF32Array(...dims) {
        const totalFloats = dims.reduce((a, b) => a * b);
        const bytes = new Uint8Array(this.buffer, this.position, totalFloats * f32bytes);
        const ret = new Float32Array(bytes.buffer, bytes.byteOffset, totalFloats);
        this.position += totalFloats * f32bytes;
        return ret;
    }
    getF32Arrays(dim0, ...dims) {
        let array = new Array(dim0);
        for (let i = 0; i < dim0; ++i) {
            array[i] = this.getF32Array(...dims);
        }
        return array;
    }
    getInt32LE() {
        let value = this.view.getInt32(this.position, true);
        this.position += 4; // size of Int32 in bytes
        return value;
    }
    getFloat32LE() {
        let value = this.view.getFloat32(this.position, true);
        this.position += 4; // size of Float32 in bytes
        return value;
    }
    getBytesInto(bytes) {
        let byteLength = bytes.length;
        let newBytes = new Uint8Array(this.buffer, this.position, byteLength);
        bytes.set(newBytes);
        this.position += byteLength;
        return bytes;
    }
}
function readConfig(buffer) {
    let c = {};
    c.dim = buffer.getInt32LE();
    c.hidden_dim = buffer.getInt32LE();
    c.n_layers = buffer.getInt32LE();
    c.n_heads = buffer.getInt32LE();
    c.n_kv_heads = buffer.getInt32LE();
    let vocab_size = buffer.getInt32LE();
    c.vocab_size = Math.abs(vocab_size);
    c.seq_len = buffer.getInt32LE();
    c.shared_weights = vocab_size > 0;
    c.head_size = c.dim / c.n_heads;
    return c;
}
// function readWeights(config: Config, buffer: FileHandleReader, shared_weights:boolean):TransformerWeights {
//   let w={} as TransformerWeights;
//   w.token_embedding_table = buffer.getF32Array(config.vocab_size, config.dim);
//   w.rms_att_weight = buffer.getF32Arrays(config.n_layers, config.dim);
//   w.wq = buffer.getF32Arrays(config.n_layers, config.dim, config.dim);
//   w.wk = buffer.getF32Arrays(config.n_layers, config.dim, config.dim);
//   w.wv = buffer.getF32Arrays(config.n_layers, config.dim, config.dim);
//   w.wo = buffer.getF32Arrays(config.n_layers, config.dim, config.dim);
//   w.rms_ffn_weight = buffer.getF32Arrays(config.n_layers, config.dim); // jagged pointer arithmetic lol
//   w.w1 = buffer.getF32Arrays(config.n_layers, config.hidden_dim, config.dim);
//   w.w2 = buffer.getF32Arrays(config.n_layers, config.dim, config.hidden_dim);
//   w.w3 = buffer.getF32Arrays(config.n_layers, config.hidden_dim, config.dim);
//   w.rms_final_weight = buffer.getF32Array(config.dim);
//   w.freq_cis_real = buffer.getF32Array(config.seq_len, config.head_size / 2);
//   w.freq_cis_imag = buffer.getF32Array(config.seq_len, config.head_size / 2);
//   w.wcls = shared_weights ? w.token_embedding_table : buffer.getF32Array(config.vocab_size, config.dim);
//   return w;
// }
function readWeights(config, buffer, shared_weights) {
    let w = {};
    w.token_embedding_table = buffer.getF32Array(config.vocab_size, config.dim);
    w.rms_att_weight = buffer.getF32Arrays(config.n_layers, config.dim);
    w.wq = buffer.getF32Arrays(config.n_layers, config.dim, config.dim);
    w.wk = buffer.getF32Arrays(config.n_layers, config.dim, config.dim);
    w.wv = buffer.getF32Arrays(config.n_layers, config.dim, config.dim);
    w.wo = buffer.getF32Arrays(config.n_layers, config.dim, config.dim);
    w.rms_ffn_weight = buffer.getF32Arrays(config.n_layers, config.dim); // jagged pointer arithmetic lol
    w.w1 = buffer.getF32Arrays(config.n_layers, config.hidden_dim, config.dim);
    w.w2 = buffer.getF32Arrays(config.n_layers, config.dim, config.hidden_dim);
    w.w3 = buffer.getF32Arrays(config.n_layers, config.hidden_dim, config.dim);
    w.rms_final_weight = buffer.getF32Array(config.dim);
    w.freq_cis_real = buffer.getF32Array(config.seq_len, config.head_size / 2);
    w.freq_cis_imag = buffer.getF32Array(config.seq_len, config.head_size / 2);
    w.wcls = shared_weights ? w.token_embedding_table : buffer.getF32Array(config.vocab_size, config.dim);
    return w;
}
function newRunState(config) {
    let s = {};
    s.indices = new Array(config.vocab_size);
    s.x = new Float32Array(config.dim);
    s.xb = new Float32Array(config.dim);
    s.xb2 = new Float32Array(config.dim);
    s.hb = new Float32Array(config.hidden_dim);
    s.hb2 = new Float32Array(config.hidden_dim);
    s.q = new Float32Array(config.dim);
    s.k = new Float32Array(config.dim);
    s.v = new Float32Array(config.dim);
    s.att = new Float32Array(config.n_heads * config.seq_len);
    s.logits = new Float32Array(config.vocab_size);
    s.key_cache = new Float32Array(config.n_layers * config.seq_len * config.dim);
    s.value_cache = new Float32Array(config.n_layers * config.seq_len * config.dim);
    return s;
}
// ----------------------------------------------------------------------------
// neural net blocks
function accum(a, b, size) {
    for (let i = 0; i < size; i++) {
        a[i] += b[i];
    }
}
function rmsnorm(o, x, weight, size) {
    let ss = 0;
    for (let j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss = 1.0 / Math.sqrt(1e-5 + ss);
    for (let j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
    // debugger;
}
function softmax(x, xPtr, size) {
    let max_val = x[xPtr];
    for (let i = 1; i < size; i++) {
        if (x[i + xPtr] > max_val) {
            max_val = x[i + xPtr];
        }
    }
    for (let i = 0; i < size; i++) {
        x[i + xPtr] = Math.exp(x[i + xPtr] - max_val);
    }
    let sum = 0;
    for (let i = 0; i < size; i++) {
        sum += x[i + xPtr];
    }
    for (let i = 0; i < size; i++) {
        x[i + xPtr] /= sum; //Accumulator[0]; // ah forget it, it's numerically stable enough
    }
}
function matmul(xout, x, w, n, d) {
    // W (d, n) @ x (n,) -> xout (d,)
    for (let i = 0; i < d; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++) {
            sum += w[i * n + j] * x[j];
        }
        xout[i] = sum; //sumAccumulator[0];
    }
}
function transformer(token, pos, p, s, w) {
    const x = s.x;
    const dim = p.dim;
    const hidden_dim = p.hidden_dim;
    const head_size = dim / p.n_heads;
    x.set(w.token_embedding_table.subarray(token * dim, token * dim + dim));
    //debugger;
    // forward all the layers
    for (let l = 0; l < p.n_layers; l++) {
        rmsnorm(s.xb, x, w.rms_att_weight[l], dim);
        // qkv matmuls for this position
        matmul(s.q, s.xb, w.wq[l], dim, dim);
        matmul(s.k, s.xb, w.wk[l], dim, dim);
        matmul(s.v, s.xb, w.wv[l], dim, dim);
        // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
        for (let i = 0; i < dim; i += 2) {
            const q0 = s.q[i];
            const q1 = s.q[i + 1];
            const k0 = s.k[i];
            const k1 = s.k[i + 1];
            const fcr = w.freq_cis_real[pos * head_size / 2 + (i % head_size) / 2];
            const fci = w.freq_cis_imag[pos * head_size / 2 + (i % head_size) / 2];
            s.q[i] = q0 * fcr - q1 * fci;
            s.q[i + 1] = q0 * fci + q1 * fcr;
            s.k[i] = k0 * fcr - k1 * fci;
            s.k[i + 1] = k0 * fci + k1 * fcr;
        }
        // save key,value at this time step (pos) to our kv cache
        const loff = l * p.seq_len * dim; // kv cache layer offset for convenience
        s.key_cache.set(s.k, loff + pos * dim);
        s.value_cache.set(s.v, loff + pos * dim);
        //debugger;
        // multihead attention. iterate over all heads
        for (let h = 0; h < p.n_heads; h++) {
            let q = s.q.subarray(h * head_size, h * head_size + head_size);
            let attPtr = h * p.seq_len;
            // iterate over all timesteps, including the current one
            for (let t = 0; t <= pos; t++) {
                const cached_k = s.key_cache.subarray(loff + t * dim + h * head_size);
                let scope = 0.0;
                for (let i = 0; i < head_size; i++) {
                    scope += q[i] * cached_k[i];
                }
                s.att[attPtr + t] = scope / Math.sqrt(head_size);
            }
            softmax(s.att, attPtr, pos + 1);
            s.xb.fill(0, h * head_size, h * head_size + head_size);
            // weighted sum of the values, store back into xb
            for (let t = 0; t <= pos; t++) {
                const att_t = s.att[attPtr + t];
                for (let i = 0; i < head_size; i++) {
                    s.xb[h * head_size + i] += att_t * s.value_cache[loff + t * dim + h * head_size + i];
                }
            }
        }
        // final matmul to get the output of the attention
        matmul(s.xb2, s.xb, w.wo[l], dim, dim);
        // residual connection back into x
        accum(x, s.xb2, dim);
        // ffn rmsnorm
        rmsnorm(s.xb, x, w.rms_ffn_weight[l], dim);
        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s.hb, s.xb, w.w1[l], dim, hidden_dim);
        matmul(s.hb2, s.xb, w.w3[l], dim, hidden_dim);
        // F.silu; silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        for (let i = 0; i < hidden_dim; i++) {
            s.hb[i] = s.hb[i] * (1.0 / (1.0 + Math.exp(-s.hb[i])));
        }
        // elementwise multiply with w3(x)
        for (let i = 0; i < hidden_dim; i++) {
            s.hb[i] = s.hb[i] * s.hb2[i];
        }
        // final matmul to get the output of the ffn
        matmul(s.xb, s.hb, w.w2[l], hidden_dim, dim);
        // residual connection
        accum(x, s.xb, dim);
    }
    // final rmsnorm
    rmsnorm(x, x, w.rms_final_weight, dim);
    // classifier into logits
    matmul(s.logits, x, w.wcls, p.dim, p.vocab_size);
}
function bpe_encode(text, vocab, vocab_scores, vocab_size, tokens) {
    // first encode every individual byte in the input string
    let n_tokens = 0; // the number of tokens
    for (let i = 0; i < text.length; ++i) {
        let id = vocab.indexOf(text.charAt(i));
        if (id == -1) {
            throw new Error("Error: character not found in vocab: " + text.charAt(i));
        }
        tokens[n_tokens++] = id;
    }
    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (true) {
        let best_score = -1e10;
        let best_id = -1;
        let best_idx = -1;
        for (let i = 0; i < n_tokens - 1; ++i) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            let str_buffer = vocab[tokens[i]] + vocab[tokens[i + 1]];
            let id = vocab.indexOf(str_buffer);
            if (id != -1 && vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) {
            break;
        } // we couldn't find any more pairs to merge, so we're done
        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (let i = best_idx + 1; i < n_tokens - 1; i++) {
            tokens[i] = tokens[i + 1];
        }
        n_tokens--; // token length decreased
    }
    return n_tokens;
}
// ----------------------------------------------------------------------------
// utilities: time / rng
let rng_seed = 0n;
function random_u32() {
    rng_seed ^= (rng_seed >> 12n);
    rng_seed ^= (rng_seed << 25n) & 0xffffffffffffffffn;
    rng_seed ^= (rng_seed >> 27n);
    return Number(((rng_seed * 0x2545f4914f6cdd1dn) >> 32n) & 0xffffffffn);
}
const floatCaster = new Float32Array(1);
function random_f32() {
    floatCaster[0] = (random_u32() / 256) / 16777216.0;
    return floatCaster[0]; // force f32
}
// ----------------------------------------------------------------------------
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
function argmax(arr) {
    return arr.reduce((maxIdx, val, idx, array) => (val > array[maxIdx] ? idx : maxIdx), 0);
}
function sample(logits, vocabSize) {
    const sum = logits.reduce((acc, val) => acc + val, 0);
    const randValue = random_f32() * sum;
    let cumProb = 0;
    for (let i = 0; i < vocabSize; i++) {
        cumProb += logits[i];
        if (randValue < cumProb)
            return i;
    }
    return 0;
}
function sample_topp(logits, topp, probindex) {
    for (let i = 0; i < probindex.length; i++) {
        probindex[i] = { index: i, prob: logits[i] };
    }
    probindex.sort((a, b) => b.prob - a.prob);
    let cumProb = 0;
    let lastIdx = 0;
    for (let i = 0; i < probindex.length; i++) {
        cumProb += probindex[i].prob;
        if (cumProb > topp) {
            lastIdx = i;
            break;
        }
    }
    const randValue = random_f32() * cumProb;
    cumProb = 0;
    for (let i = 0; i < lastIdx; i++) {
        cumProb += probindex[i].prob;
        if (randValue < cumProb)
            return probindex[i].index;
    }
    return 0;
}
// // New function to handle file operations
// function readFileOperations(checkpoint) {
//   let fileHandle = fs.openSync(checkpoint, "r");
//   let configSize = 7 * i32bytes;
//   // read in the config header
//   let configBuffer = Buffer.alloc(configSize);
//   fs.readSync(fileHandle, configBuffer, 0, configSize, 0);
//   let config = readConfig(new BufferReader(configBuffer));
//   let weights = readWeights(config, new FileHandleReader(fileHandle, configSize), config.shared_weights);
//   fs.closeSync(fileHandle);
//   // Read in the tokenizer.bin file
//   let vocab = new Array<string>(config.vocab_size);
//   let vocab_scores = new Array<number>(config.vocab_size);
//   let tokBuffer = new BufferReader(fs.readFileSync("tokenizer.bin"));
//   let ignored_max_token_length = tokBuffer.getInt32LE();
//   for (let i = 0; i < config.vocab_size; i++) {
//     vocab_scores[i] = tokBuffer.getFloat32LE();
//     vocab[i] = new TextDecoder().decode(tokBuffer.getBytesInto(new Uint8Array(tokBuffer.getInt32LE())));
//   }
//   return { config, weights, vocab, vocab_scores };
// }
async function storeCheckpointFile(file) {
    return new Promise((resolve, reject) => {
        const openRequest = indexedDB.open('MyDatabase', 1);
        console.log("Opened indexdb");
        openRequest.onupgradeneeded = function () {
            const db = openRequest.result;
            if (!db.objectStoreNames.contains('files')) {
                db.createObjectStore('files', { keyPath: 'id' });
                console.log("performed create object store");
            }
        };
        openRequest.onerror = function () {
            reject(openRequest.error);
        };
        openRequest.onsuccess = function () {
            const db = openRequest.result;
            const transaction = db.transaction('files', 'readwrite');
            const filesStore = transaction.objectStore('files');
            const request = filesStore.put({ id: 'checkpointFile', file: file });
            console.log("successfully saved file to db");
            request.onsuccess = function () {
                resolve(request.result);
            };
            request.onerror = function () {
                reject(request.error);
            };
        };
    });
}
async function retrieveCheckpointFile() {
    return new Promise((resolve, reject) => {
        const openRequest = indexedDB.open('MyDatabase', 1);
        console.log("opened db");
        openRequest.onerror = function () {
            reject(openRequest.error);
        };
        openRequest.onsuccess = function () {
            const db = openRequest.result;
            const transaction = db.transaction('files', 'readonly');
            const filesStore = transaction.objectStore('files');
            const request = filesStore.get('checkpointFile');
            request.onsuccess = function () {
                console.log("succesfully retrieved file from dv");
                resolve(request.result.file);
            };
            request.onerror = function () {
                reject(request.error);
            };
        };
    });
}
// fileState.ts
let checkpointFile = null;
export async function setCheckpointFile(file) {
    console.log("started set checkpoint file");
    checkpointFile = file;
    await storeCheckpointFile(file); // Assuming storeCheckpointFile is your IndexedDB storage function
}
export async function getCheckpointFile() {
    if (checkpointFile) {
        return checkpointFile; // Return from memory if available
    }
    else {
        // Try to retrieve from IndexedDB
        try {
            return await retrieveCheckpointFile();
        }
        catch (error) {
            console.error("Error retrieving file from IndexedDB:", error);
            return null;
        }
    }
}
async function handleFiles() {
    const fileInputElement = document.getElementById('fileInput');
    const files = fileInputElement.files;
    console.log("handling file");
    if (files && files.length > 0) {
        const checkpointFile = files[0]; // Only the checkpoint file is needed
        setCheckpointFile(files[0]);
        console.log("Uploaded file without error");
    }
    else {
        console.error("No checkpoint file selected");
    }
}
async function processPrompt() {
    const promptInputElement = document.getElementById('promptInput');
    const prompt = promptInputElement.value;
    if (!prompt) {
        console.error("Please enter a prompt.");
        return;
    }
    // Use the checkpoint file selected by handleFiles
    const checkpointFile = getCheckpointFile();
    if (checkpointFile) {
        try {
            const { config, weights, vocab, vocab_scores } = await readFileOperations(checkpointFile);
            main(config, weights, vocab, vocab_scores, { prompt: prompt });
        }
        catch (e) {
            console.error(e);
        }
    }
    else {
        console.error("Checkpoint file not selected.");
    }
}
async function readFileOperations(checkpointFile) {
    const configSize = 7 * i32bytes;
    // Read the config header
    let configBuffer = await checkpointFile.slice(0, configSize).arrayBuffer();
    let config = readConfig(new ArrayBufferReader(configBuffer));
    // Read weights
    let weightsBuffer = await checkpointFile.slice(configSize).arrayBuffer();
    let weights = readWeights(config, new ArrayBufferReader(weightsBuffer), config.shared_weights);
    // Fetch the tokenizer file from the root directory
    const response = await fetch('/tokenizer.bin');
    if (!response.ok) {
        throw new Error('Failed to load tokenizer.bin');
    }
    let tokenizerBuffer = await response.arrayBuffer();
    let tokBuffer = new ArrayBufferReader(tokenizerBuffer);
    let vocab = new Array(config.vocab_size);
    let vocab_scores = new Array(config.vocab_size);
    let ignored_max_token_length = tokBuffer.getInt32LE();
    for (let i = 0; i < config.vocab_size; i++) {
        vocab_scores[i] = tokBuffer.getFloat32LE();
        vocab[i] = new TextDecoder().decode(tokBuffer.getBytesInto(new Uint8Array(tokBuffer.getInt32LE())));
    }
    return { config, weights, vocab, vocab_scores };
}
// async function readFileOperations(combinedFile) {
//   const configSize = 7 * i32bytes;
//   const checkpointDataSize = 100; // Replace with the actual size of the checkpoint data
//   // Read the config header from the beginning of the combined file
//   let configBuffer = await combinedFile.slice(0, configSize).arrayBuffer();
//   let config = readConfig(new ArrayBufferReader(configBuffer));
//   // Read weights, starting right after the config
//   let weightsBuffer = await combinedFile.slice(configSize, checkpointDataSize).arrayBuffer();
//   let weights = readWeights(config, new ArrayBufferReader(weightsBuffer), config.shared_weights);
//   // The start position of the tokenizer data in the combined file
//   let tokenizerStart = checkpointDataSize;
//   let tokenizerBuffer = await combinedFile.slice(tokenizerStart).arrayBuffer();
//   let tokBuffer = new ArrayBufferReader(tokenizerBuffer);
//   let vocab = new Array<string>(config.vocab_size);
//   let vocab_scores = new Array<number>(config.vocab_size);
//   let ignored_max_token_length = tokBuffer.getInt32LE();
//   for (let i = 0; i < config.vocab_size; i++) {
//     vocab_scores[i] = tokBuffer.getFloat32LE();
//     vocab[i] = new TextDecoder().decode(tokBuffer.getBytesInto(new Uint8Array(tokBuffer.getInt32LE())));
//   }
//   return { config, weights, vocab, vocab_scores };
// }
// ----------------------------------------------------------------------------
// int main
function main(config, weights, vocab, vocab_scores, prompt) {
    // defaults
    let temperature = 1.0; // 0.0 = greedy deterministic. 1.0 = original. Don't set higher
    let topp = 1.0; // Top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    let rng_seed = BigInt(Date.now()); // Seed RNG with time by default, if not provided
    let steps = 256; // Max number of steps to run for, 0: use seq_len
    // for (let i = 0; i < args.length; i += 2) {
    //   if (i + 1 >= args.length) { return error_usage(); } // must have arg after flag
    //   let [arg, val] = [args[i], args[i + 1]];
    //   if (arg.charAt(0) != '-') { return error_usage(); } // must start with dash
    //   if (arg.length != 2) { return error_usage(); } // must be -x (one dash, one letter)
    //   // read in the args
    //   switch (args[i][1]) {
    //     case 't': temperature = parseFloat(val);break;
    //     case 'p': topp = parseFloat(val);break;
    //     case 's': rng_seed = BigInt(parseInt(val));break;
    //     case 'n': steps = parseInt(val);break;
    //     case 'i': prompt = val;break;
    //     default: return error_usage();
    //   }
    // }
    // if (!checkpoint) {return error_usage();}
    // const { config, weights, vocab, vocab_scores } = readFileOperations(checkpoint);
    if (rng_seed == 0n) {
        rng_seed = BigInt(Date.now());
    }
    // create and init the application RunState
    let state = newRunState(config);
    //debugger;
    // process the prompt, if any
    let prompt_tokens = new Int32Array(config.seq_len);
    let num_prompt_tokens = 0;
    if (prompt != null) {
        num_prompt_tokens = bpe_encode(prompt, vocab, vocab_scores, config.vocab_size, prompt_tokens);
    }
    // start the main loop
    let start = 0; // used to time our code, only initialized after first iteration
    let next; // will store the next token in the sequence
    let token = 1; // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    let pos = 0; // position in the sequence
    while (pos < steps) {
        // forward the transformer to get logits for the next token
        transformer(token, pos, config, state, weights);
        // advance the state machine
        if (pos < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos];
        }
        else {
            // sample the next token
            if (temperature == 0.0) {
                // greedy argmax sampling: take the token with the highest probability
                next = argmax(state.logits);
            }
            else {
                // apply the temperature to the logits
                for (let q = 0; q < config.vocab_size; q++) {
                    state.logits[q] /= temperature;
                }
                // apply softmax to the logits to get the probabilities for next token
                softmax(state.logits, 0, config.vocab_size);
                // we sample from this distribution to get the next token
                if (topp <= 0 || topp >= 1) {
                    // simply sample from the predicted probability distribution
                    next = sample(state.logits, config.vocab_size);
                }
                else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    next = sample_topp(state.logits, topp, state.indices);
                }
            }
        }
        pos++;
        // data-dependent terminating condition: the BOS (1) token delimits sequences
        if (next == 1) {
            break;
        }
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR#89)
        let token_str = (token == 1 && vocab[next].charAt(0) == ' ') ? vocab[next].substring(1) : vocab[next];
        console.log(token_str); // note: assumes utf8 terminal
        token = next;
        // init the timer here because the first iteration can be slower
        if (start == 0)
            start = Date.now();
    }
    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    console.log("\n\nachieved tok/s: %f\n", (pos - 1) / (Date.now() - start) * 1000.0);
}
// function error_usage(): never {
//   console.error("Usage: ... llama2.ts <checkpoint> [options]");
//   console.error("Example: llama2.ts model.bin -n 256 -i \"Once upon a time\"");
//   console.error("Options:");
//   console.error("  -t <float>  temperature, default 1.0");
//   console.error("  -p <float>  p value in top-p (nucleus) sampling. default 0.9, 0 = off");
//   console.error("  -s <int>    random seed, default time(NULL)");
//   console.error("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len");
//   console.error("  -i <string> input prompt");
//   process.exit(1);
// }
