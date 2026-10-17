// 学生 GGUF 数值校验：加载 gpt2 结构的 GGUF，对给定 token 编号序列算 sum log p(t[i+1]|t[..i]) 与末位置前 5 个 logits。
// 用法：student_check.exe model.gguf 2,3,4,5
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: student_check model.gguf id,id,...\n"); return 2; }
    std::vector<llama_token> ids;
    for (char * p = argv[2]; *p;) { ids.push_back((llama_token)strtol(p, &p, 10)); if (*p == ',') ++p; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64; cp.n_batch = 64; cp.n_threads = 1; cp.n_threads_batch = 1;
    llama_context * ctx = llama_new_context_with_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    int n = (int)ids.size();
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = ids[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    const int V = llama_vocab_n_tokens(llama_model_get_vocab(model));
    double total = 0;
    for (int i = 0; i + 1 < n; ++i) {
        const float * lg = llama_get_logits_ith(ctx, i);
        double mx = lg[0]; for (int v = 1; v < V; ++v) if (lg[v] > mx) mx = lg[v];
        double s = 0; for (int v = 0; v < V; ++v) s += std::exp(lg[v] - mx);
        total += lg[ids[i + 1]] - mx - std::log(s);
    }
    printf("LLAMA sum_logprob %.5f\n", total);
    const float * last = llama_get_logits_ith(ctx, n - 1);
    printf("LLAMA last_logits");
    for (int v = 0; v < 5; ++v) printf(" %.5f", last[v]);
    printf("\n");
    // 计时模式：第 3 个参数 = 线程数。模拟一次重排：前缀（ctx_len 个 token）解码一次，再对 8 个候选（各 2 token）逐个复用缓存。
    if (argc > 3) {
        int nthr = atoi(argv[3]);
        llama_free(ctx);
        cp.n_threads = nthr; cp.n_threads_batch = nthr; cp.n_seq_max = 9;
        ctx = llama_new_context_with_model(model, cp);
        double t_pre = 0, t_cand = 0;
        auto run = [&](int mode) {
            llama_memory_clear(llama_get_memory(ctx), true);
            auto dec = [&](const std::vector<llama_token> & t, int pos0, int seq = 0) {
                llama_batch bb = llama_batch_init((int)t.size(), 0, 1);
                for (int i = 0; i < (int)t.size(); ++i) { bb.token[i] = t[i]; bb.pos[i] = pos0 + i; bb.n_seq_id[i] = 1; bb.seq_id[i][0] = seq; bb.logits[i] = 1; }
                bb.n_tokens = (int)t.size();
                llama_decode(ctx, bb);
                llama_batch_free(bb);
            };
            std::vector<llama_token> pre(ids.begin(), ids.begin() + std::min<int>(n, 30));
            int L = (int)pre.size();
            auto ta = std::chrono::steady_clock::now();
            dec(pre, 0);
            auto tb = std::chrono::steady_clock::now();
            t_pre += std::chrono::duration<double, std::milli>(tb - ta).count();
            if (mode == 2) {
                auto * mem = llama_get_memory(ctx);
                llama_batch bb = llama_batch_init(16, 0, 1);
                int k = 0;
                for (int c = 0; c < 8; ++c) {
                    llama_memory_seq_cp(mem, 0, c + 1, 0, -1);
                    llama_token cand[2] = {ids[(c + 3) % n], ids[(c + 5) % n]};
                    for (int j = 0; j < 2; ++j, ++k) { bb.token[k] = cand[j]; bb.pos[k] = L + j; bb.n_seq_id[k] = 1; bb.seq_id[k][0] = c + 1; bb.logits[k] = 1; }
                }
                bb.n_tokens = 16;
                llama_decode(ctx, bb);
                llama_batch_free(bb);
                for (int c = 0; c < 8; ++c) llama_memory_seq_rm(mem, c + 1, -1, -1);
            } else for (int c = 0; c < 8; ++c) {
                std::vector<llama_token> cand = {ids[(c + 3) % n], ids[(c + 5) % n]};
                if (mode == 1) {
                    llama_memory_seq_rm(llama_get_memory(ctx), 0, L, -1);
                    dec(cand, L);
                } else {
                    llama_memory_clear(llama_get_memory(ctx), true);
                    std::vector<llama_token> full = pre; full.insert(full.end(), cand.begin(), cand.end());
                    dec(full, 0);
                }
            }
            t_cand += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tb).count();
        };
        for (int mode = 2; mode >= 0; --mode) {
            for (int w = 0; w < 5; ++w) run(mode);
            auto t0 = std::chrono::steady_clock::now();
            const int R = 100; t_pre = t_cand = 0;
            for (int r = 0; r < R; ++r) run(mode);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / R;
            printf("TIMING threads=%d %s %.2f ms/item (8 candidates)\n", nthr, mode == 2 ? "batched " : (mode == 1 ? "cached  " : "uncached"), ms);
            if (mode >= 1) printf("  其中 前缀 %.2f ms（可后台预解码），候选 %.2f ms（关键路径）\n", t_pre / R, t_cand / R);
        }
    }
    llama_batch_free(b); llama_free(ctx); llama_model_free(model);
    return 0;
}
