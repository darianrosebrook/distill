# evaluation/perf_mem_eval.py
# @author: @darianrosebrook

import time


# Placeholder: wire to your runtime. Measure wall clock.

def main():
    t0 = time.time()
    # call generate(...)
    time.sleep(0.05)
    tffa = time.time() - t0
    # generate 512 tokens ...
    toks = 512
    elapsed = 1.5
    toks_s = toks / elapsed
    peak_mem_gb = 8.7
    print({"ttfa": tffa, "toks_s": toks_s, "peak_mem_gb": peak_mem_gb})


if __name__ == "__main__":
    main()
