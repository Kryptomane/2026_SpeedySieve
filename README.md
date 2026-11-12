# Furious Speed

**Algorithm optimisation on processor architecture – counting primes really fast.**

Furious Speed counts the primes in `[0, x]` for any `x < 2^64` with a segmented sieve of Eratosthenes. It has two engines: a multi-threaded CPU engine in modern C++ (AVX2) and a CUDA engine for NVIDIA GPUs. Both were written from scratch. The project grew over about 15 years, from a Pascal trial-division program (8 minutes up to 10^9) to the version described here.

This repository contains the documentation only, not the source code.

## Benchmarks

### CPU

Intel Core i5-13600KF (6 P-cores + 8 E-cores, 20 threads, P 5.4 GHz / E 4.2 GHz), Windows, MSVC 2026, Qt 6.12. primesieve was measured on the same machine.

| x | π(x) | primesieve | **Furious Speed** | Difference |
|---|---:|---:|---:|---:|
| 10^12 | 37,607,912,018 | 17 s | **14.6 s** | −14 % |
| 10^13 | 346,065,536,839 | 203 s | **176 s** | −13 % |
| 2·10^13 | | – | 390 s | |

### GPU

| x | π(x) | LingSieve V2.3 <br>GTX 1080 | LingSieve V3.2 <br>RTX 5060 Ti | **Furious Speed** <br>**RTX 2080 Ti** | Furious Speed <br>RTX 5060 Ti (*estimated*) |
|---|---:|---:|---:|---:|---:|
| 10^12 | 37,607,912,018 | 5.91 s | 2.37 s | **2.48 s** | ≈ 2.0 – 2.4 s |
| 10^13 | 346,065,536,839 | 75.9 s | 30.3 s | **31.3 s** | ≈ 26 – 30 s |

LingSieve figures are taken from the [LingSieve README](https://github.com/LingUaan/LingSieve). Its author describes it as the fastest GPU implementation of the sieve of Eratosthenes he knows of. Furious Speed reaches the same times on a GPU that is three generations older (Turing vs. Blackwell).

**How the estimate was made.** The last column is a model, not a measurement. The two cards differ as follows:

| | RTX 2080 Ti (Turing) | RTX 5060 Ti (Blackwell) | Ratio |
|---|---:|---:|---:|
| SMs × boost clock | 68 × 1545 MHz | 36 × 2572 MHz | 0.88 |
| INT32 lanes per SM | 64 | 128 | |
| INT32 throughput | | | 1.76 |
| Shared memory per SM | 64 KB | up to 100 KB | |
| L2 cache | 5.5 MB | 32 MB | |
| Memory bandwidth | 616 GB/s | 448 GB/s | |

At 10^12 the measured time inside a segment splits roughly as follows:

- pre-sieving: about 35 %. It is integer- and L2-bound, so it should scale by about 1.5–1.76.
- crossing off with the wheel primes and the other small primes: about 60 %. This is dominated by shared-memory atomics, which scale with SMs × clock, by about 0.88–1.0.
- counting: about 5 %.

Weighting the phases gives a factor of 0.87–0.95, which is the range in the table. Larger segments, possible with 100 KB of shared memory, would push it towards the lower end. At 10^13 the big-prime kernel (about 25 %) would additionally profit from the 32 MB L2.

## Algorithm

The core is a segmented sieve of Eratosthenes: O(x log log x) operations and O(√x) memory. At these sizes the run time is limited by the memory hierarchy and by branch prediction more than by arithmetic, so every design decision below is about keeping data in L1/L2 and branches predictable.

1. **Exact integer square root.** √x is computed bit by bit, with shifts, additions and comparisons only. The function is `constexpr`, so the compiler can evaluate it at compile time. A floating-point root can be off by one near prime squares and silently miss a sieving prime.
2. **First segment.** `[0, G)` with `G ≥ √x`, rounded up to a multiple of 510510. A small 6k±1 sieve counts its primes and stores all sieving primes at the same time. Every later segment starts on a multiple of 510510, and no sieving prime ever lies inside a sieved segment.
3. **Segments.** Everything after the first segment is split into segments that live in the cache. Segments are processed by a pool of threads.

## CPU engine

**Bit array of odd numbers.** Only odd numbers are stored, one bit each: bit `k` of a segment starting at `b` is the number `b + 2k + 1`. A 64-bit word covers 128 numbers.

**Sieving primes as half gaps.** All sieving primes are stored as `(p_next − p) / 2` in one byte. The first gap above 510 occurs after 304,599,508,537, whose square is far beyond 2^64, so one byte is enough for the full 64-bit range. This is 77 KB instead of 314 KB at 10^12.

**Modulo 30030 wheel.** The wheel skips multiples of 2, 3, 5, 7, 11 and 13, leaving 38 % of the odd multiples. It uses three tables:

- step lengths: 5760 entries, 1 byte each;
- admissible residues: 2 bytes each;
- an entry table for all 30030 residues: 2 bytes each.

All three together take about 80 KB.

**Entry by residue classes.** The first multiple `m·p ≥ b` with `gcd(m, 30030) = 1` comes from `s = ⌈b/p⌉`, `r = s mod 30030` (a constant modulus, which the compiler turns into a multiplication) and two table lookups. There is no trial-and-error loop, so there are no unpredictable branches.

**Doubled step table.** The 5760 steps are stored twice (11,520 bytes). The wheel span of a prime, `p · 30030 / 2` bits, is larger than an L1 block for `p ≥ 14` and larger than a segment for medium primes. A prime therefore wraps around the table at most once per run, and the end-of-table check moves out of the inner loop. This saves about 6 %. It is also the real reason for choosing 30030 over 210:

- with a mod 210 wheel the check would have to stay in the loop;
- a mod 510510 wheel saves only 2 more percentage points of crossings, but needs about 2.4 MB of tables.

**Pre-sieve up to 223 (AVX2).** The primes 3 … 223 are split into 16 groups, following the approach of [primesieve](https://github.com/kimwalisch/primesieve). Each group has its own periodic pattern, 524 KB for all 16 together (primesieve pre-sieves up to 163). For each segment the 16 patterns are ANDed into place with AVX2:

- four patterns per pass, so the 16 shift constants fit in the 16 YMM registers without spilling;
- the work is done in 8 KB chunks that stay in L1;
- the first pass stores, the other three AND;
- an overlapping last vector replaces the scalar tail loop;
- there is one canonical copy per pattern with shift-merge, instead of 8 pre-rotated copies (1/8 of the memory);
- the patterns are extended by their own beginning, so there is no wrap check.

This stage went through five generations. It removes about 70 % of all crossings, at about 8 % of the run time. A single 2…17 pattern leaves 36.1 % of the odd numbers as candidates; the groups up to 223 leave 20.6 %.

**Three classes of sieving primes.**

| Class | Range (default) | Technique |
|---|---|---|
| small | 227 … ~20,000 | Crossed off in 24 KB L1 blocks inside the segment. The position and wheel index of each prime are kept between blocks (more than 20 % faster). |
| medium | … ~2.5·10^6 | State kept from segment to segment, so there is no division after the first segment of a work package. 128 entries are computed together so the divisions overlap. |
| big | above | Bucket sieve: each prime is placed into the bucket of the segment it hits next. Entries are 8 bytes (half prime + mod 210 wheel index, and the next bit). The segment index comes from a Lemire multiply-high instead of a division. |

**State instead of entry.** At 10^12 a thread would compute about 78,500 entries per segment, roughly 3 million cycles. With the kept state (4-byte bit position + 2-byte wheel index, 613 KB per thread at 10^12) only the first segment of a package needs them.

**Inner loop.**

- Bits are cleared by `p[k >> 6] &= ~(1ULL << k)` without masking the shift count. On 64-bit words MSVC compiles this to a single `btr`; the switch from bytes to 64-bit words gave about 8 %.
- Small and medium primes are processed in pairs, which gives two independent dependency chains.
- Loop variables are copied into locals so the compiler keeps them in registers.

**Threads.** A queue holds only the start of each work package (8 bytes). A package is 30 consecutive segments, so states can be reused. The threads are completely independent: no locks while sieving, and load balancing comes from the queue. The segment size is about 45 % of a P-core L2 (2 MB). That is about 90 % of the 1 MB per thread on both core types: two hyperthreads share 2 MB on a P-core, four E-cores share 4 MB. The remaining 10 % is left for the small tables, and the read-only pattern and wheel tables are shared through the L3. Using all 20 threads instead of one per core gives about 7 %.

**Everything is a parameter.** The segment size, block size, class limits, package size, batch size and pre-sieve set can all be set from the GUI. Most gains came from re-tuning old ideas after other changes.

**What did not work** (all measured, all removed):

| Idea | Effect | Why it failed |
|---|---:|---|
| Precomputed `step · p` table ("kdiff") | up to +20 % | The multiply is off the critical path. The table adds loads and is 4× larger than L1. |
| Bit mask from a 64-entry table | +7 % | Trades ALU work for an extra load. |
| Inner loop unrolled 4× | +5 % | The loop branch was already well predicted; the larger loop no longer fits the µop buffer. |
| Fully unrolled mod 210 wheel (`switch`) | much slower | Overflows the instruction cache, and the indirect jump is unpredictable. |
| Separate parameters for E-cores | slower | Two pattern sets, and the queue already balances the cores. |
| Mod 510510 wheel | slower | 2.4 MB of tables against 2 percentage points fewer crossings. |

Compiler: for this code, MSVC produced clearly faster binaries than LLVM/Clang.

## GPU engine (CUDA)

The CPU tricks do not carry over directly. The GPU has no integer divide instruction, 64-bit arithmetic is emulated, tables in memory are slow, and a thread block always waits for its slowest thread.

- **One block = one segment in shared memory.** By default 47 KB (385,024 bits = 770,048 numbers) with 1,024 threads per block. `__launch_bounds__` limits the kernel to 64 registers with no spills. Unlike on the CPU, all segments of a chunk run at the same time, so no state can be passed on and every segment needs its own entry points.
- **Sieving primes as flat 32-bit array.** Thread `i` must read prime `i` directly; gaps would need a prefix sum.
- **Mod 30 wheel in registers.** The 8 half-steps `3,2,1,2,1,2,3,1` are packed into the constant `0x13212123`. The residues and the entry index are packed into two more constants. There are no memory accesses and no divergent branches. It is used only for primes below 16,384 (−16 %). A table-based 210/2310/30030 wheel was 9 %, 28 % and 93 % *slower*.
- **Pre-sieve with funnel shifts.** Each thread builds the words `t, t+1024, …`, so a warp reads neighbouring words and the loads are coalesced. All 16 patterns are combined per word using `__funnelshift_r` on 32-bit words. No atomics are needed, because every word has exactly one owner (−9 %).
- **Load balancing.** Wheel primes are split into slices of about 48 hits each (prime, first bit, last bit). Before, the thread with p = 227 did about 900 crossings while the block waited at the barrier for it. This step halved the run time (−48 %), and after it the largest segment variant became the best.
- **Division without division.** The remainder `b mod p` and the quotient mod 30 are carried from chunk to chunk by adding a precomputed `C mod p`. Inside a chunk all values are 32-bit, and `x / p` uses `__fdividef` plus one correction in each direction, which is exact for `x < 2^27`. Written as a loop, the correction was turned back into a division by the compiler. Gain at 10^13: −33 %.
- **Big primes.** One thread per prime crosses off with `atomicAnd` in a chunk buffer sized to about 60 % of L2.
- **Few launches.** Counting is done in the segment kernel when there are no big primes, with two waves per chunk. The chunk index lives on the device, and 32 chunks are recorded once as a CUDA Graph: about 6,000 launches instead of about 760,000 at 10^13.
- **Auto-tuning.** At startup the program reads the number of SMs, shared memory, registers, L2 size and occupancy from the card. From these it derives the segment variant, the segment/buffer limit (about ⅔ of a segment length, or everything inside the segment if √x < about 2.7 segment lengths), the wheel limit and the waves per chunk. It is built for sm_75, sm_86, sm_89 and sm_120, plus PTX.

| GPU step (10^12) | Time relative to first GPU version |
|---|---:|
| first GPU version (thread per prime, 16 KB segments) | 100 % |
| mod 30 wheel in registers | 89 % |
| 32-bit pre-sieve with funnel shift | 81 % |
| load-balanced slices, 47 KB segments | 43 % |
| count in kernel, full power limit | 39 % |
| carried remainders, CUDA Graph | 38 % |
| auto-tuned parameters | 34 % |

## Verification

Every change is checked against known values of π(x):

- 10^9: 50,847,534
- 10^10: 455,052,511
- 10^11: 4,118,054,813
- 10^12: 37,607,912,018
- 10^13: 346,065,536,839

The tests use parameter sets that exercise every path (for example, artificially low class limits and odd upper bounds). Without a local GPU, the kernels were verified with a CPU emulation of blocks, barriers and atomics, and with clang `--cuda-device-only` builds.

## References

1. Kim Walisch, [primesieve](https://github.com/kimwalisch/primesieve) and its [algorithm notes](https://github.com/kimwalisch/primesieve/blob/master/doc/ALGORITHMS.md): pre-sieve groups, libpopcnt, cache-size detection.
2. LingUaan, [LingSieve](https://github.com/LingUaan/LingSieve): CUDA sieve of Eratosthenes, GPU reference times.
3. D. Lemire, O. Kaser, N. Kurz, *Faster Remainder by Direct Computation*, 2019.
4. T. Oliveira e Silva, tables of π(x).
