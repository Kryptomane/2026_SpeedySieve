# Furious Speed

**Algorithm optimisation on processor architecture – counting primes really fast.**

Furious Speed counts all primes from 0 up to any limit below 2^64 ≈ 1.8 · 10^19. It uses a segmented sieve of Eratosthenes and comes with two engines:

- a multi-threaded **CPU engine** in modern C++ with AVX2,
- a **GPU engine** in CUDA for NVIDIA graphics cards.

The project started about 15 years ago as a small Pascal program that tested every number by trial division. That took 8 minutes to count the primes up to 10^9. Today the CPU engine counts up to 10^12, a thousand times further, in under 15 seconds, and the GPU engine does it in 2.5 seconds.

This repository contains the documentation only, not the source code.

---

## The main idea

Run time is always a competition between **data availability** and the **number of instructions**.

The fewest instructions come from a simple wheel sieve running on one thread over one big table. But for large ranges that table no longer fits into the CPU cache, and the processor spends most of its time waiting for main memory.

To avoid this waiting, the number range is cut into **segments** that fit into the cache. Each segment is sieved completely, counted and then thrown away. This costs a few extra instructions per segment, but it is far cheaper than waiting for memory. It also lets the segments be shared between all CPU cores.

At 10^12 the run time is limited less by arithmetic than by the memory hierarchy (L1/L2 latency and bandwidth) and by branch mispredictions. The whole design is built around these two limits.

---

## How a run works

1. **Exact square root.** Only primes up to √x are needed for sieving. √x is calculated with integers only, bit by bit, so it is always exact. The function is `constexpr`, so the compiler can even evaluate it at compile time. A floating-point root can be off by one and silently lose a sieving prime.
2. **First segment.** The range `[0, G)` with `G ≥ √x` is sieved by a small, simple sieve. It counts the primes in that range and, at the same time, stores all sieving primes. `G` is a multiple of 510510, so every later segment starts at a multiple of 510510.
3. **Precomputation.** The wheel tables and the pre-sieve patterns are built once.
4. **Segments.** All further segments are processed in parallel by a pool of threads. Each segment runs through the same steps: pre-sieve, small primes, medium primes, big primes, count.

---

## CPU engine

### Tricks that are used

- **Only odd numbers, one bit each.** Bit `k` of a segment starting at `b` stands for the number `b + 2k + 1`. A 64-bit word covers 128 numbers.
- **Segments sized for the L2 cache.** Each thread works on its own segment, about 45 % of the L2 cache of a performance core (about 7.6 million bits, 15 million numbers). The segment is touched by only one thread, so no locking is needed.
- **Sieving primes stored as half gaps.** Each sieving prime is stored as half the distance to the previous one, in a single byte. The first prime gap that does not fit into a byte appears far beyond what a 64-bit number can reach. Up to 10^12 the sieving primes need 77 KB instead of 314 KB.
- **Pre-sieving with AVX2.** Before any sieving prime crosses anything off, the multiples of all primes from 3 to 223 are removed at once. This uses 16 small periodic tables, one per group of primes, which are combined with AVX2:
  - four tables per pass, in 8 KB chunks that stay in L1;
  - each table is extended periodically, so a run can cross the end of a table without a special case.

  The idea of grouped pre-sieve tables comes from primesieve. This engine uses them up to 223 (524 KB), where primesieve stops at 163. Pre-sieving removes about 70 % of all crossings, at about 8 % of the run time.
- **Wheel of 30030.** Multiples of 2, 3, 5, 7, 11 and 13 are skipped, so only 38 % of the odd multiples remain. For comparison, a mod-30 wheel keeps 53 % and a mod-210 wheel 46 %.
- **Doubled step table.** The 5760 wheel steps are stored twice in a row (11,520 bytes). One turn of the wheel moves a prime further than a whole L1 block or segment, so a prime can never run past the end of the doubled table. The inner loop needs no wrap-around check, and its only branch is the loop exit. The index is reduced once after the loop.
- **Entry by residue classes.** To find where a prime starts in a segment, one division gives the first multiple. Its residue modulo 30030 then picks the right wheel position from a precomputed table. The modulo by a constant becomes a multiplication, so there is no searching and no unpredictable branch.
- **Clearing a bit with one instruction.** A bit is cleared by `p[k >> 6] &= ~(1ULL << k)`, without masking the shift count. On 64-bit words MSVC compiles this to a single `btr` instruction.
- **Keeping loop variables in registers.** Loop variables are copied into local variables so that the compiler keeps them in registers.

### Three kinds of sieving primes

Small primes hit a segment thousands of times, big primes almost never. Each kind gets its own technique:

| Kind | Primes | Hits per segment | Technique |
|---|---|---|---|
| small | up to ≈ 20,000 | hundreds to thousands | Crossed off in L1-sized blocks (24 KB) inside the segment. Two primes at a time give two independent dependency chains. Position and wheel index of every prime are kept between blocks and between segments. |
| medium | up to ≈ 2.5 · 10^6 | a few to ≈ 150 | Same kept state per prime, so later segments need no division and no table lookups. The first segment of a work package starts 128 primes at once, so that their divisions overlap. |
| big | above | about one or less | Bucket sieve: each prime waits in the bucket of the segment it hits next. A bucket entry is 8 bytes. The bucket index is found by a multiplication instead of a division. |

### Multithreading

- A shared queue holds only the start of each work package, a single 64-bit number. Each thread has all other data itself.
- A work package is 30 consecutive segments, so the kept state of each prime can be reused from segment to segment.
- Threads are completely independent. Faster cores simply take more packages.
- On the test CPU each thread effectively has about 1 MB of L2: two hyperthreads share the 2 MB of a performance core, and four efficiency cores share 4 MB. The segment uses about 90 % of that. The read-only pre-sieve and wheel tables are shared by all threads through the L3 cache.

### Everything is a parameter

Segment size, block size, the limits between small, medium and big primes, package size, batch size and the pre-sieve set can all be changed from the GUI. Many of the gains came from re-tuning old ideas after something else had changed.

---

## GPU engine

A graphics card works very differently from a CPU. It has no integer division instruction, 64-bit arithmetic is slow, tables in memory are expensive, and a group of threads always waits for its slowest member. Most CPU tricks had to be rebuilt.

- **One thread block per segment.** Each segment (47 KB = 770,048 numbers) lives in the fast shared memory of one multiprocessor and is sieved by 1,024 threads together. Bits are cleared with atomic operations, because several threads may hit the same word.
- **Sieving primes as 32-bit numbers.** Each thread must read its own prime directly, so the byte gaps of the CPU version are expanded into a flat list.
- **A wheel without memory.** The mod-30 wheel is packed into a few constants. The eight steps `3,2,1,2,1,2,3,1` fit into the single number `0x13212123`. There are no memory accesses and no diverging branches. The wheel is only used for primes below 16,384, which hit a segment often enough to profit from it.
- **Pre-sieving with funnel shifts.** Thread `t` builds the words `t, t+1024, …`, so neighbouring threads read neighbouring memory. All 16 patterns are combined in 32-bit steps with the GPU's funnel-shift instruction.
- **Equal work for every thread.** Small primes are split into slices of about 48 hits each. Before this change, the thread with prime 227 did 900 crossings while all others waited for it. This step alone halved the run time.
- **Division without division.**
  - The remainder of each prime is carried from one chunk to the next by an addition instead of a division.
  - Inside a chunk all values fit into 32 bits, and the quotient comes from a fast approximate reciprocal plus one correction step.
- **Big primes.** One thread per prime crosses off in a chunk buffer that fits into about 60 % of the L2 cache.
- **Few kernel launches.**
  - Without big primes, each block counts its own segment.
  - 32 chunks are recorded once as a CUDA Graph and started with a single call.
- **Auto-tuning.** At start-up the program reads the number of multiprocessors, shared memory, registers and L2 size of the card and chooses all GPU parameters itself. It is built for all NVIDIA generations from Turing to Blackwell.

---

## Verification

Every change is checked against known values of π(x) with parameter sets that use every code path:

| x | π(x) |
|---|---:|
| 10^9 | 50,847,534 |
| 10^10 | 455,052,511 |
| 10^11 | 4,118,054,813 |
| 10^12 | 37,607,912,018 |
| 10^13 | 346,065,536,839 |
| 10^14 | 3,204,941,750,802 |

---

## Benchmarks

### CPU

Intel Core i5-13600KF (6 performance + 8 efficiency cores, 20 threads), Windows, MSVC. primesieve was measured on the same machine.

| x | primesieve | Furious Speed |
|---|---:|---:|
| 10^12 | 17 s | **14.6 s** (−14 %) |
| 10^13 | 203 s | **176 s** (−13 %) |
| 2 · 10^13 | – | **390 s** |

### GPU

| x | LingSieve <br>GTX 1080 | LingSieve <br>RTX 5060 Ti | Furious Speed <br>RTX 2080 Ti | Furious Speed <br>RTX 5060 Ti (*estimated*) |
|---|---:|---:|---:|---:|
| 10^12 | 5.91 s | 2.37 s | **2.48 s** | ≈ 2.2 s (−7 %) |
| 10^13 | 75.9 s | 30.3 s | **31.3 s** | ≈ 28 s (−8 %) |

LingSieve figures are taken from the [LingSieve README](https://github.com/LingUaan/LingSieve). The last column is an estimate, not a measurement; the percentages compare it with LingSieve on the same card.

---

## Third-party code and references

- Kim Walisch, [primesieve](https://github.com/kimwalisch/primesieve): pre-sieve groups. Also by him: `libpopcnt` and the L1/L2 cache size detection used here.
- Thread queue adapted from juanchopanzacpp.wordpress.com.
- LingUaan, [LingSieve](https://github.com/LingUaan/LingSieve): CUDA sieve of Eratosthenes, GPU reference times.
- D. Lemire, O. Kaser, N. Kurz: *Faster Remainder by Direct Computation*, 2019.

## Author

Silas Maile
