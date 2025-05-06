#include "benchmarks/benchmark_hybrid_pgm_lipp.h"

#include "benchmark.h"
#include "benchmarks/common.h"
#include "competitors/hybrid_pgm_lipp.h"

template <typename Searcher>
void benchmark_64_hybrid_pgm_lipp(tli::Benchmark<uint64_t>& benchmark, bool pareto, const std::vector<int>& params) {
  if (!pareto) {
    util::fail("HybridPGMLIPP requires pareto mode with fixed hyperparameters");
  } else {
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 8>>();
  }
}

template <int record>
void benchmark_64_hybrid_pgm_lipp(tli::Benchmark<uint64_t>& benchmark, const std::string& filename) {
  benchmark.template Run<HybridPGMLIPP<uint64_t, LinearSearch<record>, 8>>();
}

INSTANTIATE_TEMPLATES_MULTITHREAD(benchmark_64_hybrid_pgm_lipp, uint64_t);
