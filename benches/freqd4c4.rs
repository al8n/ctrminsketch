#![allow(warnings)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ctrminsketch::{FreqD4C4, Table};
use std::hint::black_box;

/// Generate a pseudo-random sequence for benchmarking
fn gen_sequence(seed: u64, len: usize) -> Vec<u64> {
  let mut rng = seed;
  (0..len)
    .map(|_| {
      // Simple LCG for reproducible benchmarks
      rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
      rng
    })
    .collect()
}

/// Benchmark increment operations
fn bench_increment(c: &mut Criterion) {
  let mut group = c.benchmark_group("increment");

  for capacity in [256, 1024, 4096, 16384] {
    group.throughput(Throughput::Elements(1));
    group.bench_with_input(
      BenchmarkId::new("single", capacity),
      &capacity,
      |b, &cap| {
        let mut sketch: FreqD4C4 = FreqD4C4::with_capacity(cap);
        let mut counter = 0u64;
        b.iter(|| {
          counter = counter.wrapping_add(1);
          sketch.increment(black_box(counter));
        });
      },
    );
  }

  group.finish();
}

/// Benchmark estimate operations
fn bench_estimate(c: &mut Criterion) {
  let mut group = c.benchmark_group("estimate");

  for capacity in [256, 1024, 4096, 16384] {
    group.throughput(Throughput::Elements(1));

    // Pre-populate sketch
    let mut sketch: FreqD4C4 = FreqD4C4::with_capacity(capacity);
    let sequence = gen_sequence(12345, capacity * 10);
    for &hash in &sequence {
      sketch.increment(hash);
    }

    group.bench_with_input(BenchmarkId::new("lookup", capacity), &capacity, |b, _| {
      let mut counter = 0u64;
      b.iter(|| {
        counter = counter.wrapping_add(1);
        black_box(sketch.estimate(black_box(counter)))
      });
    });
  }

  group.finish();
}

/// Benchmark combined increment + estimate (typical cache usage pattern)
fn bench_mixed_workload(c: &mut Criterion) {
  let mut group = c.benchmark_group("mixed_workload");

  for capacity in [256, 1024, 4096] {
    group.throughput(Throughput::Elements(100));
    group.bench_with_input(
      BenchmarkId::new("80_20_read_write", capacity),
      &capacity,
      |b, &cap| {
        let mut sketch: FreqD4C4 = FreqD4C4::with_capacity(cap);
        let sequence = gen_sequence(54321, 100);

        b.iter(|| {
          for (i, &hash) in sequence.iter().enumerate() {
            if i % 5 == 0 {
              // 20% writes
              sketch.increment(black_box(hash));
            } else {
              // 80% reads
              black_box(sketch.estimate(black_box(hash)));
            }
          }
        });
      },
    );
  }

  group.finish();
}

/// Benchmark reset (aging) operation
fn bench_reset(c: &mut Criterion) {
  let mut group = c.benchmark_group("reset");

  for capacity in [256, 1024, 4096, 16384] {
    // Pre-populate sketch to full
    let mut sketch: FreqD4C4 = FreqD4C4::with_capacity(capacity);
    let sequence = gen_sequence(99999, capacity * 20);
    for &hash in &sequence {
      sketch.increment(hash);
    }

    group.bench_with_input(
      BenchmarkId::new("full_table", capacity),
      &sketch,
      |b, sketch| {
        let mut s = sketch.clone();
        b.iter(|| {
          s.reset();
          // Restore state for next iteration
          s = sketch.clone();
        });
      },
    );
  }

  group.finish();
}

/// Benchmark encoding operations
fn bench_encoding(c: &mut Criterion) {
  let mut group = c.benchmark_group("encoding");

  for capacity in [256, 1024, 4096] {
    let mut sketch: FreqD4C4 = FreqD4C4::with_capacity(capacity);
    let sequence = gen_sequence(11111, capacity * 5);
    for &hash in &sequence {
      sketch.increment(hash);
    }

    // Fixed encoding
    let encoded_len = sketch.encoded_len().get();
    group.throughput(Throughput::Bytes(encoded_len as u64));
    group.bench_with_input(
      BenchmarkId::new("encode_fixed", capacity),
      &sketch,
      |b, sketch| {
        let mut buffer = vec![0u8; encoded_len];
        b.iter(|| {
          sketch.encode_to(&mut buffer).unwrap();
          black_box(&buffer);
        });
      },
    );

    // Compact encoding
    let compact_len = sketch.compact_encoded_len().get();
    group.throughput(Throughput::Bytes(compact_len as u64));
    group.bench_with_input(
      BenchmarkId::new("encode_compact", capacity),
      &sketch,
      |b, sketch| {
        let mut buffer = vec![0u8; compact_len];
        b.iter(|| {
          sketch.encode_compact_to(&mut buffer).unwrap();
          black_box(&buffer);
        });
      },
    );
  }

  group.finish();
}

/// Benchmark decoding operations
fn bench_decoding(c: &mut Criterion) {
  let mut group = c.benchmark_group("decoding");

  for capacity in [256, 1024, 4096] {
    let mut sketch: FreqD4C4 = FreqD4C4::with_capacity(capacity);
    let sequence = gen_sequence(22222, capacity * 5);
    for &hash in &sequence {
      sketch.increment(hash);
    }

    // Fixed decoding
    let encoded_len = sketch.encoded_len().get();
    let mut buffer = vec![0u8; encoded_len];
    sketch.encode_to(&mut buffer).unwrap();

    group.throughput(Throughput::Bytes(encoded_len as u64));
    group.bench_with_input(
      BenchmarkId::new("decode_fixed", capacity),
      &buffer,
      |b, buf| {
        b.iter(|| {
          let (_, decoded) = FreqD4C4::<Box<[u64]>>::decode(buf).unwrap();
          black_box(decoded);
        });
      },
    );

    // Compact decoding
    let compact_len = sketch.compact_encoded_len().get();
    let mut compact_buffer = vec![0u8; compact_len];
    sketch.encode_compact_to(&mut compact_buffer).unwrap();

    group.throughput(Throughput::Bytes(compact_len as u64));
    group.bench_with_input(
      BenchmarkId::new("decode_compact", capacity),
      &compact_buffer,
      |b, buf| {
        b.iter(|| {
          let (_, decoded) = FreqD4C4::<Box<[u64]>>::decode_compact(buf).unwrap();
          black_box(decoded);
        });
      },
    );
  }

  group.finish();
}

/// Benchmark realistic cache admission control scenario
fn bench_cache_admission(c: &mut Criterion) {
  let mut group = c.benchmark_group("cache_admission");

  // Simulate cache admission policy with frequency threshold
  for capacity in [1024, 4096] {
    group.throughput(Throughput::Elements(1000));
    group.bench_with_input(
      BenchmarkId::new("admission_policy", capacity),
      &capacity,
      |b, &cap| {
        let mut sketch: FreqD4C4 = FreqD4C4::with_capacity(cap);
        let requests = gen_sequence(33333, 1000);

        b.iter(|| {
          let mut admitted = 0;
          for &hash in &requests {
            let freq = sketch.estimate(hash);
            sketch.increment(hash);

            // Admit if frequency >= 2 (has been seen before)
            if freq >= 2 {
              admitted += 1;
            }
          }
          black_box(admitted);
        });
      },
    );
  }

  group.finish();
}

/// Benchmark creation overhead
fn bench_creation(c: &mut Criterion) {
  let mut group = c.benchmark_group("creation");

  for capacity in [256, 1024, 4096, 16384] {
    group.bench_with_input(
      BenchmarkId::new("with_capacity", capacity),
      &capacity,
      |b, &cap| {
        b.iter(|| {
          let sketch: FreqD4C4 = FreqD4C4::with_capacity(cap);
          black_box(sketch);
        });
      },
    );
  }

  // Fixed-size array creation (stack allocation)
  group.bench_function("new_fixed_64", |b| {
    b.iter(|| {
      let sketch: FreqD4C4<[u64; 64]> = FreqD4C4::new();
      black_box(sketch);
    });
  });

  group.finish();
}

/// Benchmark heavy hitter detection scenario
fn bench_heavy_hitters(c: &mut Criterion) {
  let mut group = c.benchmark_group("heavy_hitters");
  group.throughput(Throughput::Elements(10000));

  group.bench_function("zipf_distribution", |b| {
    let mut sketch: FreqD4C4 = FreqD4C4::with_capacity(4096);

    // Simulate Zipf-like distribution (common in caching)
    // 10% of items account for 90% of accesses
    let mut items = Vec::with_capacity(10000);
    for i in 0..1000 {
      // Hot items (10%)
      for _ in 0..9 {
        items.push(i);
      }
    }
    for i in 1000..10000 {
      // Cold items (90%)
      items.push(i);
    }

    b.iter(|| {
      sketch.clear();
      for &item in &items {
        sketch.increment(item);
      }

      // Check frequency of hot item
      let hot_freq = sketch.estimate(0);
      black_box(hot_freq);
    });
  });

  group.finish();
}

criterion_group!(
  benches,
  bench_increment,
  bench_estimate,
  bench_mixed_workload,
  bench_reset,
  bench_encoding,
  bench_decoding,
  bench_cache_admission,
  bench_creation,
  bench_heavy_hitters,
);

criterion_main!(benches);
