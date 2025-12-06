// Portions of this module are adapted from
// Caffeine's `FrequencySketch` (Apache-2.0):
// https://github.com/ben-manes/caffeine/blob/master/caffeine/src/main/java/com/github/benmanes/caffeine/cache/FrequencySketch.java

use super::{Table, TableViewD4C4, D4C4Storage, WithCapacity, FixedSizeStorage};

const RESET_MASK: u64 = 0x7777_7777_7777_7777;
const ONE_MASK: u64 = 0x1111_1111_1111_1111;

/// Implementation of [Caffeine's Frequency Sketch](https://github.com/ben-manes/caffeine/blob/master/caffeine/src/main/java/com/github/benmanes/caffeine/cache/FrequencySketch.java).
///
/// Depth: 4, Counter: 4-bit
#[derive(Clone, PartialEq, Eq, Hash)]
#[repr(C)]
#[cfg(any(feature = "std", feature = "alloc"))]
pub struct FreqD4C4<T = std::boxed::Box<[u64]>> {
  size: u32,
  sample_size: u32,
  block_mask: u32,
  // pad-only, not used currently, may be used in future for alignment
  pad: u32,
  table: T,
}

/// Implementation of [Caffeine's Frequency Sketch](https://github.com/ben-manes/caffeine/blob/master/caffeine/src/main/java/com/github/benmanes/caffeine/cache/FrequencySketch.java).
///
/// Depth: 4, Counter: 4-bit
#[derive(Clone, PartialEq, Eq, Hash)]
#[repr(C)]
#[cfg(not(any(feature = "std", feature = "alloc")))]
pub struct FreqD4C4<T> {
  size: u32,
  sample_size: u32,
  block_mask: u32,
  // pad-only, not used currently, may be used in future for alignment
  pad: u32,
  table: T,
}

impl<T> core::fmt::Debug for FreqD4C4<T>
where
  T: D4C4Storage,
{
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("FreqD4C4")
      .field("size", &self.size)
      .field("sample_size", &self.sample_size)
      .field("mask", &self.block_mask)
      .field("table", &TableViewD4C4(self.table.as_ref()))
      .finish()
  }
}

impl<T> Default for FreqD4C4<T>
where
  T: FixedSizeStorage,
{
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

impl<T> FreqD4C4<T>
where
  T: FixedSizeStorage,
{
  /// Creates a new `FreqD4C4` with fixed size storage.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    let table = T::EMPTY;
    let table_len = T::SIZE as u32;

    // sampleSize = 10 * maximum (bounded)
    let sample_size = 10u32.wrapping_mul(table_len);
    let block_mask = (table_len >> 3) - 1;

    Self {
      size: 0,
      sample_size,
      block_mask,
      pad: 0,
      table,
    }
  }
}

impl<T> FreqD4C4<T>
where
  T: WithCapacity,
{
  /// Creates a new `FreqD4C4` with the given cache capacity.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_capacity(cache_capacity: u32) -> Self
  where
    T: WithCapacity,
  {
    // clamp max size like Caffeine
    let maximum = cache_capacity.min(i32::MAX as u32 >> 1);

    // table length = next power of two >= maximum, min 8
    let table_len = maximum.next_power_of_two().max(8);
    // allocate 64-bit counter blocks
    let table = T::with_capacity(table_len as usize);

    // sampleSize = 10 * maximum (bounded)
    let sample_size = if cache_capacity == 0 {
      10
    } else {
      (10u32.wrapping_mul(maximum)).min(i32::MAX as u32)
    };
    let block_mask = (table_len >> 3) - 1;

    Self {
      size: 0,
      sample_size,
      block_mask,
      pad: 0,
      table,
    }
  }
}

impl<T> FreqD4C4<T>
where
  T: D4C4Storage,
{
  /// Returns the mask
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn mask(&self) -> u32 {
    self.block_mask
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn increment_at(&mut self, slot: usize, index: usize) -> bool {
    let offset = (index << 2) as u64;
    let mask = 0xF_u64 << offset;

    let slot = &mut self.table.as_mut()[slot];
    let value = *slot;

    if (value & mask) != mask {
      *slot = value + (1_u64 << offset);
      true
    } else {
      false
    }
  }
}

impl<T> Table for FreqD4C4<T>
where
  T: D4C4Storage,
{
  type Counter = u8;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn increment(&mut self, h: u64) {
    let bh = spread_d4(h);
    let ch = rehash_d4(bh);
    let block = (bh & self.block_mask).wrapping_shl(3);

    let h0 = ch;
    let h1 = ch >> 8;
    let h2 = ch >> 16;
    let h3 = ch >> 24;

    let idx0 = ((h0 >> 1) & 15) as usize;
    let idx1 = ((h1 >> 1) & 15) as usize;
    let idx2 = ((h2 >> 1) & 15) as usize;
    let idx3 = ((h3 >> 1) & 15) as usize;

    let block = block as usize;
    let slot0 = block + ((h0 & 1) as usize);
    let slot1 = block + ((h1 & 1) as usize) + 2;
    let slot2 = block + ((h2 & 1) as usize) + 4;
    let slot3 = block + ((h3 & 1) as usize) + 6;

    let added = self.increment_at(slot0, idx0)
      | self.increment_at(slot1, idx1)
      | self.increment_at(slot2, idx2)
      | self.increment_at(slot3, idx3);

    if added {
      self.size += 1;
      if self.size == self.sample_size {
        self.reset();
      }
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn estimate(&self, h: u64) -> Self::Counter {
    let bh = spread_d4(h);
    let ch = rehash_d4(bh);
    let block = (bh & self.block_mask).wrapping_shl(3);
    let mut freq = u8::MAX;

    let table = self.table.as_ref();

    macro_rules! unroll {
      ($idx:literal) => {{
        let h = ch >> ($idx << 3);
        let idx = ((h >> 1) & 15) as usize;
        let offset = h & 1;
        let slot = block + offset + ($idx << 1);
        let count = (table[slot as usize] >> (idx << 2)) & 0xF;
        freq = freq.min(count as u8);
      }};
    }

    unroll!(0);
    unroll!(1);
    unroll!(2);
    unroll!(3);

    freq
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn reset(&mut self) {
    let mut odd_count = 0u32;
    for w in self.table.as_mut() {
      odd_count += (*w & ONE_MASK).count_ones();
      *w = (*w >> 1) & RESET_MASK;
    }
    self.size = (self.size - (odd_count >> 2)) >> 1;
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn clear(&mut self) {
    self.table.as_mut().fill(0);
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn rehash_d4(mut h: u32) -> u32 {
  h = h.wrapping_mul(0x31848bab);
  h ^= h >> 14;
  h
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn spread_d4(h: u64) -> u32 {
  let mut h = (h ^ (h >> 32)) as u32;
  h ^= h >> 17;
  h = h.wrapping_mul(0xed5ad4bb);
  h ^= h >> 11;
  h = h.wrapping_mul(0xac4c1b51);
  h ^= h >> 15;
  h
}
