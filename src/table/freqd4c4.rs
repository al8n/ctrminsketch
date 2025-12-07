// Portions of this module are adapted from
// Caffeine's `FrequencySketch` (Apache-2.0):
// https://github.com/ben-manes/caffeine/blob/master/caffeine/src/main/java/com/github/benmanes/caffeine/cache/FrequencySketch.java

use core::num::NonZeroUsize;

use super::{D4C4Storage, FixedSizeStorage, Table, TableViewD4C4, WithCapacity};
use varing::{
  decode_u32_varint, encode_u32_varint_to,
  encoded_u32_varint_len, DecodeError, EncodeError, InsufficientSpace,
};

const RESET_MASK: u64 = 0x7777_7777_7777_7777;

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
  /// Creates a new `FreqD4C4` with the given capacity.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_capacity(capacity: usize) -> Self
  where
    T: WithCapacity,
  {
    let capacity = capacity as u32;

    // clamp max size like Caffeine
    let maximum = capacity.min(i32::MAX as u32 >> 1);

    // table length = next power of two >= maximum, min 8
    let table_len = maximum.next_power_of_two().max(8);
    // allocate 64-bit counter blocks
    let table = T::with_capacity(table_len as usize);

    // sampleSize = 10 * maximum (bounded)
    let sample_size = if capacity == 0 {
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

impl<T> FreqD4C4<T> {
  /// Returns the mask
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn mask(&self) -> u32 {
    self.block_mask
  }

  /// Returns the current size
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn size(&self) -> u32 {
    self.size
  }

  /// Returns the sample size
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sample_size(&self) -> u32 {
    self.sample_size
  }

  /// Returns a reference to the underlying table storage.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn table(&self) -> &T {
    &self.table
  }
}

impl<T> FreqD4C4<T>
where
  T: D4C4Storage,
{
  /// Encodes the sketch into the provided byte slice in compact form.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn encode_compact_to(&self, dst: &mut [u8]) -> Result<NonZeroUsize, EncodeError> {
    let cap = dst.len();
    let encoded_len = self.compact_encoded_len();
    if cap < encoded_len.get() {
      return Err(EncodeError::InsufficientSpace(InsufficientSpace::new(
        encoded_len,
        cap,
      )));
    }

    let mut offset = 0;
    offset += encode_u32_varint_to(self.size, &mut dst[offset..])?.get();
    offset += encode_u32_varint_to(self.sample_size, &mut dst[offset..])?.get();
    offset += encode_u32_varint_to(self.block_mask, &mut dst[offset..])?.get();
    offset += self.table.encode_compact_to(&mut dst[offset..])?;
    // Safety: offset > 0
    Ok(unsafe { NonZeroUsize::new_unchecked(offset) })
  }

  /// Returns the encoded length in bytes in compact form.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn compact_encoded_len(&self) -> NonZeroUsize {
    // Safety: encoded length is always > 0
    unsafe {
      NonZeroUsize::new_unchecked(
        encoded_u32_varint_len(self.size).get()
          + encoded_u32_varint_len(self.sample_size).get()
          + encoded_u32_varint_len(self.block_mask).get()
          + { self.table.compact_encoded_len() },
      )
    }
  }

  /// Encodes the sketch into the provided byte slice in fixed form.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn encode_to(&self, dst: &mut [u8]) -> Result<NonZeroUsize, EncodeError> {
    let cap = dst.len();
    let encoded_len = self.encoded_len();
    if cap < encoded_len.get() {
      return Err(EncodeError::InsufficientSpace(InsufficientSpace::new(
        encoded_len,
        cap,
      )));
    }

    let mut offset = 0;
    dst[offset..offset + 4].copy_from_slice(&self.size.to_le_bytes());
    offset += 4;
    dst[offset..offset + 4].copy_from_slice(&self.sample_size.to_le_bytes());
    offset += 4;
    dst[offset..offset + 4].copy_from_slice(&self.block_mask.to_le_bytes());
    offset += 4;
    offset += self.table.encode_to(&mut dst[offset..])?;
    // Safety: offset > 0
    Ok(unsafe { NonZeroUsize::new_unchecked(offset) })
  }

  /// Returns the encoded length in bytes in fixed form.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn encoded_len(&self) -> NonZeroUsize {
    // Safety: encoded length is always > 0
    unsafe {
      NonZeroUsize::new_unchecked(
        4 + // size
        4 + // sample_size
        4 + // block_mask
        self.table.encoded_len(),
      )
    }
  }

  /// Decodes the sketch from the provided byte slice in fixed form.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn decode(src: &[u8]) -> Result<(usize, Self), DecodeError>
  where
    T: super::TryFromIterator<Item = u64>,
  {
    let mut offset = 0;

    if src.len() < offset + 4 {
      return Err(DecodeError::InsufficientData {
        available: src.len(),
      });
    }

    let size = u32::from_le_bytes(src[offset..offset + 4].try_into().unwrap());
    offset += 4;

    if src.len() < offset + 4 {
      return Err(DecodeError::InsufficientData {
        available: src.len(),
      });
    }
    let sample_size = u32::from_le_bytes(src[offset..offset + 4].try_into().unwrap());
    offset += 4;
    if src.len() < offset + 4 {
      return Err(DecodeError::InsufficientData {
        available: src.len(),
      });
    }
    let block_mask = u32::from_le_bytes(src[offset..offset + 4].try_into().unwrap());
    offset += 4;

    let table_len = (block_mask + 1) << 3;
    if src.len() < offset + (table_len as usize * 8) {
      return Err(DecodeError::InsufficientData {
        available: src.len(),
      });
    }

    let (read, table) = T::decode(&src[offset..], table_len as usize)?;
    offset += read;

    Ok((
      offset,
      Self {
        size,
        sample_size,
        block_mask,
        pad: 0,
        table,
      },
    ))
  }

  /// Decodes the sketch from the provided byte slice in compact form.
  pub fn decode_compact(src: &[u8]) -> Result<(usize, Self), DecodeError>
  where
    T: super::TryFromIterator<Item = u64>,
  {
    let mut offset = 0;

    let (sz_len, size) = decode_u32_varint(&src[offset..])?;
    offset += sz_len.get();

    let (ss_len, sample_size) = decode_u32_varint(&src[offset..])?;
    offset += ss_len.get();

    let (bm_len, block_mask) = decode_u32_varint(&src[offset..])?;
    offset += bm_len.get();

    let table_len = (block_mask + 1) << 3;
    let (read, table) = T::decode_compact(&src[offset..], table_len as usize)?;
    offset += read;

    Ok((
      offset,
      Self {
        size,
        sample_size,
        block_mask,
        pad: 0,
        table,
      },
    ))
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
    const ONE_MASK: u64 = 0x1111_1111_1111_1111;

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

#[cfg(test)]
#[cfg(any(feature = "std", feature = "alloc"))]
mod tests {

  use super::*;

  fn random() -> u32 {
    getrandom::u32().expect("getrandom failed")
  }

  // Helper: make a sketch (like Java makeSketch)
  fn make_sketch(maximum: u32) -> FreqD4C4 {
    FreqD4C4::with_capacity(maximum)
  }

  #[test]
  fn increment_once() {
    let mut sketch = make_sketch(512);
    let h = random();
    sketch.increment(h as u64);
    assert_eq!(sketch.estimate(h as u64), 1);
  }

  #[test]
  fn increment_max() {
    let mut sketch = make_sketch(512);
    let h = random();
    for _ in 0..20 {
      sketch.increment(h as u64);
    }
    assert_eq!(sketch.estimate(h as u64), 15);
  }

  #[test]
  fn increment_distinct() {
    let mut sketch = make_sketch(512);

    let a = random();
    let b = a + 1;
    let c = b + 1;

    sketch.increment(a as u64);
    sketch.increment(b as u64);

    assert_eq!(sketch.estimate(a as u64), 1);
    assert_eq!(sketch.estimate(b as u64), 1);
    assert_eq!(sketch.estimate(c as u64), 0);
  }

  #[test]
  fn increment_zero() {
    let mut sketch = make_sketch(512);

    sketch.increment(0);
    assert_eq!(sketch.estimate(0), 1);
  }

  #[test]
  fn reset_behavior() {
    let mut sketch = make_sketch(64);
    let mut reset_happened = false;

    for i in 1..(20 * sketch.table.len() as u32) {
      sketch.increment(i as u64);
      if sketch.size != i {
        reset_happened = true;
        break;
      }
    }

    assert!(reset_happened);
    assert!(sketch.size <= sketch.sample_size / 2);
  }

  #[test]
  fn full() {
    let mut sketch = make_sketch(512);
    sketch.sample_size = u32::MAX;

    for i in 0..100_000 {
      sketch.increment(i as u64);
    }

    // Every slot should have 64 bits = all counters = 4-bit full
    for slot in sketch.table.iter() {
      assert_eq!(slot.count_ones(), 64); // full bits
    }

    sketch.reset();

    for slot in sketch.table.iter() {
      assert_eq!(*slot, RESET_MASK);
    }
  }

  #[test]
  fn heavy_hitters() {
    let mut sketch = make_sketch(512);

    for i in 100..100_000u32 {
      sketch.increment(i as u64);
    }

    for i in (0..10u32).step_by(2) {
      for _ in 0..i {
        sketch.increment(i as u64);
      }
    }

    let mut popularity = [0u8; 10];
    for (i, p) in popularity.iter_mut().enumerate() {
      *p = sketch.estimate(i as u64);
    }

    // Exactly match Java logic
    for i in 0..10 {
      if matches!(i, 0 | 1 | 3 | 5 | 7 | 9) {
        assert!(popularity[i] <= popularity[2]);
      } else if i == 2 {
        assert!(popularity[2] <= popularity[4]);
      } else if i == 4 {
        assert!(popularity[4] <= popularity[6]);
      } else if i == 6 {
        assert!(popularity[6] <= popularity[8]);
      }
    }
  }
}
