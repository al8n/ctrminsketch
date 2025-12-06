#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
mod cms4;

#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub use freqd4c4::FreqD4C4;

#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
mod freqd4c4;

/// A trait for frequency estimation tables.
pub trait Table {
  /// Type of counter used in the table.
  type Counter;

  /// Increments the counter for the given hash.
  fn increment(&mut self, h: u64);

  /// Estimates the count for the given hash.
  fn estimate(&self, h: u64) -> Self::Counter;

  /// Resets (ages) the table.
  fn reset(&mut self);

  /// Clears the table.
  fn clear(&mut self);
}

mod sealed {
  pub trait Sealed {}
}

#[doc(hidden)]
pub trait D4C4Storage: AsRef<[u64]> + AsMut<[u64]> + sealed::Sealed {}

#[doc(hidden)]
pub trait WithCapacity: sealed::Sealed {
  fn with_capacity(capacity: usize) -> Self;
}

#[doc(hidden)]
pub trait FixedSizeStorage: D4C4Storage {
  const EMPTY: Self;
  const SIZE: usize;
}

macro_rules! impl_d4c4_storage_for_array {
  ($($size:literal),+$(,)?) => {
    $(
      impl sealed::Sealed for [u64; $size] {}

      impl FixedSizeStorage for [u64; $size] {
        const EMPTY: Self = [0u64; $size];
        const SIZE: usize = $size;
      }

      impl D4C4Storage for [u64; $size] {}
    )+
  };
}

impl_d4c4_storage_for_array! {
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
  33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
  49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
}

/// A wrapper to pretty-print the table as list of 16 4-bit counters.
struct TableViewD4C4<'a>(&'a [u64]);

impl core::fmt::Debug for TableViewD4C4<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    let mut list = f.debug_list();

    for (slot_idx, word) in self.0.iter().enumerate() {
      list.entry(&SlotView { slot_idx, word });
    }

    list.finish()
  }
}

/// One slot displayed as: {slot: N, counters: [0, 3, 2, ...]}
struct SlotView<'a> {
  slot_idx: usize,
  word: &'a u64,
}

impl core::fmt::Debug for SlotView<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    let mut ds = f.debug_struct("Slot");

    // Extract 16 counters without allocating.
    struct Counters<'a>(&'a u64);

    impl core::fmt::Debug for Counters<'_> {
      fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut list = f.debug_list();
        for i in 0..16 {
          let c = ((self.0 >> (i * 4)) & 0xF) as u8;
          list.entry(&c);
        }
        list.finish()
      }
    }

    ds.field("slot", &self.slot_idx)
      .field("counters", &Counters(self.word))
      .finish()
  }
}

#[cfg(any(feature = "std", feature = "alloc"))]
const _: () = {
  impl sealed::Sealed for std::boxed::Box<[u64]> {}

  impl D4C4Storage for std::boxed::Box<[u64]> {}

  impl WithCapacity for std::boxed::Box<[u64]> {
    #[cfg_attr(not(tarpaulin), inline(always))]
    fn with_capacity(capacity: usize) -> Self {
      std::vec![0u64; capacity].into_boxed_slice()
    }
  }

  impl sealed::Sealed for std::vec::Vec<u64> {}

  impl D4C4Storage for std::vec::Vec<u64> {}

  impl WithCapacity for std::vec::Vec<u64> {
    #[cfg_attr(not(tarpaulin), inline(always))]
    fn with_capacity(capacity: usize) -> Self {
      vec![0u64; capacity]
    }
  }
};
