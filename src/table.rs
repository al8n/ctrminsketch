//! Core traits and storage implementations for Count-Min Sketch tables.
//!
//! This module provides the fundamental building blocks for Count-Min Sketch
//! implementations, including storage traits, encoding/decoding utilities, and
//! the main [`Table`] trait.

use core::num::NonZeroUsize;

use varing::{
  decode_u64_varint, encode_u64_varint_to, encoded_u64_varint_len, DecodeError, EncodeError,
  InsufficientSpace,
};

pub use freqd4c4::FreqD4C4;

mod freqd4c4;

/// A trait for frequency estimation tables.
///
/// This trait defines the core operations for probabilistic frequency counting
/// data structures. Implementations track approximate counts of items based on
/// their hash values.
///
/// # Example
///
/// ```rust
/// # #[cfg(any(feature = "std", feature = "alloc"))] {
/// use ctrminsketch::{FreqD4C4, Table};
///
/// let mut sketch: FreqD4C4 = FreqD4C4::boxed(256);
/// let hash = 42u64;
///
/// // Track frequency
/// sketch.increment(hash);
/// sketch.increment(hash);
///
/// // Estimate frequency
/// assert_eq!(sketch.estimate(hash), 2);
///
/// // Age all counters (halve them)
/// sketch.reset();
/// assert_eq!(sketch.estimate(hash), 1);
///
/// // Clear all counters
/// sketch.clear();
/// assert_eq!(sketch.estimate(hash), 0);
/// # }
/// ```
pub trait Table {
  /// Type of counter used in the table.
  ///
  /// This is typically `u8` for 4-bit counters (values 0-15).
  type Count;

  /// Increments the frequency counter for the given hash.
  ///
  /// The hash should be a well-distributed 64-bit value. Multiple hash
  /// functions are applied internally to update different counter positions.
  ///
  /// # Behavior
  ///
  /// - Counters saturate at their maximum value (implementation-specific)
  /// - May trigger automatic aging (reset) when sample size is reached
  fn increment(&mut self, h: u64);

  /// Estimates the frequency count for the given hash.
  ///
  /// Returns a conservative estimate - the actual count is guaranteed to be
  /// at least as high as the estimate, but may be higher due to hash collisions.
  ///
  /// # Returns
  ///
  /// The minimum of all counter positions associated with this hash.
  fn estimate(&self, h: u64) -> Self::Count;

  /// Resets (ages) the table by halving all counters.
  ///
  /// This operation:
  /// - Divides all counters by 2 (integer division)
  /// - Updates internal size tracking
  /// - Maintains recency bias by reducing old item counts
  ///
  /// Automatically called when the sample size threshold is reached.
  fn reset(&mut self);

  /// Clears the table, setting all counters to zero.
  ///
  /// This completely resets the frequency tracking state.
  fn clear(&mut self);
}

mod sealed {
  pub trait Sealed {}
}

#[doc(hidden)]
pub trait D4C4Storage: sealed::Sealed {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn encoded_len(&self) -> usize
  where
    Self: AsRef<[u64]>,
  {
    self.as_ref().len() * 8
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn compact_encoded_len(&self) -> usize
  where
    Self: AsRef<[u64]>,
  {
    self
      .as_ref()
      .iter()
      .map(|&w| varing::encoded_u64_varint_len(w).get())
      .sum()
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn encode_to(&self, dst: &mut [u8]) -> Result<usize, EncodeError>
  where
    Self: AsRef<[u64]>,
  {
    let mut offset = 0;
    let len = dst.len();
    for &w in self.as_ref() {
      if offset + 8 > len {
        return Err(EncodeError::InsufficientSpace(InsufficientSpace::new(
          // Safety: request must greater than 0, if we enter here.
          unsafe { NonZeroUsize::new_unchecked(self.encoded_len()) },
          len,
        )));
      }
      dst[offset..offset + 8].copy_from_slice(&w.to_le_bytes());
      offset += 8;
    }
    Ok(offset)
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn encode_compact_to(&self, dst: &mut [u8]) -> Result<usize, EncodeError>
  where
    Self: AsRef<[u64]>,
  {
    let mut offset = 0;
    let len = dst.len();
    for &w in self.as_ref() {
      let encoded_len = encoded_u64_varint_len(w).get();
      if offset + encoded_len > len {
        return Err(EncodeError::InsufficientSpace(InsufficientSpace::new(
          // Safety: request must greater than 0, if we enter here.
          unsafe { NonZeroUsize::new_unchecked(self.compact_encoded_len()) },
          len,
        )));
      }
      offset += encode_u64_varint_to(w, &mut dst[offset..offset + encoded_len])?.get();
    }
    Ok(offset)
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn decode(src: &[u8], want: usize) -> Result<(usize, Self), DecodeError>
  where
    Self: TryFromIterator<Item = u64> + Sized,
  {
    if src.len() < want * 8 {
      return Err(DecodeError::other("insufficient counters"));
    }

    let table = Self::try_from_iterator(
      src[..want * 8]
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap())),
    )?;

    Ok((want * 8, table))
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn decode_compact(src: &[u8], want: usize) -> Result<(usize, Self), DecodeError>
  where
    Self: TryFromIterator<Item = u64> + Sized + AsRef<[u64]>,
  {
    let mut num_elements = 0;
    let mut offset = 0;
    let table = Self::try_from_iterator(core::iter::from_fn(|| {
      // we need to check both src.len() and num_elements here
      // as we don't know how many bytes each u64 takes
      if offset < src.len() && num_elements < want {
        let (w_len, w) = decode_u64_varint(&src[offset..]).ok()?;
        offset += w_len.get();
        num_elements += 1;
        Some(w)
      } else {
        None
      }
    }))?;

    if table.as_ref().len() != want {
      return Err(DecodeError::other("insufficient counters"));
    }

    Ok((offset, table))
  }
}

#[doc(hidden)]
pub trait TryFromIterator: sealed::Sealed {
  type Item;

  fn try_from_iterator<I: IntoIterator<Item = Self::Item>>(
    iter: I,
  ) -> Result<Self, varing::DecodeError>
  where
    Self: Sized;
}

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

      impl TryFromIterator for [u64; $size] {
        type Item = u64;

        fn try_from_iterator<I: IntoIterator<Item = Self::Item>>(iter: I) -> Result<Self, DecodeError>
        where
          Self: Sized,
        {
          let mut array = core::mem::MaybeUninit::<[u64; $size]>::uninit();
          let mut count = 0;

          let ptr = array.as_mut_ptr() as *mut u64;
          for (i, value) in iter.into_iter().enumerate() {
            if i >= $size {
              return Err(DecodeError::other(concat!("too many counters for [u64; ", stringify!($size), "]")));
            }
            // Safety: We are within bounds.
            unsafe { ptr.add(i).write(value); }
            count += 1;
          }

          if count != $size {
            return Err(DecodeError::other(concat!("insufficient counters for [u64; ", stringify!($size), "]")));
          }

          // Safety: All elements have been initialized.
          Ok(unsafe { array.assume_init() })
        }
      }
    )+
  };
}

impl_d4c4_storage_for_array! {
  8, 16, 32, 64, 128, 256, 512,
}

/// A wrapper to pretty-print the table as a list of 16 4-bit counters per slot.
///
/// Each `u64` word contains 16 packed 4-bit counters. This debug view
/// unpacks them for human-readable output.
pub(crate) struct TableViewD4C4<'a>(pub(crate) &'a [u64]);

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

    ds.field("id", &self.slot_idx)
      .field("counters", &Counters(self.word))
      .finish()
  }
}

#[cfg(any(feature = "std", feature = "alloc"))]
const _: () = {
  impl sealed::Sealed for std::boxed::Box<[u64]> {}

  impl D4C4Storage for std::boxed::Box<[u64]> {}

  impl TryFromIterator for std::boxed::Box<[u64]> {
    type Item = u64;

    #[cfg_attr(not(tarpaulin), inline(always))]
    fn try_from_iterator<I: IntoIterator<Item = Self::Item>>(iter: I) -> Result<Self, DecodeError>
    where
      Self: Sized,
    {
      <std::vec::Vec<u64> as TryFromIterator>::try_from_iterator(iter).map(|v| v.into_boxed_slice())
    }
  }

  impl WithCapacity for std::boxed::Box<[u64]> {
    #[cfg_attr(not(tarpaulin), inline(always))]
    fn with_capacity(capacity: usize) -> Self {
      std::vec![0u64; capacity].into_boxed_slice()
    }
  }

  impl sealed::Sealed for std::vec::Vec<u64> {}

  impl TryFromIterator for std::vec::Vec<u64> {
    type Item = u64;

    #[cfg_attr(not(tarpaulin), inline(always))]
    fn try_from_iterator<I: IntoIterator<Item = Self::Item>>(iter: I) -> Result<Self, DecodeError>
    where
      Self: Sized,
    {
      Ok(iter.into_iter().collect())
    }
  }

  impl D4C4Storage for std::vec::Vec<u64> {}

  impl WithCapacity for std::vec::Vec<u64> {
    #[cfg_attr(not(tarpaulin), inline(always))]
    fn with_capacity(capacity: usize) -> Self {
      std::vec![0u64; capacity]
    }
  }
};
