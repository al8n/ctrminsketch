// use crate::table::{D4, Depth, Table};

// use std::boxed::Box;

// const RESET_MASK: u64 = 0x7777_7777_7777_7777;

// /// Counter Min Sketch with 4-bit counters.
// pub struct Cms4<D: Depth = D4> {
//   width: u64,
//   seeds: D::Seed,
//   storage: Box<[u64]>,
// }

// impl<D: Depth> Cms4<D> {
//   // /// Increments the counter for the given hash.
//   // pub fn increment(&mut self, h: u64) {
//   //   // Implementation goes here
//   // }

//   /// Returns the mask for indexing.
//   #[cfg_attr(not(feature = "tarpaulin"), inline(always))]
//   pub const fn mask(&self) -> u64 {
//     self.width - 1
//   }

//   // /// reset (aging): all counters >>= 1, and mask off high bits
//   // #[cfg_attr(not(feature = "tarpaulin"), inline(always))]
//   // pub fn reset(&mut self) {
//   //   for w in &mut self.storage {
//   //     *w = (*w >> 1) & RESET_MASK;
//   //   }
//   // }

//   #[cfg_attr(not(feature = "tarpaulin"), inline(always))]
//   fn index(&self, h: u64, depth: u8) -> usize {

//   }
// }

// impl Table for Cms4<D4> {
//   type Counter = u8;

//   #[cfg_attr(not(feature = "tarpaulin"), inline(always))]
//   fn increment(&mut self, h: u64) {
//     // Implementation goes here
//   }

//   #[cfg_attr(not(feature = "tarpaulin"), inline(always))]
//   fn estimate(&self, h: u64) -> Self::Counter {
//     // Implementation goes here
//     0
//   }

//   #[cfg_attr(not(feature = "tarpaulin"), inline(always))]
//   fn reset(&mut self) {
//     for w in &mut self.storage {
//       *w = (*w >> 1) & RESET_MASK;
//     }
//   }

//   #[cfg_attr(not(feature = "tarpaulin"), inline(always))]
//   fn clear(&mut self) {
//     for w in &mut self.storage {
//       *w = 0;
//     }
//   }
// }
