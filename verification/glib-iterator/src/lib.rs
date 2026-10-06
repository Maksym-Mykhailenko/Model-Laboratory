//! Optimised regression coverage for the GTK3-compatible GLib soundness backport.

#[cfg(test)]
mod tests {
    use glib::{variant::ToVariant, Variant};

    fn strings() -> Variant {
        Variant::array_from_iter::<String>(["", "one", "два", "three", "four", "five"]
            .into_iter().map(|value| value.to_variant()))
    }

    #[test]
    fn forward_and_reverse_iteration_borrow_valid_strings() {
        let values = strings();
        assert_eq!(values.array_iter_str().unwrap().collect::<Vec<_>>(),
                   ["", "one", "два", "three", "four", "five"]);
        assert_eq!(values.array_iter_str().unwrap().rev().collect::<Vec<_>>(),
                   ["five", "four", "three", "два", "one", ""]);
    }

    #[test]
    fn mixed_iteration_respects_head_tail_and_exhaustion() {
        let values = strings();
        let mut iter = values.array_iter_str().unwrap();
        assert_eq!(iter.next(), Some(""));
        assert_eq!(iter.next_back(), Some("five"));
        assert_eq!(iter.nth(1), Some("два"));
        assert_eq!(iter.nth_back(0), Some("four"));
        assert_eq!(iter.len(), 1);
        assert_eq!(iter.last(), Some("three"));
    }

    #[test]
    fn last_and_out_of_bounds_advance_are_safe() {
        let values = strings();
        assert_eq!(values.array_iter_str().unwrap().last(), Some("five"));
        let mut iter = values.array_iter_str().unwrap();
        assert_eq!(iter.nth(6), None);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
        let empty = Variant::array_from_iter::<String>([]);
        assert_eq!(empty.array_iter_str().unwrap().last(), None);
    }
}
