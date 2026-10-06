# GLib iterator soundness backport

This is the unmodified `glib` 0.18.5 crates.io source archive, except for the upstream
`VariantStrIter::impl_get` fix and this provenance note. The archive SHA-256 is
`233daaf6e83ae6a12a52055f568f9d7cf4671dabb78ff9560ab6da230ce00ee5`, matching the original
Cargo.lock entry. The upstream MIT licence and notices are retained.

Upstream fix: https://github.com/gtk-rs/gtk-rs-core/pull/1343
Advisory: https://rustsec.org/advisories/RUSTSEC-2024-0429.html

The pointer passed to `g_variant_get_child` is now mutable and is passed as `&mut p`. This
backports the upstream fix without changing the GTK3-compatible 0.18 API, dependency versions
or public behaviour. Tauri's GTK3 dependency cannot accept the 0.20 API as a drop-in upgrade.

The version remains 0.18.5 so that provenance and dependency compatibility are accurate.
Version-only advisory scanners may continue to report RUSTSEC-2024-0429; the source-level fix
is documented here rather than hiding the advisory or relabelling the package as 0.20.

Verify the affected iterator tests with optimisations enabled:

```sh
cargo test --manifest-path verification/glib-iterator/Cargo.toml --locked --release
```

Remove this patch when the desktop stack supports a maintained, upstream-fixed GLib release.
