# HCA2/acetazolamide example input

This directory contains the small, reproducible input bundle used by the
example configuration: human carbonic anhydrase II, acetazolamide, Zn2+, and
explicit OPC water.

* `system_1264.parm7` — Amber topology consumed by OpenMM/PartQMMM.
* `hca2_azm_withwater_run1_combined.pdb` — coordinate frame 0 and box data.

These files are example research inputs, not a general-purpose force-field
distribution. Keep their checksums unchanged when comparing validation logs:

```text
system_1264.parm7  54741ef6612e99e8f4a4ec1991007cf0572ba19efc644e53e1acc78531dae0d1
hca2_azm_withwater_run1_combined.pdb  6b897d1d85dc00409cfdeeea735f5bc18f1410d1089e4f1add8afab39f92a295
```

The same files are retained at the repository root for backwards compatibility
with the original scripts. New work should use these paths through a config
file rather than hard-coding filenames.
