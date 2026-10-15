# Static translated-palm diagnostic summary

These poses are diagnostic only. Original mapper outputs and extra direction refusals are retained; scientific surface acceptance is reported separately.

| Index | Method / callback | Palm XYZ shift (mm) | Shift (mm) | Objective: original → candidate | Tip mean: original → candidate (mm) | Max exact cap excess (m) | Extra direction guard |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 8 | free_palm_v1 / 65 | (-0.000003, 0.000001, -0.000001) | 0.000003 | 7.56703342 → 7.40234531 | 5.844763 → 5.829876 | -9.99997091e-10 | REFUSED |
| 12 | free_palm_v1 / 65 | (-0.000003, 0.000001, -0.000001) | 0.000003 | 8.03769913 → 7.90028027 | 5.866399 → 5.851859 | -9.99997094e-10 | REFUSED |
| 16 | free_palm_v1 / 64 | (-0.000001, -0.000001, 0.000000) | 0.000001 | 10.1948869 → 10.1688614 | 6.646875 → 6.645969 | -9.99999764e-10 | REFUSED |
| 20 | free_palm_v1 / 19 | (-0.912801, -0.289991, -4.797014) | 4.891691 | 12.4496473 → 11.3740391 | 8.189192 → 7.240657 | -9.99999432e-10 | REFUSED |
| 24 | free_palm_v1 / 17 | (-1.060839, -3.561077, -0.074697) | 3.716480 | 8.24125267 → 7.67544516 | 6.609030 → 6.193915 | -9.99996148e-10 | REFUSED |
| 28 | free_palm_v1 / 17 | (-0.897766, -3.660960, -0.095213) | 3.770633 | 8.41083154 → 7.75948393 | 6.608610 → 6.181258 | -9.99999851e-10 | REFUSED |

A nonpositive cap excess means all five literal original-cap constraints pass. The table does not add any numerical tolerance.

- 8: middle:direction_sse_increased; ring:direction_sse_increased
- 12: index:direction_sse_increased; ring:direction_sse_increased
- 16: ring:direction_sse_increased; little:direction_sse_increased; thumb:original_valid_sites_invalidated; thumb:direction_sse_increased
- 20: middle:original_valid_sites_invalidated; middle:direction_sse_increased; ring:original_valid_sites_invalidated; ring:direction_sse_increased; little:direction_sse_increased; thumb:original_valid_sites_invalidated; thumb:direction_sse_increased
- 24: index:original_valid_sites_invalidated; index:direction_sse_increased; middle:original_valid_sites_invalidated; middle:direction_sse_increased; ring:direction_sse_increased; little:original_valid_sites_invalidated; little:direction_sse_increased; thumb:original_valid_sites_invalidated; thumb:direction_sse_increased
- 28: middle:original_valid_sites_invalidated; middle:direction_sse_increased; ring:original_valid_sites_invalidated; ring:direction_sse_increased; little:original_valid_sites_invalidated; little:direction_sse_increased; thumb:original_valid_sites_invalidated; thumb:direction_sse_increased
