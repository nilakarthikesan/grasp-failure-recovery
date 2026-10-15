# Explicit translation-seed coverage supplement

This one post-hoc supplement preserves all original selection results. It uses three existing grasps, no new fitting, and no physical episodes. All listed poses remain refused by the extra direction selector and are not promoted.

| Case | Method / seed / proposal / callback | Requested shift norm (mm) | Actual shift norm (mm) | Objective original → candidate | Tip mean original → candidate (mm) | Largest signed cap excess (m) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | fine_translation_v2 / 51 / 52 / 53 | 0.156250 | 0.063565 | 7.56703342 → 7.57727345 | 5.844763 → 5.838904 | -1.00000002e-09 |
| 12 | free_palm_v1 / 21 / 22 / 23 | 5.000000 | 1.994976 | 8.03769913 → 7.9663163 | 5.866399 → 5.708878 | -9.99998658e-10 |
| 16 | fine_translation_v2 / 48 / 49 / 50 | 0.156250 | 0.031372 | 10.1948869 → 10.2015555 | 6.646875 → 6.645773 | -9.99999986e-10 |

Seed indices are zero based; proposal and callback sequences are one based. Nonpositive cap excess means every literal original cap passes, without an acceptance allowance.

- 8: index:original_valid_sites_invalidated; index:direction_sse_increased

- 12: index:original_valid_sites_invalidated; index:direction_sse_increased; middle:original_valid_sites_invalidated; middle:direction_sse_increased; little:direction_sse_increased

- 16: middle:original_valid_sites_invalidated; middle:direction_sse_increased; ring:direction_sse_increased; little:direction_sse_increased; thumb:direction_sse_increased
