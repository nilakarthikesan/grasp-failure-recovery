# Independent reasoning for the exposed-boundary diagnostic

15 October 2026. This is a geometric claim definition, not a new HUG acceptance tolerance.

Let U be the union of all object-compatible, closed Shadow collision solids, including carriers. For a point strictly outside U, distance to U and to its boundary equals the minimum component distance. A nearer union boundary is a component boundary; if a chosen minimum component point were strictly inside another component, the path from the exterior query to that point would encounter the other component earlier, contradicting minimality. Numeric ties/near-boundary cases require explicit handling.

This theorem does not justify using the minimum absolute component signed distance inside U. For two unit balls centered at (-0.5,0,0) and (0.5,0,0), the origin has component-boundary distances 0.5, but the nearest union boundary is at distance sqrt(0.75). Both individual nearest points are buried. A valid interior lower bound is the largest depth into any containing component; a verified union-boundary witness supplies an upper bound. A fixed finite set of rays cannot generally certify the nearest boundary.

An anatomical-part component minimum is also only a lower bound on distance to the portion of the whole-union boundary attributable to that part. For the same balls, a query at (-2,0,0) lies outside the union. The right ball's nearest component point is inside the left ball. Replacing it with the left ball's exposed point changes the anatomical correspondence; it cannot repair the right-part distance query. A missing upper witness must remain missing.

No-strict-interior membership is insufficient to prove exposure: two abutting boxes have a shared face that lies on both component boundaries while being inside the union. An outward finite endpoint probe can also skip a sufficiently thin neighbor. Immediate free ray intervals or another explicit exposure proof can establish an exposed witness, while unresolved seams remain unresolved. A finite ray set may miss a valid escape direction.

These statements concern the geometric boundary of the declared union, including boundaries of cavities. They do not prove exterior-air connectivity, access for a finite-size object, dynamic contact forces, or equivalence to MuJoCo's pairwise contact solver. The native MANO skin/patches remain a separately declared reference; source self-intersection and exposure are not automatically certified by a closed mesh.

Historical per-component surface measurements and acceptance results are preserved. New evidence should report component, union, anatomical-part and numerical assumptions separately. No practical error allowance is inferred from these definitions.
