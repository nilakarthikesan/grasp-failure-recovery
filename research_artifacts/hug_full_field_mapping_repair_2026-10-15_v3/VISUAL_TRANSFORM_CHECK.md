The v3 visual arithmetic wrapper calls the original frozen `visual_mesh` exactly
once and returns its triangle, normal and metadata objects unchanged. It does
not substitute reference geometry or relax any eligibility or mapping rule.

Every vertex coordinate in both original 3-by-3 transforms is compared with
`numpy.einsum(optimize=False)` and scalar `math.fsum` of three products. Original
triangles are checked against all reference face coordinates. Signed volumes
are independently summed over every face, with unchanged nonzero sign required.
The fixed limits are 1e-14 metres for coordinate differences and 1e-14 cubic
metres for signed-volume differences. These limits check arithmetic only.

Original/replay warnings remain visible in the receipt. Nonfinite inputs or
outputs, changed operands, comparison failures or work domains above 250,000
vertices or 500,000 faces refuse rather than sampling or substituting outputs.
Each case saves exclusive operand/output NPZ and JSON files, including partial
evidence on failure. The returned outer path/hash binding belongs in the case
source receipt and is indexed by the parent result. No extra model, forward,
distance, inference, fitting or task-outcome operation is performed.
