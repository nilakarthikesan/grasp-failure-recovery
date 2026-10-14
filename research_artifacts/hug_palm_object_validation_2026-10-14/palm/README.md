# Existing HUG wrist/palm diagnostic

14 October 2026. This is a static diagnostic of the same 32 existing native
HUG poses and original nominal Shadow commands in the prior saved-proposal
bundle: cube, apple, milk box and mug; two observation groups and four seeds
per object. All sources share one MANO shape. These reused development families
are neither newly collected evaluation data nor an independent holdout.

The software anchor is reproduced: the saved MANO wrist landmark and Shadow
palm origin agree within 1.429e-16 m. The actual Shadow `wrist` body is 34 mm
away, exactly as the URDF's `WRJ1` origin specifies. That body difference is an
expected model relationship, not a measured translation bug. Origin agreement
does not establish anatomical wrist/palm alignment.

## Anatomy proxies and their limits

The script reads the actual pinned `manotorch` reorder and HUG wrist centering.
The 21-landmark order is wrist followed by thumb, index, middle, ring and little
chains. Nonthumb MCP candidates are indices 5, 9, 13 and 17, taken from the first
joint of the respective MANO chains. Shadow candidates are the actual `FFJ3`,
`MFJ3`, `RFJ3` and `LFJ3` joint anchors. Their co-located abduction/flexion joints
and the articulated little metacarpal are recorded from the actual URDF. Native
thumb index 1 is compared separately with `THJ5` as a thumb CMC/base proxy; it
is not relabeled a nonthumb MCP.

These candidates share anatomical names and chain roles across different
models. There is no ground-truth claim that they identify the same anatomical
surface location. No anatomical pair, model frame or replacement rotation has
been calibrated.

Native wrist to nonthumb MCP centroid is 95.351 mm. Shadow palm origin to its
candidate MCP centroid is 93.889 mm, while the actual Shadow wrist body to that
centroid is 127.866 mm. Native index-to-little MCP span is 73.136 mm; Shadow's
candidate span is 66.517 mm. Naming the robot `wrist` body therefore does not
by itself establish that it is the correct counterpart to the native wrist.

We construct geometry directions explicitly: longitudinal is anchor to
nonthumb MCP centroid; transverse is the little-to-index MCP span after
projection perpendicular to longitudinal; their cross product is a signed
normal. We make no palmar/dorsal assignment. Mean native versus Shadow direction
angles are 5.385°, 6.441° and 7.778°, respectively. The longitudinal angle
ranges 5.055–9.515°. These are standalone proxy geometry comparisons, not an
identified frame correction. Shadow's stored wrist angles vary across cases.

The report also describes palm shape bounds. Native palm triangles are assigned
by vertex-majority grouped MANO skinning, with grouped mean weights breaking
ties. This open subset includes assigned wrist-cap/mixed seam triangles and
their boundary vertices. Its average bounding extents in its own proxy frame
are 94.518, 131.746 and 73.260 mm for transverse, longitudinal and signed-normal
directions. Actual eight Shadow palm collision boxes have corresponding average
extents 82.466, 94.099 and 36.066 mm. These subset-dependent bounds describe
different geometry representations; they are not equal anatomical tissue
regions, thickness measurements, volumes or contact-area metrics.

## Could a common placement translation explain the point errors?

The script defines residuals as `r = Shadow - native` in the already implemented
native operator/fixed-URDF base directions. It computes their algebraic mean
`rbar` and differential residuals `r - rbar`. A hypothetical Shadow translation
would be `-rbar`, but no geometry, model pose, target, controller or adapter is
modified. The squared-error identity is checked for each case.

One shared translation over all 160 fingertips has a hypothetical operator
shift of (+1.813, +0.580, +2.383) mm. It removes 14.61% of squared point error:
mean separation changes 6.428 to 6.156 mm and RMS 7.978 to 7.372 mm in the
residual arrays. Removing each source's own mean removes 17.35%, leaving mean
5.964 mm and RMS 7.253 mm. Per-source mean residual vectors differ from the
shared mean by 1.093 mm on average and up to 3.159 mm.

Even after each source's fingertip mean is removed, mean differential
separations remain thumb 11.785, index 3.224, middle 2.689, ring 2.333 and little
9.789 mm; the worst differential tip is 13.830 mm. The raw fingertip means are
14.547, 1.908, 4.601, 2.884 and 8.198 mm, respectively. Thus the algebraic mean
improves some fingers while worsening others. Object-level mean differential
tip separations after each source mean is removed are apple 5.477, milk box
6.355, mug 6.190 and cube 5.835 mm.

For all ten task points (five tips and five middle-link points), one common
translation explains only 4.29% of squared error; separate per-source means
explain 5.19%. The remaining mean/RMS are 6.511/7.652 mm after per-source means.
These ratios are analytic squared-error components, not mapping accuracy or
scientific pass rates.

Candidate nonthumb MCP points have different residual means: their single
shared hypothetical shift is (+9.758, +2.691, +1.881) mm, 8.237 mm away from the
fingertip-derived vector. Removing it algebraically from fingertip residuals
increases mean fingertip error to 10.371 mm. Removing per-source MCP means still
leaves mean MCP differential separation 5.456 mm and RMS 5.529 mm. The
fingertip-derived and all-ten-point-derived vectors differ by 3.644 mm as well.
Different proxy choices therefore yield different putative offsets.

The diagnostic does not support one translation offset as a complete
explanation of the residuals. It does not identify which remaining error comes
from morphology, joint policy, a rotation assumption, or imperfections in the
declared proxies. It does not justify applying any measured mean as calibration.
Surface/contact fidelity remains unconfirmed, including for poor native grasps.

## Reproduction and preservation

`diagnose_palm_anchor.py` installs rejecting guards for `mj_step`, `mj_step1`
and `mj_step2` before project imports or model loading. Three deliberately
invalid guard calls are rejected in self-tests; the diagnostic attempts zero
integration calls. Each saved scene is forwarded at time zero after installing
its existing original nominal targets. Initial object qpos is unchanged, and
the 10 target links reproduce prior saved FK exactly to the report's precision.

`palm_anchor_report.json` records every input hash, actual URDF joint definition,
32 case results, proxy definitions, residual decompositions and limitations.
All 108 Python files under production `src` and `scripts` match their before
hashes after the run. All 199 recorded input files match before/after hashes.
The local MANO asset, HUG wrapper and Shadow URDF also match their hashes in the
earlier confirmation; one unique saved native shape is checked.
No learned generator, retargeting optimizer, calibration, physics rollout, new
grasp, predictor fit or production change occurs. Licensed MANO weights are
read locally; the asset and raw weights are never copied into this bundle.

To reproduce, copy this script into
`/private/tmp/hug-palm-anchor-2026-10-14/`, preserve the source/input paths in
the report, and run `sh reproduce.sh` there. The runtime uses the existing
project Python environment and dependency directories. The output-directory
assertion confines new diagnostic writes to that directory. `diagnostic_stdout.json`
is a compact summary of the saved report, not an additional experiment.

Any proposed calibration would need a separately justified correspondence,
versioned procedure and separate validation cases. This report supplies static
evidence for that later decision; it does not perform the calibration.
