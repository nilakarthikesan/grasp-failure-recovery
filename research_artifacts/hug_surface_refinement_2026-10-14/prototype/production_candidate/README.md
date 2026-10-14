# Experimental static surface-refinement API derivative

This directory contains an API validation derivative of the two immutable
modules used for the saved-grasp development experiment. It adds input checks,
defensive callback handling and explicit scope documentation. It changes no
objective component, default weight, solver setting, existing engineering
guard, acceptance rule, controller or generator.

The eventual project paths are
`src/grasp_failure_prediction/evaluation/surface_refinement.py` and
`src/grasp_failure_prediction/evaluation/surface_refinement_guard.py`. The
tests import those exact package paths. These APIs are explicit static research
tools: the default mapper and `grasp-eval` do not call them automatically.

## API changes

- Numeric configuration fields and guard metrics reject booleans and unusable
  scalar types. A supplied mutable guard-alpha list is copied into a tuple.
- Site selection accepts array-like inputs and rejects empty, nonvector,
  negative-distance and invalid-area domains. Area metadata assumes caller
  inputs are square millimetres.
- Source/raw q must be inside actual supplied bounds; the API does not clip an
  infeasible source target. This prevents a formerly allowed subnanoradian excess
  from later reaching an infeasible SciPy initial point.
- Fixed reference arrays are copied, and each callback receives a read-only
  copy of q. Negative callback distances are refused because this API requires
  unsigned distances. Callback errors remain explicit refusals.
- Guard assessments are copied recursively so a reused callback dictionary
  cannot rewrite baseline or earlier-alpha audit history.
- Documentation permits wrist or finger correction groups while requiring
  the declared palm anchor and q-dependent carrier compensation to be retained.
  The unused `asdict` import is removed.

None of these checks defines scientific mapping acceptance. A callback must
still supply the same fixed reference arrays, actual geometry, query sites,
weights, axes and units on every call. The pure API cannot independently certify
that external callback implementation.

## Verification and limitations

`api_derivative_verification.json` records 70 passing manufactured pytest cases
and 11 exact frozen/API output-and-diagnostic parity comparisons. It does not
claim that the derivative reran all six saved geometries. The separate
`api_guard_parity.json` replays recorded engineering assessments for all six
saved cases and reproduces targets and all selection diagnostics exactly;
it does not recompute or validate their geometry.

A manufactured test deliberately demonstrates that default local optimization
can report lower objective while remaining on the wrong side of a native
closest-point relationship. A separate manufactured weight of 3 tests
representability of negative-side and away-facing native targets. That test
weight changes neither API defaults nor the saved-grasp experiment.

Existing candidate geometry results continue to belong to the immutable frozen
modules. Candidate static acceptance still depends on the separate full audit;
the boolean objective-improvement flag and engineering fingertip guard are not
scientific preservation certificates.

Reproduce the manufactured check using the established project environment:

```sh
python production_candidate/check_api_derivative.py
```

The script temporarily loads the derivative under its eventual package names
and runs tests without a pytest cache. It parses/compiles source in memory and
checks input/source hashes before and after. No inference, physics steps, eval
records, labels or training occur.
