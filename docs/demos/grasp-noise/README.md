# Recorded grasp noise demo

The [complete website](index.html) is one self-contained HTML file. It embeds
five recorded trajectories, both success/failure replay videos, interactive
charts and the 50-attempt pilot summary. The [preview](preview.jpg) shows the
first lift command for an index-finger joint.

## Open the website

Download `index.html` and open it in a browser. On GitHub, use **Download raw
file** on the HTML file page: GitHub normally shows its source rather than
executing it. You can also serve the folder from the repository root:

```bash
python3 -m http.server 8767 --bind 127.0.0.1 --directory docs/demos/grasp-noise
```

Open <http://127.0.0.1:8767/> on that computer. The page needs no network
requests or simulation dependencies. The address is a local preview, not a
publicly hosted site. If port 8767 is occupied, use another port in both the
command and address.

## Try the worked example

1. Select **No added noise** or **±0.6 rad**. These use the same starting HUG
   proposal and paired noise direction.
2. Select **FFJ1** and **First lift interval**. At 3.00 seconds, the stronger
   condition changes the command from about 42.1° to 8.1° by adding −34.0°.
3. Select **THJ3** to see a thumb target clipped to its allowed joint range.
4. Move the time slider to compare requested/applied offsets, command endpoints,
   measured motion and object lift.
5. Watch both saved videos below the charts to compare the final outcomes.

The controls inspect existing data; they do not run new physics. Commands are
endpoints of 40 ms control intervals, with interpolation between endpoints.
Measured angles are logged at the interval end. Each joint's requested bias
stays fixed during closing, lifting, holding and scoring; wrist offsets stay
zero. Joint limits determine the applied command offset.

## Included records and limitations

Recorded on **9 October 2026**, packaged on **10 October 2026**:

- Five paired attempts for `pilot_cube_hug_seed2`, at bounds 0, ±0.05, ±0.15,
  ±0.3 and ±0.6 radians, each with 163 recorded samples. Charts include two
  selected joints and object lift.
- Two MP4 state replays: `batch/episode_000006` succeeded; the ±0.6-radian
  `stress_batch/episode_000005` failed the required lift and hold.
- Aggregate counts for all 50 trials: 40 successes, 10 failures and zero errors.
- [Source manifest](source_manifest.json): original episode paths and file
  hashes, runtime reference, collector hash and packaged artifact hashes.

The experiment uses one cube and one initial RGB-D observation. Strong offsets
are stress conditions, not calibrated hardware-error distributions. Binary
labels describe completed outcomes, not the onset of an impending failure.
No failure predictor or train/validation/test split has been built.

The embedded selected records and rendered videos are intentionally included
for this demo. Full raw runs remain in the ignored local
`runs/noisy_pilot_2026-10-09/` folder. They are not required to open this page,
and the source paths shown on the site identify those original files rather
than downloadable files bundled here.

The [Step 1 walkthrough](../../HUG_STEP_1_WALKTHROUGH.md) explains the dataset
choice, complete scoring rule and remaining simulation integration work.

## Credits

Initial grasps use Kevin Yuanbo Wu and coauthors' *Human Universal Grasping*
(2026): [project](https://grasping.io/), [code](https://github.com/KevinyWu/hug).

The videos render the Shadow Hand assets distributed through
[DexSuite dex-urdf](https://github.com/dexsuite/dex-urdf), pinned at
`7304c7fb59214dab870eca02cf26f76e944e12df`. The hand asset license credits
**Shadow Robot Company Ltd, copyright 2022, Apache-2.0**, and records mesh/URDF
modifications by dex_urdf authors. The dex-urdf root license is **MIT,
copyright 2023–2024 DexSuite**. Full copies of both notices are embedded in
the website's expandable credits section, preserving them when the HTML is
shared alone.
