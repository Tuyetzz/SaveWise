# Design: a journey report a judge can trust

**Date:** 2026-08-08
**Status:** Approved
**Builds on:** `PRD.md` v3, `2026-08-08-rescue-vision-design.md`

## Context

The rover drives itself and this subsystem observes. Its output is a log of who
was seen. The demo is 3–5 minutes in front of hackathon judges, staged with
**standing people only** — nobody can lie on the venue floor.

Detection is not the differentiator; every team will have YOLO drawing boxes.
The differentiator is answering **"how many people did you find?"** correctly,
and being able to show it.

That question is harder than it looks, and the current code gets it wrong.

### The measured defect

`SightingRecorder.finalise_absent()` closes a sighting the moment a track is
missing from a single frame. But detection legitimately drops frames — measured
at ~23% under camera-module noise conditions (PRD §6.6). Each miss closes the
sighting; the next hit opens a new one. ByteTrack correctly keeps the same track
ID across the gap; the recorder discards that and starts over.

Simulated, one person standing in view for 60 frames:

| Frame drop rate | Sightings logged |
|---|---|
| 0% | 1.0 ✓ |
| 10% | 5.0 |
| **23%** (measured camera rate) | **9.4** |
| 40% | 13.6 |

One person in front of the rover would be reported as ~9 survivors. That makes
the log worthless, and it is invisible unless you go looking.

## Approach

Three changes, strictly ordered. Each is independently shippable; if time runs
out, stopping after any one leaves a coherent result.

### 1. Sighting continuity — everything rests on this

Add `sighting_gap_s: float = 1.5` to `Config`. `finalise_absent()` closes a
sighting only when it has not been seen for longer than that, rather than on the
first missing frame.

**Why 1.5 s specifically.** ByteTrack's default `track_buffer` is 30 frames
(~2–3 s at our frame rates). A track reappearing inside that window still
carries the *same ID*, so the returning detection rejoins its existing sighting.
A grace period longer than ByteTrack's buffer would be actively harmful — the
returning person gets a new ID, and merging on a stale record would fuse two
different people.

`last_seen_s` continues to record the last frame the person was actually seen,
not when the grace expired, so `duration_s` stays truthful.

### 2. Per-person colour

Colour is keyed on `sighting_id`, never on the person's index in the current
frame — the rule is *colour follows the entity, never its rank*. A filter or a
departure must not repaint the survivors. This is only stable because of change
1; without it a person's colour would change ~9 times in six seconds.

Slots assigned in fixed order from the validated categorical palette:

| P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 |
|---|---|---|---|---|---|---|---|
| `#2a78d6` | `#eb6834` | `#1baf7a` | `#eda100` | `#e87ba4` | `#4a3aa7` | `#e34948` | `#008300` |

The first three are the deliberate choice for the 1–3 person case the PRD
assumes. Validated all-pairs (`validate_palette.js --pairs all --mode light`):
CVD ΔE 9.2, normal-vision ΔE 24.0, all checks pass. The same run at six slots
**hard-fails** the normal-vision floor (magenta↔orange ΔE 12.9, below 15) —
meaning full-colour-vision viewers cannot reliably separate them. Colour alone
therefore cannot carry identity past three people.

Two consequences:

- **Label carries a person number**: `P2 · confidence_score = 0.89`. Identity is
  never colour-alone, so it survives 4+ people and a colourblind viewer (~1 in
  12 men). It also makes the report cross-referenceable against the video.
- **Dark outline beneath the coloured stroke.** The palette's contrast figures
  are measured against a controlled `#fcfcfb` chart surface. Ours is whatever
  the camera sees — a blue box on a blue door vanishes. The outline restores
  legibility on any background.

**Unconfirmed tracks stay grey**, and this is meaningful rather than incidental:
grey means "not yet counted", colour means "this person is in the report".

### 3. `output/report.html`

Self-contained: images inlined as base64, no server, opens offline, survives as
a leave-behind. Header gives sweep duration, people found, and the coverage
caveat. One card per person — colour swatch matching the video, best photo,
first seen, duration, peak confidence, bearing, distance when trustworthy.

## Explicitly out of scope

- **Posture / triage.** Cut. Aspect ratio is a crude proxy that misfires on
  crouching, occlusion, and foreshortening, and cannot be staged at the venue.
- **Maps, positions, re-identification across the journey.** Without odometry
  these would be fiction. The report says *when* and *at what angle*, and states
  plainly that it does not know *where*.
- **Merging re-acquired tracks across long gaps.** Beyond ByteTrack's buffer a
  returning person is honestly a new sighting. A wrong merge is invisible; two
  adjacent entries are legible.

## Verification

1. **The defect, as a regression test.** One person, 60 frames, at 0/10/23/40%
   simulated dropout → exactly 1 sighting in every case. This test is the
   deliverable's credibility and must be written first.
2. Colour stability: a person whose detection drops and returns inside the grace
   window keeps their original colour and sighting ID.
3. Colour follows the entity: when P1 leaves and P2 remains, P2 keeps orange.
4. `.\demo.bat clip` → 3 sightings, 3 distinct colours, `report.html` with 3
   cards whose swatches match the video.
5. Full suite green; no test asserts the old finalise-immediately behaviour.

## The demo claim this buys

*"A naive log reports this person nine times. We measured that, then fixed it.
This report says three, and there were three."*
