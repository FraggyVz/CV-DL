# Recording and Frame-Rate Comparison

Use [README](../README.md) for installation, commands and current project status.
The [tactic guide](tactic-definitions.md) covers labelling after preparation.

## Current capture setup

Use **2560×1440 output, 60 fps, constant 2× demo playback**. Record manually in OBS.
The processor restores normal game speed, crops the minimap and exports aligned
5/10/30 fps videos. It does not operate CS2, control OBS or identify match endings.

Obsolete capture assets were cleared on 14 October. Calibrate once for the current
layout, then reuse the saved profile only when resolution and radar layout match.

## OBS and CS2

1. Use a separate OBS profile/scene collection for demos.
2. Capture the full game frame. Set base and output resolution to 2560×1440,
   common FPS to 60, with no output scaling, stretch or overlays.
3. In Simple output mode, use High Quality, Medium File Size, hardware NVENC H.264
   and Matroska MKV. Disable automatic remuxing to avoid a duplicate full-screen MP4.
4. Keep radar orientation, zoom, position and team colours fixed throughout.
   Move the spectator camera away if its white cone covers the map.
5. Keep demo playback at 2×; pauses, seeks and speed changes invalidate a single
   speed multiplier. Check game rendering FPS as well as OBS output FPS.

MKV tolerates interrupted recording better than ordinary MP4; see the
[OBS recording guide](https://obsproject.com/kb/standard-recording-output-guide).
A nominal 60 fps file can contain repeated frames. Timestamp correction cannot
recover frames that CS2 never rendered.

## Radar preset and restoring gameplay

Before changing settings, save your current radar values in your personal
`autoexec.cfg` or a separate gameplay preset. Query command names without values
in the CS2 console if necessary. Personal game settings stay outside this repository.

Save the following as `stratscout_recording.cfg` in CS2's `game/csgo/cfg` folder,
beside your autoexec. Load it manually with `exec stratscout_recording.cfg`;
do not load the recording preset automatically at launch.

```text
cl_hud_radar_scale 1.3
cl_hud_radar_background_alpha 1
cl_hud_radar_blur_background 0
cl_hud_radar_map_additive 0
cl_radar_square_always 1
cl_radar_square_when_spectating 1
cl_radar_square_with_scoreboard 0
cl_radar_rotate 0
cl_radar_always_centered 0
cl_radar_scale_dynamic 0
cl_radar_scale 0.25
cl_radar_scale_alternate 0.25
cl_radar_icon_scale_min 0.6
cl_teammate_colors_show 0
cl_radar_show_all_players_when_spectating 1
```

Check these settings in your installed CS2 build. Confirm a fixed, opaque radar
with distinct team colours. Close the console, scoreboard and replay UI before
capturing. After recording, use `exec autoexec.cfg` to restore your saved gameplay
values. Restarting CS2 alone is not a reliable reset.

## Prompt-based processing

Wait until OBS has stopped and finished writing the recording. Put the completed
file directly in `data/raw/inbox/`, then run from the repository root:

```powershell
.\process-recordings.ps1
```

The launcher uses the project's Python environment. For each new video it asks:

```text
Demo playback speed [1 = normal, 2 = double; Enter = 2]:
Start time [Enter = 0:00]:
End time [Enter = end of video]:
```

Press Enter for the current 2× setup. Choose start/end by watching the raw video:
use seconds, `MM:SS` or `HH:MM:SS`, including fractional seconds if needed.
These are raw-player times, not the demo clock or restored playback time.
For example, start `0` and end `23:51` retain 1431 recorded seconds and produce
approximately 2862 seconds at normal speed. The example is not a recommended cut.

Invalid answers are explained and asked again. Type `s` at any prompt to skip
that file. Ctrl+C or closed terminal input stops the batch. No per-video settings
JSON is required; answers are retained automatically in output metadata.

## Crop selection

On first use, select the entire minimap. Odd dimensions are made even by extending
an edge by one pixel where possible (or trimming one pixel if the whole source
dimension is odd). Pixels are never resized. Enter confirms the rectangle, then
Enter confirms the adjusted native-resolution preview; Escape cancels the preview.
The terminal prints the saved profile path after confirmation.
The script saves `configs/recording-video.json` automatically and
reuses it for the same resolution and radar layout.

The initial preview is ten raw seconds after the selected start, or halfway through
a short retained clip. Use `--at SECONDS` to change that offset. Dimension mismatches
are rejected; same-resolution HUD changes cannot be recognised automatically.
For a changed layout, pass a fresh `--profile configs/new-layout.json` and select
the crop again. No historical crop coordinates are supplied.
Explicit `--crop` coordinates and existing profiles remain strictly validated;
automatic edge adjustment applies only to a newly dragged selection.

## Outputs and recovery

Each source gets `data/processed/recordings/<match-name>/`, named from its video
filename (for example, `furia-falcons-mirage-cologne-major-2026/`), containing:

- Three native-resolution MP4s at 5, 10 and 30 fps, restored to normal game speed.
- A manifest with source hash, match identity, crop, speed, trim, counts and versions.
- Encoding/validation logs and `comparison/report.html`, previews and timestamped stills.

The comparison samples up to three spread-out ten-second windows in game time.
It does not identify deaths, bomb events or rounds. Inspect the preview to check
the crop, start/end, normal playback speed, frame holding and visible artefacts.

FFmpeg decodes/crops once. Speed correction precedes a canonical 30 fps stream;
10/5 fps select every third/sixth frame and are independently encoded. Settings are
libx264, CRF 18, medium, yuv420p, no audio and MP4 fast-start. Every new output is
fully decoded and checked before the bundle is marked complete.

Completed sources are skipped before prompts using their saved settings. Reruns
find bundles by the full source hash in their manifests, so readable names do not
break duplicate detection. Existing hash-only folders remain recognised. Different
source files with the same cleaned name get a short hash suffix to prevent collisions.
Keep each manifest with its videos; the folder name is for display, not dataset identity.
Reruns
check full source hashes, crop compatibility and output sizes/existence; they do
not detect same-size output corruption. A missing comparison can be generated
without re-encoding. Incomplete or incompatible destinations are refused. Use a
new `--output` directory to retry or change a cut; existing results are not overwritten.

Only direct MKV/MP4/MOV/AVI children of the inbox are scanned. Subfolders and demos
are ignored. One failed file does not block the others; failures produce a non-zero
exit status. The script is not a watcher and must not run on an active OBS recording.

## Acceptance checks

Before using a new recording for labelling:

1. Confirm OBS has stopped and verify resolution, nominal FPS and constant replay speed.
   Inspect movement for stalls or repeated frames; header FPS alone is insufficient.
2. Review the raw-time cut and crop. Keep gameplay through the actual round end, without
   unnecessary menus or victory-screen tails. Never modify the master to trim exports.
3. Confirm the manifest and comparison report are complete. Check all three exports
   retain native crop dimensions and their duration equals retained raw time × replay speed.
4. Inspect opening, middle and ending footage, including deaths, close icons, player
   numbers and bomb-carrier overlap. Check the normal-speed playback visually.
5. Keep the source hash, match identity and manifest with the exports. A batch rerun should
   skip complete bundles; it is not a full corruption check for existing outputs.

The processor fully decodes new exports before marking them complete. Passing these
checks establishes media preparation, not CV accuracy or the best training frame rate.
Those require reviewed labels and match-separated evaluation. Past measurements are
preserved in [archived processing notes](archive/processing-validation.md).

## Sharing processed matches

ZIP the complete match folders and share the archive through shared storage.
Teammates should extract the ZIP, then open each match's `comparison/report.html`
or the `minimap-5fps.mp4`, `minimap-10fps.mp4` and `minimap-30fps.mp4` files directly.
Keep relative folders intact so report videos and stills load correctly. Raw masters
are not needed for viewing these exports. Manifests retain provenance and match IDs.
Generated media and archives under `data/processed/` stay outside Git; share the
code through Git and the archive separately. These videos are not labelled datasets.

## Individual commands

These reusable commands remain available for explicit processing or debugging:

```powershell
python -m stratscout video-profile --video data/raw/inbox/match.mkv --output configs/recording-video.json --at 60
python -m stratscout prepare-video --video data/raw/inbox/match.mkv --profile configs/recording-video.json --playback-speed 2 --start 0 --duration 120 --fps 5 10 30 --output data/processed/pilot-001
python -m stratscout compare-fps --input data/processed/pilot-001 --windows 10:5 50:5 100:5
```

Skip the first command if that profile already exists. The standalone preparation
command defaults to speed 1, so explicitly pass `--playback-speed 2`. Start/duration
and comparison windows use raw-video seconds. A five-second window plays over ten
seconds at normal speed. Source time is `start + output_elapsed / playback_speed`.

For unattended whole-file batch processing, use `--non-interactive` with an existing
crop profile; its speed default remains 2. Use `prepare-video --match-id NAME` for
separate recordings belonging to the same match. Otherwise identity defaults to the
source hash. All variants of one match must stay in one future dataset split.

The processor checks `source_fps / playback_speed >= highest_output_fps`.
Timestamp rounding can shift observations slightly and round the final duration
to complete frames. Keep the source demo separately for later alignment; processing
requires only video. Storage and visual comparisons are not detector accuracy tests.
