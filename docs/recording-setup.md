# Recording and Frame-Rate Comparison

## Manual OBS setup

Recording is manual. The Python tools start from an existing video; they do not
connect to OBS. No WebSocket server, password or specially named profile is needed.
You can disable OBS WebSocket if no other application uses it.

1. Create a separate OBS profile and scene collection for recording demos so your
   personal capture settings remain separate. Names are your choice.
2. Add Game Capture targeting CS2, or a working display capture source. Fit the
   full 2560×1440 game frame without source crop, stretch or overlays. Check the
   preview is moving and not black.
3. Settings → Video: base **2560×1440** for your 1440p game, output **1920×1080**,
   common FPS **30**. This intentionally downscales once in OBS to reduce master
   storage. A teammate playing at 1080p can use a 1080p canvas/output. Calibrate
   each layout separately; the same output resolution alone does not prove the
   radar occupies the same pixels. 1440p output is also supported if needed later.
4. Settings → Output → **Simple**: **High Quality, Medium File Size**, **Hardware
   (NVENC, H.264)**, **Matroska Video (.mkv)**. Turn automatic remuxing off to avoid
   an extra full-screen MP4. Keep the match in one recording.

These are capture recommendations, not settings enforced by the processor. MKV
protects recordings against interrupted finalisation; see the
[OBS recording guide](https://obsproject.com/kb/standard-recording-output-guide).

## CS2 capture preset and restoring gameplay

Load the `.dem` yourself. Before changing settings, save your **current playing values**
for every command below in your personal `autoexec.cfg`, or a `gameplay-radar.cfg`
loaded by it. Query command names without values if you still need to collect them.
Keep personal settings outside version control. Restore with `exec autoexec.cfg`
after recording; restarting CS2 alone is not a reliable reset.

This preset is retained from the earlier project recording guide. Recheck it in your
installed CS2 build; the processor does not set game options.
Save this block as `stratscout_recording.cfg` alongside your autoexec in CS2's
`game/csgo/cfg` folder. Do not load the recording preset automatically at launch.

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

Confirm a fixed, opaque, square radar with both teams in distinct colours, without
rotation, dynamic zoom, blur or map blending. Verify T/CT colours from the replay.
Move the spectator camera away if its white cone covers the radar; the processor
does not remove it. Close console, scoreboard and replay UI. Keep normal replay
speed throughout; pauses/seeks break demo/video synchronisation.

If a command is unknown, check console help/menu equivalents before capture. The
teammate's historical capture used different icon/teammate settings; do not mix
presets within a match. Recalibrate after changing HUD size/resolution. The retained
crop `[16,12,508,510]` belongs to the supplied Falcons–FURIA video, not every 1440p
recording. Confirm it on a short new recording before recording a whole match.

## Record manually, then process

Load the demo, apply `exec stratscout_recording.cfg`, position the spectator camera
and start playback. Click **Start Recording** in OBS. Click **Stop Recording** when
finished and wait for OBS to finish writing the MKV before processing it. Restore
gameplay settings with `exec autoexec.cfg` if your normal radar values are saved there.

The preferred route is to save/copy the finished video into `data/raw/inbox/` and
run `.\process-recordings.ps1` from the repository root. It selects a crop on first
use, then runs preparation and comparison automatically. The following separate
commands remain useful for debugging or choosing specific comparison events.

First make a 30-second test. Save the master under a match-specific directory, such
as `data/raw/mirage-match-001/mirage.mkv`. Preserve the source demo beside it when
available, with a short note of the match, map, radar settings, resolution, frame
rate and replay speed. Processing itself requires only video.

Play the master to check motion and icon clarity. Then select its crop and prepare
the three variants (activate the project environment first):

```powershell
python -m stratscout video-profile --video data/raw/mirage-match-001/mirage.mkv --output configs/mirage-match-001.json --at 5
python -m stratscout prepare-video --video data/raw/mirage-match-001/mirage.mkv --profile configs/mirage-match-001.json --match-id mirage-match-001 --fps 5 10 30 --output data/processed/mirage-test-001
python -m stratscout compare-fps --input data/processed/mirage-test-001 --windows 5:10
```

Choose a native crop with even width and height. Confirm the selection with Enter,
inspect the preview and press Enter again to save. Check the exports include the
entire map. A 1440p source alone does not prove the old Falcons–FURIA crop fits.

After the short test passes, record the full match once and run the same processor
with a new output directory. Keep all clips/variants from one match under the same
`--match-id`. Output bundles contain the three MP4s, `manifest.json` and logs;
comparisons add `comparison/report.html`, JSON, previews and stills.

Existing destinations are refused. Cancellation or encoding failure preserves the
master and leaves an incomplete bundle/logs for diagnosis. Retry `prepare-video`
with a new destination; do not overwrite either the source or a completed bundle.

## Batch processing and recovery

Run `.\process-recordings.ps1` (or `python -m stratscout process-recordings` in your
active environment). It creates `data/raw/inbox/` if missing. Only finished video
files directly inside that folder are processed; `.dem` files and subfolders are
ignored. No watcher or OBS connection runs in the background.

The first usable video opens crop selection at ten seconds, or halfway through a
short clip. Override with `--at 60` if the minimap appears later. Select an even-sized
rectangle; Enter confirms selection, then Enter confirms the native preview.
`configs/recording-video.json` is reused thereafter. Keep radar position/size and
output resolution fixed. Dimension mismatches are rejected; same-resolution HUD
changes cannot be detected automatically. Use `--profile configs/new-layout.json`
for a different layout, with a separate input folder if needed.

Each source gets `data/processed/recordings/<first-16-SHA256-characters>/` containing
three MP4s, a manifest, logs and `comparison/report.html`. Full hashes are checked
before reuse, so renamed byte-identical copies are skipped too. The manifest retains
the full source hash as match identity unless a matching source-pinned profile
supplies one. Record one whole match per file; separate recordings of the same match
still need a shared `--match-id` via the individual `prepare-video` interface before
dataset splitting. The batch does not infer match identity from team names.

Automatic previews sample three spread-out windows of up to ten seconds. Short
videos use one window. These are visual spot checks, not event detection or labels.
Use `compare-fps --output NEW_DIRECTORY --windows ...` for selected gameplay events.

Reruns check completed manifests, settings and exported file sizes/existence before
skipping; they do not decode every existing video again or detect same-size file
corruption. If preparation completed but comparison never started, rerunning adds
the comparison without encoding again. An incomplete existing bundle/comparison,
missing output or changed crop is refused. Keep it for diagnosis and retry with
`--output data/processed/recordings-retry-001`; no masters or prior results are deleted.
One failed file does not block other files; any failure gives a non-zero exit status.
Ctrl+C stops the batch and preserves diagnostic outputs. Only process files after
OBS has stopped writing them; the source hash/stat checks are not an OBS lock.

## Prepare existing footage

The full-screen source stays unchanged. From the repository root:

```powershell
python -m stratscout prepare-video --video "data/raw/falcons-vs-furia/Falcons vs Furia Major Final.mp4" --profile configs/falcons-furia-video.json --fps 5 10 30 --output data/processed/falcons-furia-fps-003
python -m stratscout compare-fps --input data/processed/falcons-furia-fps-003 --windows 55:10 115:10 190:25
```

Use a fresh suffix: `fps-002` contains the final processing acceptance run. Pilot
example: add `--start 115 --duration 10` and choose a fresh output directory. Its
comparison window is `115:10`, not `0:10`. The three default rates share a timestamp-
based 30 fps stream; every third/sixth frame supplies 10/5 fps. Requested subsets
use their highest rate as the base. Rates outside 5/10/30, duplicate rates, invalid
intervals and sources below the highest requested rate are refused.

FFmpeg decodes/crops once and directly encodes each branch: **libx264, CRF 18, medium,
yuv420p, no audio, MP4 fast-start**. Lower rates are never recompressed from the 30 fps
export. Native icon resolution is preserved. Every output is fully decoded before
publication, with frame-count, dimension, rate and duration checks. See the
[timestamp-aware fps filter](https://ffmpeg.org/ffmpeg-filters.html#fps).

The manifest records the requested source offset and sampling grid, not an exact
mapping to every original frame/demo tick. Non-grid start times and quantised source
timestamps can shift the chosen observation slightly. End duration is rounded to
complete frames, up to a low-rate sampling interval; playback is not slowed down.
Nominal/average source FPS cannot prove that a recording has no dropped frames.

## Review the comparison

Open `comparison/report.html`. Previews play at 30 fps; lower-rate panels hold the
previous observation. Labels show common playback time and held observation time.
PNG stills show start/middle/end; JSON records their frame indices. File sizes use
decimal MB. Decode times are complete-file validation wall times, include process
start-up and depend on caching/CPU load. They exclude detector processing.

Review 55–65 s for deaths/movement, 115–125 s for bomb carriers, and 190–215 s for
numbered/overlapping icons. Lower rates cannot uncover bomb-covered players. These
reports establish neither accuracy nor a winning frame rate. All variants/windows
retain one match ID for future dataset splitting. The old 32-frame sampler and
tracker are outside this comparison.

Use `--match-id` when separate source files belong to one match. Otherwise identity
defaults to the source SHA-256. The supplied profile's readable match name is used
only when its pinned source hash matches, so reusing a crop cannot silently label
an unrelated recording as Falcons–FURIA.

Manual capture replaces the former OBS automation workflow. Processing has been
validated on the existing match; check each new capture layout separately. Choose
the training frame rate later using reviewed labels and held-out matches.
