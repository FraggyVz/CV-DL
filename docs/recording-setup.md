# CS2 Recording and Radar Setup

Use a dedicated radar preset for demo recording. These settings are intended for
StratScout capture, not normal play. Many are saved by CS2; restarting the game
is not a reliable undo. No game settings are changed by the Python detector.

## Preserve your playing settings first

Before applying the preset, enter each command name below **without its value**
in the CS2 developer console to inspect its current value. For example:

```text
cl_radar_scale
cl_radar_rotate
```

Record the current values for all 15 commands in the preset, as `command value`
lines in a personal text file outside version control. Re-enter those lines after
recording to restore your exact preferences. Keep this backup separate from the
recording preset; do not replace it while the capture settings are active.

If you already applied the preset without a backup, your previous custom values
cannot be inferred from it. Use the default reset below, then adjust the radar
to your preference and save those values for future sessions. Do not reset all
CS2 settings just to undo these radar changes.

## Recording preset

Open a Mirage demo, then enter these commands in the **CS2 console**, not PowerShell.
Keep them out of an automatically executed gameplay config.

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

This starting preset requests a large, opaque, square radar without rotation or
dynamic zoom, zoomed out with moderate icon size. Individual teammate colours are
disabled and spectator visibility includes unspotted players. Explicitly verify
which colour means T and which means CT in the actual replay.

**This preset has not been validated in-game for this project.** Disabling player
centring alone does not guarantee a stationary map. Spectator radar behaviour can
differ from the normal radar; validate the rendered result before a long capture.

## Check a short recording first

Record 20–30 seconds before recording the match:

- Capture the full CS2 game frame. Start with 1920 × 1080 at constant 60 fps,
  normal replay speed and sufficient quality to preserve small icon edges.
- Verify all ten players appear while alive, including players behind walls,
  with two distinct team colours. Check movement, deaths and objective markers.
- Switch observed players during this test. Check that the map background stays
  aligned, fully visible and at the same scale. Do not use a moving crop.
- Check close icons remain distinguishable. Larger icons can merge; do not
  maximise icon size simply because the radar itself is larger.
- Close the console, scoreboard and replay controls before the actual recording.
  Keep resolution, HUD size, colours, orientation and zoom fixed throughout.

Once the test passes, capture at least three Mirage rounds; preferably record the
whole match once. Keep the original full-frame video without subsequent resizing
or cropping. StratScout selects the minimap crop during calibration.

Save the video as `data/raw/mirage.mp4`. Keep the source `.dem` and a text record of
match identity, CS2 version, recording resolution/frame rate, exact radar commands
and any timing offset beside it. The detector requires only the video. Follow the
[README](../README.md#prepare-the-first-recording) and
[annotation guidelines](annotation-guidelines.md) for calibration and review.

## Restore the listed defaults

Use your saved playing values if available. Otherwise, these reset **only the
15 settings above** to the defaults listed in the
[CS2 engine convar dump](https://raw.githubusercontent.com/SteamDatabase/GameTracking-CS2/master/DumpSource2/convars.txt),
checked on 7 October 2026. The dump is maintained by SteamDatabase from game data;
defaults can change with CS2 updates. These are not necessarily your old settings.
Boolean values are written as `1` (true) or `0` (false).

Paste into the CS2 console:

```text
cl_hud_radar_scale 1
cl_hud_radar_background_alpha 0.627
cl_hud_radar_blur_background 1
cl_hud_radar_map_additive 1
cl_radar_square_always 0
cl_radar_square_when_spectating 1
cl_radar_square_with_scoreboard 1
cl_radar_rotate 1
cl_radar_always_centered 1
cl_radar_scale_dynamic 0
cl_radar_scale 0.7
cl_radar_scale_alternate 1
cl_radar_icon_scale_min 0.6
cl_teammate_colors_show 1
cl_radar_show_all_players_when_spectating 1
```

After recording, restore your playing values or this block before joining a match.
If they change again on launch, check your existing autoexec or launch configuration
for commands that reapply the recording preset. If a command is unknown on your
installed version, stop and check its console help rather than assuming it applied.
