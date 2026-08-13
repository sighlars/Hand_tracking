# Real-Time Hand Tracking + Air Drawing

A webcam-based hand tracking app built with **MediaPipe** and **OpenCV**. Detects up
to two hands in real time, draws landmarks, labels each hand as Left/Right, counts
raised fingers, shows a live FPS counter — and turns your index finger into a pen
so you can draw in mid-air.

## Features
- Real-time hand + finger landmark detection (21 points per hand)
- Left/Right hand classification with confidence score
- Per-hand finger-up counting (0–5)
- **Air drawing**: point with only your index finger to draw on screen
- **Gesture erase**: open palm clears the canvas
- **Pinch distance**: live pixel distance between thumb and index tip
- Adjustable brush color (4 presets) and brush size
- Live FPS counter with smoothing
- Clean on-screen HUD, mirrored camera view
- Screenshot capture (`s` key)

## Demo
*(Add a GIF or screenshot here once you've recorded one — this is the first thing
recruiters/visitors will look at.)*

## Setup

```bash
git clone <your-repo-url>
cd hand-tracking
pip install -r requirements.txt
python hand_tracking.py
```

Requires a webcam. Tested on Python 3.9–3.12.

## Controls
| Key   | Action        |
|-------|---------------|
| q     | Quit          |
| s     | Save screenshot to `./screenshots/` |
| c     | Clear the drawing canvas |
| 1–4   | Change brush color (cyan / magenta / yellow / white) |
| + / - | Increase / decrease brush size |

## Gestures
| Gesture | Effect |
|---|---|
| Only index finger extended | Draws on the canvas, following your fingertip |
| All five fingers extended (open palm) | Clears the canvas |
| Thumb + index finger apart | Shows live pinch distance in pixels, with a connecting line |

## How it works
1. Each frame is captured from the webcam and flipped horizontally for a natural
   mirror view.
2. MediaPipe's `Hands` solution detects up to 2 hands and returns 21 3D landmarks
   per hand, plus a Left/Right classification.
3. Finger state is computed by comparing the y-coordinate (or x-coordinate, for the
   thumb) of each fingertip landmark against its corresponding joint — if the tip is
   "above" the joint, the finger is considered extended.
4. When a hand's finger pattern matches "only index extended," its fingertip
   position is pushed onto a persistent NumPy canvas as a line segment connecting
   this frame's position to the last one — this is what makes the drawing appear
   continuous instead of a series of disconnected dots.
5. Pinch distance is just the Euclidean distance between the thumb tip and index
   tip landmarks, converted from normalized coordinates to pixels.
6. Results are drawn back onto the frame along with an FPS counter and HUD panel.

## Possible extensions
- Gesture recognition (e.g. thumbs up, peace sign, fist) mapped to actions
- Volume/brightness control via pinch distance
- Air-drawing / virtual whiteboard mode
- Sign language letter recognition

## Tech stack
- [MediaPipe](https://github.com/google/mediapipe) — hand landmark detection
- [OpenCV](https://opencv.org/) — video capture and rendering
- Python 3

## License
MIT