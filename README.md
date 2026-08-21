# Pickleball Shot Predictor

A computer vision project that watches a video of a pickleball match, tracks the players and ball, figures out what type of shot is being hit, and predicts where the ball is likely to land next.

## What it does

Given a video of a match, the system outputs:

- Bounding boxes around each player and the ball
- A live probability breakdown of the current shot type (drive, dink, drop, lob, serve, volley)
- A heatmap over the court showing the predicted landing zone for the next shot

## How it works

The project is split into a few stages, each handled by its own model.

**1. Detection and tracking**
A YOLOv8 model, fine-tuned on labeled pickleball footage, finds the players and ball in each frame. ByteTrack is used on top of that to keep consistent IDs for each player across frames.

**2. Pose estimation**
For each tracked player, a pose model (RTMPose) extracts body keypoints per frame. This captures things like swing motion and stance, which end up being the main signal for identifying shot type.

**3. Shot classification**
A skeleton-based graph neural network (ST-GCN, with a possible upgrade to CTR-GCN) takes a short sequence of pose keypoints and ball motion around the moment of contact and predicts the shot type.

**4. Placement prediction**
A smaller model takes the predicted shot type, player position, and ball velocity, and predicts a probability distribution over zones on the opposing court, which becomes the heatmap.

**5. Rendering**
All of the above gets drawn back onto the video: bounding boxes, the shot probability panel, and the court heatmap.

## Pipeline overview

```
Video
  -> Detection + Tracking (players, ball)
  -> Pose Estimation (player keypoints)
  -> Shot Classifier (shot type probabilities)
  -> Placement Predictor (landing zone heatmap)
  -> Rendered output (annotated video)
```

## Status

Currently in the model-building phase. Detection, pose, shot classification, and placement prediction are being built and trained locally before any deployment work starts.