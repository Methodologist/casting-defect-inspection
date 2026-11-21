# From demo to factory: the simplest working setup

A practical guide to turning this project's approach into a camera inspection station on a real line. The
example is an aluminum die-cast car door on a conveyor, but the same steps fit most parts.

**The simplest version:** one station, one part type, a pass/fail decision, fixed cameras, and people sorting
photos into two folders. No custom software beyond what this repository already does.

---

## The whole process on one page

| Step | What happens | Who | Typical time |
|---|---|---|---|
| 1. Agree on "bad" | Write down which defects must be caught and show example parts | Quality team | 1–2 weeks |
| 2. Build the station | Enclosure, lights, cameras, a part sensor | Integrator / maintenance | 2–6 weeks |
| 3. Collect photos | Photograph every part; make no decisions yet | Automatic | 4–12 weeks |
| 4. Sort photos | Inspectors drag each photo into `good/` or `defective/` | Inspectors | a few minutes per 100 photos |
| 5. Train and test | Train on older photos, test on the newest ones | Engineer | a day |
| 6. Run alongside people | The model decides silently; compare with inspectors | Everyone | 2–4 weeks |
| 7. Switch on | Rejects go to a reject lane; people audit a sample | Production | ongoing |

A first station usually takes **3 to 6 months**, most of it collecting enough defective parts.

---

## 1. Agree on what "defective" means

Before buying anything, the quality team answers four questions:

1. **Which defects must be caught?** For a casting: cracks, cold shuts, misruns, flash, dents, surface porosity.
2. **What is the smallest one that matters?** For example 1 mm. This decides the camera.
3. **Where are the borderline cases?** Keep a few "just acceptable" and "just rejectable" parts as reference
   samples. The model can only be as consistent as the people who label for it.
4. **Which mistake is worse?** A missed crack in a structural part usually costs far more than scrapping a good
   one. That choice sets how cautious the system is.

**Know the limits:** a camera only sees the surface. Porosity inside the casting needs X-ray.

## 2. Build the station

Consistent photos matter more than anything else. In this project, the impeller photos (one camera, one setup)
gave 100% accuracy; the web-collected box photos (every photo different) did not.

- **Enclosure:** a light-tight hood or tunnel over the conveyor, so daylight and overhead lights never change the
  photos.
- **Lighting:** a diffuse light for the general surface, plus a low-angle light that makes cracks and dents cast
  shadows. Polarizing filters cut glare on shiny aluminum.
- **Cameras, sized to the smallest defect:** a defect needs about 3 to 5 pixels across to be seen. For 1 mm
  defects on a 1.2 m door that is roughly 5,000 pixels across the part, more than one camera gives.
  **Simplest answer: several fixed cameras, each covering one zone** (for example 4 to 6 zones per door). Each
  zone photo is judged on its own, so small defects stay large in the image.
  *Alternative in a robot cell:* the robot holds the part in front of one camera at several angles.
- **Trigger:** a sensor sees the part arrive, and the PLC fires the lights and cameras at the same spot every time.
  A stop, fixture or robot pick keeps the part's position repeatable.
- **Inspect after the part has cooled and been trimmed:** heat, steam and release-agent residue all change the photos.
- **Part ID:** if parts are laser-marked with a code, read it in the same shot so every photo links to its die,
  shift and time.
- **Computer:** an industrial PC next to the station, with a graphics card. It saves photos now and runs the model
  later.

## 3. Collect photos

Run the station for weeks **without making any decisions**. Save every photo, named by part ID, zone and time.

- **Defects are rare.** At 2% scrap, 1,000 parts give about 20 defective ones, too few to learn from. Aim for at
  least **200 to 300 defective zone photos**. To get there faster, pull known scrap from the scrap bin and run it
  back through the station.
- **Capture normal variation:** different shifts, die changes, die temperatures, release-agent batches. The model
  should learn that these are normal.

## 4. Sort the photos (labeling, the simple way)

Inspectors sort zone photos into two folders, exactly like the datasets in this project:

```
data/doors/train/good/        data/doors/train/defective/
data/doors/test/good/         data/doors/test/defective/
```

- A file browser is enough. A free labeling tool (Label Studio or CVAT) is faster once there are thousands of photos.
- Write a one-page defect guide with example photos, and have **two inspectors sort the same 200 photos**. Where
  they disagree, decide against the physical part, and update the guide.
- When unsure, check the part itself. A wrong label teaches the model the wrong thing (see the mislabeled box
  photos in this project).

## 5. Train and test

- **Split by date, not at random.** Consecutive castings look nearly identical, so a random split puts near-copies
  in both training and testing and makes results look better than they are. For example: train on weeks 1–6,
  choose settings on week 7, and keep week 8 untouched as the final exam.
- Train a CNN the way `train_local_cnn.py` does here. On a single graphics card this takes minutes.
- **Report two numbers:** missed defects and good parts rejected. The quality manager picks the cut-off between
  them.
- **Be realistic about proof:** to claim fewer than 1 in 100 defects slip through, you need roughly 300 defective
  test photos with none missed.

## 6. Run it alongside people

For 2 to 4 weeks the model decides on every part **but nothing acts on it**. Inspectors keep working as before.
Compare the two every day:

- Where they disagree, look at the part. Sometimes the model is right.
- Add the disagreements to the training folders and retrain.

## 7. Switch it on

- **Wiring:** the PC sends pass or fail to the PLC (digital output or the plant's fieldbus). Fail sends the part to
  a reject lane or a review table.
- **Operator screen:** shows the photo, the heatmap and the verdict, so the reason for a reject is obvious.
- **Fail safe:** if a camera or the PC stops responding, the PLC **holds** parts. It never passes them unchecked.
- **Every shift:** run one reference part with a known defect through the station to prove the cameras and lights
  still work.
- **Every week or month:** add inspector overrides to the training folders, retrain, and deploy the new model only
  if it scores better on the same fixed test photos.

---

## Rough shopping list for one station

| Item | Rough cost |
|---|---|
| Industrial cameras with lenses (4 to 6) | a few thousand dollars each |
| Lights and light-tight enclosure | a few thousand dollars |
| Industrial PC with a graphics card | $3,000–8,000 |
| Part sensor, mounting, wiring | under $1,000 |
| **Integration and labeling time** | **usually the largest cost** |

Running cost is mostly electricity. A hosted cloud model, by comparison, bills by the hour around the clock.
Off-the-shelf smart cameras with built-in deep learning (from the major machine-vision vendors and robot makers)
cost more up front but come with support; they suit plants without in-house software skills.

## What this repository already gives you

| Factory step | In this repository |
|---|---|
| Folder-based labels | The datasets here are labeled the same way: one folder per class, mapped to labels in `DATASETS` in `common.py` (a new part type is one more entry there) |
| Leak-free train / test / held-out split | `common.py` (split and duplicate removal), checked by the tests in `tests/` |
| Train a CNN on a local graphics card | `train_local_cnn.py` |
| Separate counts for misses and false rejects | `step3_test.py`, `common.score()` |
| Heatmaps showing why a part was rejected | Grad-CAM in `sagemaker/job/train_gradcam.py` |
| A results page for the quality team | `build_site.py` |

**Not included (needed for a real line):** the camera station itself, PLC signals, the operator screen, and
splitting by date instead of at random.
