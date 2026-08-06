# BeyondMimic migration notes

Source project: `/home/zhang/code/train/lab/whole_body_tracking`

Migrated into `frog_lab` as:

- Runtime support package: `source/frog_lab/frog_lab/beyond_mimic`
- Task package: `source/frog_lab/frog_lab/tasks/beyond_mimic`
- Utility scripts: `scripts/beyond_mimic`

The source project was not modified. The copied task IDs are namespaced with
`FrogLab-Isaac-BeyondMimic-*` to avoid Gym registration collisions if the
original `whole_body_tracking` package is also installed.

The local source project does not currently contain the motion or SMPL asset
files referenced by the original configs, such as `smpl/motions/*.npz` and
`smpl/smpl_humanoid.usda`. Placeholder directories are present under
`source/frog_lab/frog_lab/beyond_mimic/assets`, but real motion assets still
need to be supplied before those configs can run.
