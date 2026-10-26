# Model checkpoint

Place the pretrained checkpoint in this directory:

```text
model/checkpoint-step0200000.pth
```

This is the step-200000 checkpoint of the from-scratch RF prior. The inference
configuration uses this path and the loader selects `ema_params1`.

For a checkpoint from your own training run, set `model_path` in
`config/MAR_pnp_rf.yaml` to its location. Model architecture and input size
remain JiT-B/16 and single-channel 512 x 512.

Weights are distributed separately. Checkpoint files (`*.pth`, `*.pt`,
`*.ckpt`, `*.safetensors`) are ignored by Git.
