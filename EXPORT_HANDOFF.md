# Export Handoff — Pi0.5 Epoch-5 Checkpoint

## Situation

Training job `366ae0f4-4bd7-4391-9051-b7782a195fad` was stopped mid-epoch-6.
Best checkpoint is epoch 5 (val/loss 0.1186), already copied to the model output dir.
The system rebooted to clear a hung XPU DRM lock from stuck training processes.

## What needs to happen

1. Verify the container is healthy: `docker ps | grep physical-ai-studio-xpu`
2. Verify the checkpoint exists inside the container:
   ```
   docker exec physical-ai-studio-xpu ls -lh /app/storage/models/65138616-bc53-4e25-99ef-4a7e6ec99250/model.ckpt
   ```
3. Run the OpenVINO export (the script below) — this takes ~10-20 min on XPU
4. Update the job status in the database so the UI shows the model as completed

## Export command

```bash
docker exec physical-ai-studio-xpu python3 -c "
import sys
sys.path.insert(0, '/app/application/backend/src')
sys.path.insert(0, '/app/library/src')
from pathlib import Path
from physicalai.policies import Pi05
from physicalai.export import ExportablePolicyMixin

output_dir = Path('/app/storage/models/65138616-bc53-4e25-99ef-4a7e6ec99250')
checkpoint = output_dir / 'model.ckpt'

print('Loading Pi05 from checkpoint...')
policy = Pi05.load_from_checkpoint(str(checkpoint))
policy.eval()
print('Loaded.')

if isinstance(policy, ExportablePolicyMixin):
    for backend in policy.get_supported_export_backends():
        bname = backend.value if hasattr(backend, 'value') else str(backend)
        export_dir = output_dir / 'exports' / bname
        print(f'Exporting to {bname}...')
        policy.export(export_dir, backend=backend)
        print(f'Done: {export_dir}')
else:
    print('No export support')
"
```

## After export succeeds

The training job is stuck in `running` status in the DB (job ID `366ae0f4-4bd7-4391-9051-b7782a195fad`).
The UI won't show the model until the job is marked completed and a model record exists.
Ask Claude to fix the job status and register the model in the database.

## Key paths (inside container)

| Item | Path |
|---|---|
| Model output dir | `/app/storage/models/65138616-bc53-4e25-99ef-4a7e6ec99250/` |
| Checkpoint (epoch 5) | `/app/storage/models/65138616-bc53-4e25-99ef-4a7e6ec99250/model.ckpt` |
| Training cache | `/app/storage/cache/366ae0f4-4bd7-4391-9051-b7782a195fad/` |
| Export output (after) | `/app/storage/models/65138616-bc53-4e25-99ef-4a7e6ec99250/exports/` |

## Training job details

- Job ID: `366ae0f4-4bd7-4391-9051-b7782a195fad`
- Model output ID: `65138616-bc53-4e25-99ef-4a7e6ec99250`
- Dataset: 65 episodes, white mat, fully colored blocks
- Epochs trained: 5 complete (stopped during epoch 6)
- Best val/loss: 0.1186 at epoch 5
- Batch size: 4, ~5024 steps/epoch
