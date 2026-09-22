"""Matched uint8 MRI controls and bounded DINOv2 adaptation; fixed final epoch."""

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn

from rsnaknee.baseline import _score
from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.data import sha256
from rsnaknee.features import checkpoint_provenance, normalize_rgb
from rsnaknee.image_model import read_frozen_labels
from rsnaknee.window_model import (AttentionHead, BATCH_SIZE, EPOCHS, GENERIC_WEIGHT_SHA256,
                                  SEED, build_supervision, masked_bce)

ARMS = ('frozen', 'late_blocks')
TRAINABLE_BLOCKS = {'frozen': 0, 'late_blocks': 2, 'deep_blocks': 6, 'soft_targets': 2, 'multi_windows': 2,
                    'all_windows': 2}
TARGET_MODES = {'frozen': 'binary', 'late_blocks': 'binary', 'deep_blocks': 'binary',
                'soft_targets': 'public_scores', 'multi_windows': 'binary', 'all_windows': 'binary'}
TRAIN_WINDOWS_PER_PLANE = {arm: 10 if arm == 'all_windows' else 3 if arm == 'multi_windows' else 1
                           for arm in TRAINABLE_BLOCKS}
BUDGET_SECONDS = 7.5 * 3600
PROBE_STEPS = 64


def selected_arms(arms):
    if (not isinstance(arms, (tuple, list)) or not arms
            or any(arm not in TRAINABLE_BLOCKS for arm in arms) or len(set(arms)) != len(arms)):
        raise ValueError('Require nonempty unique supported adaptation arms')
    return tuple(arms)


def window_indices(study_ids) -> np.ndarray:
    """Stable [epoch, study, plane] table independent of partition and arm order."""
    ids = list(study_ids)
    if len(set(ids)) != len(ids):
        raise ValueError('Window schedule requires unique study IDs')
    table = np.empty((EPOCHS, len(ids), 3), dtype=np.uint8)
    for epoch in range(EPOCHS):
        for row, study in enumerate(ids):
            for plane in range(3):
                digest = hashlib.sha256(f'{SEED}:{epoch}:{study}:{plane}'.encode()).digest()
                table[epoch, row, plane] = int.from_bytes(digest[:8], 'little') % 10
    return table


def multi_window_indices(study_ids) -> np.ndarray:
    """Keep the old anchor and add two distinct, independently hashed windows."""
    ids = list(study_ids)
    anchors = window_indices(ids)
    table = np.empty((*anchors.shape, 3), dtype=np.uint8)
    for epoch in range(EPOCHS):
        for row, study in enumerate(ids):
            for plane in range(3):
                anchor = int(anchors[epoch, row, plane])
                remaining = [window for window in range(10) if window != anchor]
                ranked = sorted(remaining, key=lambda window: hashlib.sha256(
                    f'multi-window-v1:{SEED}:{epoch}:{study}:{plane}:{window}'.encode()).digest())
                table[epoch, row, plane] = sorted([anchor, *ranked[:2]])
    return table


def all_window_indices(study_ids) -> np.ndarray:
    """Use all ten identities per plane in every epoch; no window sampling."""
    ids = list(study_ids)
    if len(set(ids)) != len(ids):
        raise ValueError('Window schedule requires unique study IDs')
    return np.tile(np.arange(10, dtype=np.uint8), (EPOCHS, len(ids), 3, 1))


def arm_window_schedule(tables, arm, study_count):
    table = tables[arm] if isinstance(tables, dict) else tables
    count = TRAIN_WINDOWS_PER_PLANE[arm]
    shape = (EPOCHS, study_count, 3) + ((count,) if count > 1 else ())
    if (not isinstance(table, np.ndarray) or table.shape != shape or table.dtype != np.uint8
            or not np.isin(table, range(10)).all()
            or (arm == 'all_windows' and not np.all(table == np.arange(10)))
            or (table.ndim == 4 and np.any(np.diff(np.sort(table.astype(int), axis=-1), axis=-1) == 0))):
        raise ValueError('Invalid window schedule for arm: ' + arm)
    return table


def window_schedule_provenance(table, arm):
    return {'train_windows_per_plane': TRAIN_WINDOWS_PER_PLANE[arm],
            'window_schedule_sha256': hashlib.sha256(table.tobytes(order='C')).hexdigest(),
            'artifact': 'all_window_indices.npy' if arm == 'all_windows' else
                        'multi_window_indices.npy' if arm == 'multi_windows' else 'window_indices.npy'}


def slot_logits(head: AttentionHead, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Use unchanged head parameters, masking individual sampled window identities."""
    if (features.ndim != 4 or tuple(features.shape[1:]) != (3, 10, 768)
            or mask.shape != (len(features), 30) or mask.dtype != torch.bool
            or not mask.any(1).all()):
        raise ValueError('Require 30 slots and at least one usable window per study')
    hidden = head.proj(features.float().reshape(-1, 30, 768)) + head.slot_emb
    attention = torch.einsum('bsh,th->bts', hidden, head.query) / 128 ** .5
    attention = attention.masked_fill(~mask.unsqueeze(1), float('-inf')).softmax(-1)
    context = head.drop(torch.einsum('bts,bsh->bth', attention, hidden))
    return (context * head.out.weight.unsqueeze(0)).sum(-1) + head.out.bias


class MRIModel(nn.Module):
    def __init__(self, encoder: nn.Module, arm: str):
        super().__init__()
        if arm not in TRAINABLE_BLOCKS:
            raise ValueError('Unknown adaptation arm')
        self.arm = arm
        self.encoder = encoder.requires_grad_(False)
        blocks = TRAINABLE_BLOCKS[arm]
        if blocks:
            if len(encoder.encoder.layer) < blocks:
                raise ValueError(f'Need at least {blocks} encoder blocks')
            for block in encoder.encoder.layer[-blocks:]:
                block.requires_grad_(True)
            encoder.layernorm.requires_grad_(True)
        self.head = AttentionHead()
        self.encoder.eval()

    def training_provenance(self):
        indices = [i for i, block in enumerate(self.encoder.encoder.layer)
                   if any(p.requires_grad for p in block.parameters())]
        return {'arm': self.arm, 'encoder_blocks': len(self.encoder.encoder.layer),
                'trainable_blocks': len(indices), 'trainable_block_indices': indices,
                'final_layernorm_trainable': all(p.requires_grad for p in self.encoder.layernorm.parameters()),
                'encoder_trainable_parameters': sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
                'head_trainable_parameters': sum(p.numel() for p in self.head.parameters() if p.requires_grad)}

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Late blocks retain gradients, but no encoder dropout changes.
        return self

    def forward(self, pixels: np.ndarray, presence: np.ndarray, windows=None):
        if (pixels.dtype != np.uint8 or pixels.ndim != 5 or pixels.shape[1:3] != (3, 12)
                or presence.shape != pixels.shape[:2] or not np.isin(presence, [0, 1]).all()
                or not np.all(presence.sum(1) > 0)):
            raise ValueError('Require uint8 twelve-slice pixels and usable binary presence')
        if windows is not None:
            if (not isinstance(windows, np.ndarray) or windows.shape not in
                    (presence.shape, (*presence.shape, 3), (*presence.shape, 10))
                    or not np.issubdtype(windows.dtype, np.integer) or not np.isin(windows, range(10)).all()
                    or (windows.ndim == 3 and np.any(np.diff(np.sort(windows.astype(int), axis=-1), axis=-1) == 0))):
                raise ValueError('Require one, three or ten distinct valid integer windows per plane')
        locations = [(row, plane, window) for row in range(len(pixels)) for plane in range(3)
                     if presence[row, plane] for window in
                     (range(10) if windows is None else
                      [int(windows[row, plane])] if windows.ndim == 2 else windows[row, plane].tolist())]
        device = next(self.parameters()).device
        vectors, slots = [], []
        # Only selected windows go to the GPU; frozen early layers retain no tape.
        for start in range(0, len(locations), 24):
            batch = locations[start:start + 24]
            x = np.stack([pixels[row, plane, window:window + 3] for row, plane, window in batch])
            x = torch.from_numpy(normalize_rgb(x.astype(np.float32) / 255.)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                tokens = self.encoder(pixel_values=x).last_hidden_state
                value = torch.cat((tokens[:, 0], tokens[:, 1:].mean(1)), 1)
            vectors.append(value.float())
            slots.extend(row * 30 + plane * 10 + window for row, plane, window in batch)
        values = torch.cat(vectors)
        if values.shape != (len(locations), 768) or not torch.isfinite(values).all():
            raise ValueError('Invalid encoder windows')
        indices = torch.tensor(slots, device=device)
        features = values.new_zeros((len(pixels) * 30, 768)).index_copy(0, indices, values)
        mask = torch.zeros(len(pixels) * 30, dtype=torch.bool, device=device)
        mask[indices] = True
        return slot_logits(self.head, features.reshape(-1, 3, 10, 768), mask.reshape(-1, 30))


def make_optimizer(model: MRIModel):
    groups = [{'params': model.head.parameters(), 'lr': 1e-3}]
    backbone = [p for p in model.encoder.parameters() if p.requires_grad]
    if backbone:
        groups.append({'params': backbone, 'lr': 8e-6})
    return torch.optim.AdamW(groups, weight_decay=.02)


def training_partition(labels: pd.DataFrame, heldout_fold, *, target_mode='binary'):
    if target_mode not in ('binary', 'public_scores'):
        raise ValueError('Unknown target mode')
    if (labels.index.has_duplicates or labels[['fold', 'group_id']].isna().any().any()
            or labels.groupby('group_id')['fold'].nunique().gt(1).any()):
        raise ValueError('Invalid unique studies or frozen report group assignments')
    if heldout_fold is not None and heldout_fold not in labels.fold.unique():
        raise ValueError('Unknown held-out fold')
    keep = np.ones(len(labels), dtype=bool) if heldout_fold is None else labels.fold.ne(heldout_fold).to_numpy()
    # The entire held-out fold is removed before resolving any task supervision.
    rows = np.flatnonzero(keep)
    targets, weights = build_supervision(labels.iloc[rows])
    usable = weights.sum(1) > 0
    if not usable.any():
        raise ValueError('No supervised training studies')
    for column, target in enumerate(TARGET_COLUMNS):
        if np.unique(targets[weights[:, column] > 0, column]).size != 2:
            raise ValueError(f'Training target lacks both classes: {target}')
    if target_mode == 'public_scores':
        for column, target in enumerate(TARGET_COLUMNS):
            silver = weights[:, column] == .25
            if not silver.any():
                continue
            if target + '__derived' not in labels:
                raise ValueError('Missing active public scores: ' + target)
            scores = pd.to_numeric(labels.iloc[rows[silver]][target + '__derived'], errors='coerce').to_numpy(dtype=float)
            if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError('Invalid active public scores: ' + target)
            targets[silver, column] = scores.astype(np.float32)
    return rows[usable], targets[usable], weights[usable], int((~keep).sum())


def supervision_provenance(target_mode, targets, weights):
    # Bound to the separately saved training ID order and fixed TARGET_COLUMNS.
    return {'target_mode': target_mode,
            'targets_sha256': hashlib.sha256(targets.astype('<f4').tobytes(order='C')).hexdigest(),
            'weights_sha256': hashlib.sha256(weights.astype('<f4').tobytes(order='C')).hexdigest()}


def training_step(model, optimizer, scaler, pixels, presence, targets, weights, windows):
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    logits = model(pixels, presence, windows)
    loss = masked_bce(logits, torch.from_numpy(targets).to(device), torch.from_numpy(weights).to(device))
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite adaptation training loss')
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())


def train_fold(factory, arm, pixels, presence, labels, cache_rows, table, heldout_fold):
    started = time.perf_counter()
    table = arm_window_schedule(table, arm, len(labels))
    mode = TARGET_MODES[arm]
    rows, targets, weights, excluded = training_partition(labels, heldout_fold, target_mode=mode)
    torch.manual_seed(SEED)
    model = factory(arm).train()
    optimizer = make_optimizer(model)
    scaler = torch.amp.GradScaler('cuda', enabled=next(model.parameters()).is_cuda)
    random = np.random.default_rng(SEED)
    history = []
    for epoch in range(EPOCHS):
        order = random.permutation(len(rows))
        total = 0.
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            selected = cache_rows[rows[batch]]
            loss = training_step(model, optimizer, scaler, np.asarray(pixels[selected]),
                                 np.asarray(presence[selected]), targets[batch], weights[batch], table[epoch, rows[batch]])
            total += loss * len(batch)
        history.append(total / len(rows))
        print(f'{arm} fold={heldout_fold} epoch={epoch + 1}/{EPOCHS} loss={history[-1]:.6f}', flush=True)
    return model.eval(), {'heldout_fold': heldout_fold, 'training_ids': labels.index[rows].tolist(),
                         'encoder_adaptation': model.training_provenance(),
                         **window_schedule_provenance(table, arm),
                         **supervision_provenance(mode, targets, weights),
                         'excluded_fold_studies': excluded, 'epoch_training_loss': history,
                         'runtime_seconds': time.perf_counter() - started,
                         'target_counts': {target: {'observed': int((weights[:, i] == 1).sum()),
                                                   'derived': int((weights[:, i] == .25).sum())}
                                           for i, target in enumerate(TARGET_COLUMNS)}}


@torch.no_grad()
def predict_pixels(model, pixels, presence, rows):
    model.eval()
    predictions = []
    for start in range(0, len(rows), BATCH_SIZE):
        selected = rows[start:start + BATCH_SIZE]
        predictions.append(model(np.asarray(pixels[selected]), np.asarray(presence[selected])).sigmoid().cpu().numpy())
    values = np.concatenate(predictions) if predictions else np.empty((0, 12), np.float32)
    if values.shape != (len(rows), 12) or not np.isfinite(values).all():
        raise ValueError('Incomplete or nonfinite image predictions')
    return values


def load_model_state(model: MRIModel, path: Path, *, provenance=None):
    if provenance is not None and provenance != model.training_provenance():
        raise ValueError('Encoder adaptation provenance differs from loaded model')
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return model.eval()


def project_runtime(step_seconds, training_steps, inference_seconds_per_study, inference_studies, elapsed):
    # 25% throughput margin plus ten minutes for model loads, safe saves and final hashing.
    return elapsed + 600 + 1.25 * (sum(step_seconds.values()) * training_steps
                                  + inference_seconds_per_study * inference_studies * len(step_seconds))


def throughput_probe(factory, pixels, presence, labels, cache_rows, table, elapsed, *, arms=ARMS):
    arms = selected_arms(arms)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seconds, inference, adaptation, supervision = {}, [], {}, {}
    schedules, memory, sampled_ids = {}, {}, {}
    start_probe = time.perf_counter()
    # Disposable models and forked RNG ensure the probe cannot seed actual fits.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        for arm in arms:
            arm_table = arm_window_schedule(table, arm, len(labels))
            schedules[arm] = window_schedule_provenance(arm_table, arm)
            mode = TARGET_MODES[arm]
            rows, targets, weights, _ = training_partition(labels, 0, target_mode=mode)
            supervision[arm] = supervision_provenance(mode, targets, weights)
            probe_rows = np.arange(len(rows))
            if arm in ('multi_windows', 'all_windows'):
                probe_rows = np.flatnonzero(np.asarray(presence[cache_rows[rows]]).all(1))
                if len(probe_rows) < BATCH_SIZE:
                    raise ValueError('Multi-window probe requires eight distinct full-plane training studies')
            sampled = []
            full_plane_batch_size = BATCH_SIZE
            torch.manual_seed(SEED)
            model = factory(arm).train()
            adaptation[arm] = model.training_provenance()
            optimizer = make_optimizer(model)
            scaler = torch.amp.GradScaler('cuda', enabled=next(model.parameters()).is_cuda)
            if device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for step in range(PROBE_STEPS):
                batch = probe_rows[(np.arange(BATCH_SIZE) + step * BATCH_SIZE) % len(probe_rows)]
                selected = cache_rows[rows[batch]]
                sampled.extend(labels.index[rows[batch]].tolist())
                full_plane_batch_size = min(full_plane_batch_size, int(np.asarray(presence[selected]).all(1).sum()))
                training_step(model, optimizer, scaler, np.asarray(pixels[selected]), np.asarray(presence[selected]),
                              targets[batch], weights[batch], arm_table[step % EPOCHS, rows[batch]])
            if device == 'cuda':
                torch.cuda.synchronize()
            seconds[arm] = (time.perf_counter() - started) / PROBE_STEPS
            selected = cache_rows[rows[:BATCH_SIZE]]
            started = time.perf_counter()
            predict_pixels(model, pixels, presence, selected)
            if device == 'cuda':
                torch.cuda.synchronize()
            inference.append((time.perf_counter() - started) / len(selected))
            sampled_ids[arm] = list(dict.fromkeys(sampled))
            memory[arm] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                           'peak_reserved_bytes': torch.cuda.max_memory_reserved() if device == 'cuda' else None,
                           'total_device_bytes': torch.cuda.get_device_properties(0).total_memory if device == 'cuda' else None,
                           'full_plane_batch_size': full_plane_batch_size}
            del model, optimizer, scaler
            if device == 'cuda':
                torch.cuda.empty_cache()
    steps = EPOCHS * sum(math.ceil(len(training_partition(labels, fold)[0]) / BATCH_SIZE)
                        for fold in (0, 1, 2, None))
    gold = labels[[t + '__observed' for t in TARGET_COLUMNS]].notna().any(axis=1).sum()
    test_count = len(pixels) - len(cache_rows)
    projected = project_runtime(seconds, steps, max(inference), int(gold) + test_count,
                                elapsed + time.perf_counter() - start_probe)
    return {'selected_arms': list(arms), 'encoder_adaptation': adaptation,
            'supervision_by_arm': supervision,
            'window_schedule_by_arm': schedules, 'memory_by_arm': memory,
            'probe_sampled_ids_by_arm': sampled_ids,
            'steps_per_arm': PROBE_STEPS, 'training_fold_excluded': 0,
            'seconds_per_step': seconds, 'total_training_steps_per_arm': steps,
            'inference_seconds_per_study': max(inference), 'projected_total_seconds': projected,
            'budget_seconds': BUDGET_SECONDS, 'within_budget': projected < BUDGET_SECONDS,
            'hidden_1300_cached_pixel_forward_seconds': 1300 * max(inference),
            'inference_projection_excludes_dicom_preparation': True,
            'probe_training_ids': labels.index[rows].tolist(),
            'rng_and_weights': 'Disposable probe models; torch RNG restored; real fits reseeded'}


def read_pixel_cache(cache_dir: Path, audit: dict):
    """Validate immutable cache once; the ~8GB pixel array stays read-only/mapped."""
    manifest = json.loads((cache_dir / 'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Need a complete pixel cache')
    checkpoint = manifest.get('checkpoint', {})
    if (manifest.get('encoder_frozen') is not True or manifest.get('encoder_fit_on_competition_data') is not False
            or manifest.get('labels_or_reports_read') is not False
            or checkpoint.get('model_type') != 'dinov2' or checkpoint.get('hidden_size') != 384
            or checkpoint.get('files_sha256', {}).get('pytorch_model.bin') != GENERIC_WEIGHT_SHA256):
        raise ValueError('Pixel cache must originate from audited generic extraction')
    if (manifest.get('recipe') != 'central12-neighbor3-mean10-v1'
            or manifest.get('planes') != ['Sagittal', 'Coronal', 'Axial']
            or manifest.get('window_order') != [list(range(i, i + 3)) for i in range(10)]
            or manifest.get('embedding_layout') != 'CLS[384] then mean-patch[384]'
            or manifest.get('pixel_cache_quantization') !=
            'round(255*clip(preprocessed_float,0,1)); cache only; feature extraction uses original float32'):
        raise ValueError('Pixel cache recipe or layout differs')
    names = {'train.csv', 'test.csv', 'train_series.csv', 'test_series.csv'}
    if set(manifest.get('input_sha256', {})) != names or any(
            manifest['input_sha256'][name] != audit.get('input_sha256', {}).get(name) for name in names):
        raise ValueError('Pixel cache and label source metadata differ')
    for name in ('__init__.py', 'coverage.py', 'features.py', 'imaging.py'):
        if sha256(Path(__file__).with_name(name)) != manifest.get('source_sha256', {}).get(name):
            raise ValueError('Pixel cache preprocessing source differs: ' + name)
    for name in ('IDs.csv', 'presence.npy', 'pixels_uint8.npy'):
        if sha256(cache_dir / name) != manifest.get('artifact_sha256', {}).get(name):
            raise ValueError(f'Pixel cache hash differs: {name}')
    ids = pd.read_csv(cache_dir / 'IDs.csv', dtype=str)
    if (set(ids.columns) != {ID_COLUMN, 'split'} or ids.empty or ids.isna().any().any()
            or ids[ID_COLUMN].str.strip().eq('').any() or ids[ID_COLUMN].duplicated().any()
            or set(ids['split']) != {'train', 'test'}):
        raise ValueError('Invalid pixel cache IDs')
    pixels = np.load(cache_dir / 'pixels_uint8.npy', allow_pickle=False, mmap_mode='r')
    presence = np.load(cache_dir / 'presence.npy', allow_pickle=False, mmap_mode='r')
    if (pixels.shape != (len(ids), 3, 12, 224, 224) or pixels.dtype != np.uint8
            or manifest.get('pixel_cache_shape') != list(pixels.shape) or manifest.get('pixel_cache_dtype') != 'uint8'
            or presence.shape != (len(ids), 3) or presence.dtype != np.uint8
            or manifest.get('presence_shape') != list(presence.shape)
            or manifest.get('completed_studies') != len(ids)
            or not np.isin(presence, [0, 1]).all() or not np.all(presence.sum(1) > 0)):
        raise ValueError('Invalid or incomplete pixel cache dimensions/presence')
    return ids, pixels, presence, manifest


def run_comparison(cache_dir: Path, checkpoint_dir: Path, labels_path: Path, output: Path,
                   audit_path: Path = Path('artifacts/reports/label_audit_image_v1.json'), device='cuda', *,
                   arms=ARMS, code_provenance=None):
    from transformers import AutoModel

    arms = selected_arms(arms)
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(f'Preserve earlier runs: {output}')
    audit = json.loads(audit_path.read_text())
    ids, pixels, presence, manifest = read_pixel_cache(cache_dir, audit)
    provenance = checkpoint_provenance(checkpoint_dir)
    if provenance['files_sha256'] != manifest['checkpoint']['files_sha256']:
        raise ValueError('Generic initialization and cache checkpoint hashes differ')
    labels = read_frozen_labels(labels_path, audit_path)
    cache_rows = np.flatnonzero(ids['split'].eq('train').to_numpy())
    if set(ids.iloc[cache_rows][ID_COLUMN]) != set(labels.index):
        raise ValueError('Pixel training IDs and labels differ')
    labels = labels.reindex(ids.iloc[cache_rows][ID_COLUMN])
    if set(labels.fold) != {0, 1, 2}:
        raise ValueError('Expected frozen folds 0, 1, 2')
    gold = labels[[t + '__observed' for t in TARGET_COLUMNS]].copy()
    gold.columns = TARGET_COLUMNS
    observed = gold.notna().any(axis=1)
    if observed.sum() != 58 or gold.loc[observed].isna().any().any():
        raise ValueError('Expected exactly 58 complete gold validation studies')
    table = window_indices(labels.index)
    tables = {arm: all_window_indices(labels.index) if arm == 'all_windows' else
                   multi_window_indices(labels.index) if arm == 'multi_windows' else table for arm in arms}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    def factory(arm):
        encoder = AutoModel.from_pretrained(str(checkpoint_dir), local_files_only=True,
                                            trust_remote_code=False, weights_only=True,
                                            attn_implementation='eager')
        torch.manual_seed(SEED)
        return MRIModel(encoder, arm).to(device)
    output.mkdir(parents=True)
    np.save(output / 'window_indices.npy', table)
    if 'multi_windows' in arms:
        np.save(output / 'multi_window_indices.npy', tables['multi_windows'])
    if 'all_windows' in arms:
        np.save(output / 'all_window_indices.npy', tables['all_windows'])
    pd.DataFrame({ID_COLUMN: labels.index}).to_csv(output / 'window_study_ids.csv', index=False)
    for source in (labels_path, labels_path.with_name('folds.csv'), audit_path, cache_dir / 'manifest.json'):
        shutil.copy2(source, output / ('feature_manifest.json' if source.name == 'manifest.json' else source.name))
    source_dir = output / 'source'
    source_dir.mkdir()
    for source in sorted(Path(__file__).parent.glob('*.py')):
        shutil.copy2(source, source_dir / source.name)
    summary = {'status': 'probing', 'arms': {}, 'selected_arms': list(arms), 'checkpoint': provenance,
               'code_provenance': code_provenance,
               'recipe': {'epochs': EPOCHS, 'effective_batch_size': BATCH_SIZE, 'seed': SEED,
                          'trainable_blocks_by_arm': {arm: TRAINABLE_BLOCKS[arm] for arm in arms},
                          'target_mode_by_arm': {arm: TARGET_MODES[arm] for arm in arms},
                          'head_learning_rate': .001, 'backbone_learning_rate': .000008, 'weight_decay': .02,
                          'hidden': 128, 'dropout': .2, 'silver_weight': .25, 'loss': 'BCE mean over batch x 12',
                          'train_windows_per_plane': {arm: TRAIN_WINDOWS_PER_PLANE[arm] for arm in arms},
                          'inference_windows_per_plane': 10,
                          'encoder_mode': 'eval', 'selection': 'fixed final epoch only', 'amp': device == 'cuda'},
               'input_sha256': manifest['input_sha256'], 'pixel_cache_sha256': manifest['artifact_sha256']['pixels_uint8.npy'],
               'labels_sha256': sha256(labels_path), 'split_sha256': sha256(labels_path.with_name('folds.csv')),
               'label_audit_sha256': sha256(audit_path), 'feature_manifest_sha256': sha256(cache_dir / 'manifest.json'),
               'window_indices_sha256': sha256(output / 'window_indices.npy'),
               'window_schedule_by_arm': {arm: window_schedule_provenance(tables[arm], arm) for arm in arms},
               'window_study_ids_sha256': sha256(output / 'window_study_ids.csv'),
               'source_sha256': {p.name: sha256(p) for p in source_dir.glob('*.py')},
               'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
                            'transformers': version('transformers')},
               'device': device, 'deterministic_algorithms': True,
               'limitations': ['58 repeatedly used gold studies', 'Patient independence unresolved']}
    summary_path = output / 'summary.json'
    if 'multi_windows' in arms:
        summary['multi_window_indices_sha256'] = sha256(output / 'multi_window_indices.npy')
    if 'all_windows' in arms:
        summary['all_window_indices_sha256'] = sha256(output / 'all_window_indices.npy')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    try:
        probe = throughput_probe(factory, pixels, presence, labels, cache_rows, tables, time.perf_counter() - started,
                                 arms=arms)
    except torch.cuda.OutOfMemoryError as error:
        summary.update(status='deferred_memory', runtime_seconds=time.perf_counter() - started,
                       probe={'error': str(error), 'batch_size': BATCH_SIZE, 'no_recipe_fallback': True})
        summary_path.write_text(json.dumps(summary, indent=2) + '\n')
        return summary
    probe_ids = output / 'probe_training_ids.csv'
    pd.DataFrame({ID_COLUMN: probe.pop('probe_training_ids')}).to_csv(probe_ids, index=False)
    probe['training_ids_sha256'] = sha256(probe_ids)
    summary['probe'] = probe
    print(json.dumps(probe, indent=2), flush=True)
    if not probe['within_budget']:
        summary.update(status='deferred_budget', runtime_seconds=time.perf_counter() - started)
        summary_path.write_text(json.dumps(summary, indent=2) + '\n')
        return summary
    summary['status'] = 'running'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    for arm in arms:
        directory = output / arm
        directory.mkdir()
        oof = pd.DataFrame(np.nan, index=labels.index[observed], columns=TARGET_COLUMNS)
        records = []
        for fold in (0, 1, 2, None):
            model, record = train_fold(factory, arm, pixels, presence, labels, cache_rows, tables, fold)
            stem = 'final' if fold is None else f'fold_{fold}'
            path = directory / ('model.pt' if fold is None else f'{stem}.pt')
            torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, path)
            training_ids = directory / f'{stem}_training_ids.csv'
            pd.DataFrame({ID_COLUMN: record.pop('training_ids')}).to_csv(training_ids, index=False)
            record.update(training_ids_sha256=sha256(training_ids), checkpoint_sha256=sha256(path))
            if fold is not None:
                validation = (labels.fold.eq(fold) & observed).to_numpy()
                predicted = pd.DataFrame(predict_pixels(model, pixels, presence, cache_rows[validation]),
                                         index=labels.index[validation], columns=TARGET_COLUMNS)
                oof.loc[predicted.index] = predicted
                record.update(fold=fold, **_score(gold.loc[validation], predicted))
                records.append(record)
            else:
                test_rows = np.flatnonzero(ids['split'].eq('test').to_numpy())
                predicted = pd.DataFrame(predict_pixels(model, pixels, presence, test_rows), columns=TARGET_COLUMNS)
                predicted.insert(0, ID_COLUMN, ids.iloc[test_rows][ID_COLUMN].to_numpy())
                predicted.to_csv(directory / 'submission.csv', index=False)
                final_record = record
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
        if oof.isna().any().any():
            raise ValueError('Incomplete gold OOF predictions')
        oof.to_csv(directory / 'oof.csv')
        summary['arms'][arm] = {'folds': records, 'final_fit': final_record, 'gold_validation_studies': len(oof),
                                'mean_macro_auc': float(np.mean([r['macro_auc'] for r in records])),
                                'std_macro_auc': float(np.std([r['macro_auc'] for r in records], ddof=1))}
        summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    summary.update(status='complete', runtime_seconds=time.perf_counter() - started)
    summary['artifact_sha256'] = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*')
                                  if p.is_file() and p != summary_path}
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=Path('data/processed/image-v1/report_labels.csv'))
    parser.add_argument('--label-audit', type=Path, default=Path('artifacts/reports/label_audit_image_v1.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', choices=tuple(TRAINABLE_BLOCKS), default=ARMS,
                        help='Matched arms to train; compare multi_windows (3 per plane) with all_windows (10 per plane)')
    args = parser.parse_args()
    run_comparison(args.cache, args.checkpoint, args.labels, args.output, args.label_audit, arms=args.arms)
