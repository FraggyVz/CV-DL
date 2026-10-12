"""Features: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import numpy as np

from .contracts import Round


def to_features(pos: np.ndarray, *, context) -> np.ndarray:
    vel = np.diff(pos, axis=0, prepend=pos[:1])
    feats = []
    for c in range(2):
        sl = slice(c * context.config.n_per_team, (c + 1) * context.config.n_per_team)
        team, tvel = (pos[:, sl], vel[:, sl])
        centroid = team.mean(axis=1)
        spread = np.linalg.norm(team - centroid[:, None, :], axis=-1).mean(axis=1)
        d_a = np.linalg.norm(centroid - context.state["A_SITE"][None], axis=-1)
        d_b = np.linalg.norm(centroid - context.state["B_SITE"][None], axis=-1)
        near_a = (np.linalg.norm(team - context.state["A_SITE"][None, None], axis=-1) < 0.18).mean(
            axis=1
        )
        near_b = (np.linalg.norm(team - context.state["B_SITE"][None, None], axis=-1) < 0.18).mean(
            axis=1
        )
        speed = np.linalg.norm(tvel, axis=-1).mean(axis=1)
        feats += [
            centroid[:, 0],
            centroid[:, 1],
            spread,
            d_a,
            d_b,
            near_a,
            near_b,
            speed,
        ]
    return np.stack(feats, axis=-1).astype(np.float32)


def to_sequence(pos: np.ndarray, mode: str = "features", *, context) -> np.ndarray:
    if mode == "features":
        return to_features(pos, context=context)
    if mode == "coords":
        return pos.reshape(pos.shape[0], -1).astype(np.float32)
    if mode == "both":
        return np.concatenate(
            [to_features(pos, context=context), pos.reshape(pos.shape[0], -1)], -1
        ).astype(np.float32)
    raise ValueError(mode)


def ct_post_signals(r: Round, *, context):
    """Defender signals after the plant, measured against the planted site."""
    site = context.state["A_SITE"] if r.plant_site == "A" else context.state["B_SITE"]
    d = np.linalg.norm(r.post_positions[:, context.config.n_per_team :] - site[None, None], axis=-1)
    return (d.mean(1), (d < 0.1).sum(1), d.min(1))
