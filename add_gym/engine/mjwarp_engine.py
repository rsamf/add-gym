"""MuJoCo Warp physics engine implementation.

This module implements the BaseEngine/BaseScene/BaseEntity abstraction using
MuJoCo for model compilation and MJWarp (mujoco_warp + warp) for batched stepping.

Design notes:
- Multi-env execution is done via MJWarp's `nworld` batching (not spatial env spacing).
- Base pose is accessed via `get_pos()` / `get_quat()`.
- DOF vectors (`get_dofs_position/velocity`, `control_dofs_position`) are designed
    for scalar joint control and match the *client-facing* order used by
    KinCharModel/Genesis (breadth-first traversal of the kinematic tree):
    - floating base contributes 6 DOFs (3 linear + 3 angular)
    - all non-root joints are single-scalar DOFs
    - base orientation is not represented in the DOF position vector
      (use quaternions via `get_quat()` / `set_quat()` / `set_qpos()` when needed)
- `set_qpos()` expects: root pos xyz + root quat wxyz + scalar joint qpos in the
    same client joint order (KinCharModel/Genesis), and maps internally to MuJoCo qpos.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


try:
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
except ImportError as e:
    logger.warning(
        "Failed to import mujoco_warp or warp. MJWarpEngine will not be available."
    )


from add_gym.engine.base_engine import (
    BaseCamera,
    BaseEngine,
    BaseEntity,
    BaseJoint,
    BaseLink,
    BaseScene,
)


def _resolve_asset_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)

    engine_dir = Path(__file__).resolve().parent
    gym_pkg_dir = engine_dir.parent.parent

    candidates = [
        Path.cwd() / p,
        gym_pkg_dir / p,
        gym_pkg_dir.parent / p,
    ]

    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())

    return str((gym_pkg_dir / p).resolve())


@dataclass(frozen=True)
class _DofQposSegment:
    kind: str  # "free" | "ball" | "scalar"
    dof_local_adr: int
    dof_local_num: int
    qpos_adr: int
    qpos_num: int


class MJWarpCamera(BaseCamera):
    """Minimal camera stub to satisfy the environment API.

    The current training/view code only requires that camera methods exist.
    Rendering/recording can be implemented later.
    """

    def __init__(self):
        self._recording = False

    def follow_entity(self, entity: BaseEntity) -> None:  # noqa: ARG002
        return

    def start_recording(self) -> None:
        self._recording = True

    def stop_recording(self, filename: str, fps: int = 30) -> None:  # noqa: ARG002
        self._recording = False
        # Intentionally no-op: don't crash if video is enabled.

    def render(self) -> None:
        return

    @property
    def _in_recording(self) -> bool:
        return self._recording


class MJWarpLink(BaseLink):
    def __init__(self, *, idx: int, idx_local: int, name: str):
        self._idx = int(idx)
        self._idx_local = int(idx_local)
        self._name = str(name)

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def idx_local(self) -> int:
        return self._idx_local

    @property
    def name(self) -> str:
        return self._name


class MJWarpJoint(BaseJoint):
    def __init__(
        self,
        *,
        name: str,
        dofs_idx_local: List[int],
        dofs_limit: List[Tuple[float, float]],
    ):
        self._name = str(name)
        self._dofs_idx_local = [int(x) for x in dofs_idx_local]
        self._dofs_limit = [(float(a), float(b)) for (a, b) in dofs_limit]

    @property
    def dofs_idx(self) -> List[int]:
        # The codebase expects indices local to the entity.
        return list(self._dofs_idx_local)

    @property
    def dofs_idx_local(self) -> List[int]:
        return list(self._dofs_idx_local)

    @property
    def dofs_limit(self) -> List[Tuple[float, float]]:
        return list(self._dofs_limit)

    @property
    def name(self) -> str:
        return self._name


class MJWarpEntity(BaseEntity):
    def __init__(
        self,
        scene: "MJWarpScene",
        name: str,
        *,
        body_ids: List[int],
        joint_ids: List[int],
    ):
        self._scene = scene
        self._name = name
        self._name_prefix: Optional[str] = None
        self._root_body_name: Optional[str] = None
        self._morph_file: Optional[str] = None
        self._body_ids = [int(b) for b in body_ids]
        self._body_id_set = set(self._body_ids)
        self._joint_ids = [int(j) for j in joint_ids]

        self._links: List[MJWarpLink] = []
        self._joints: List[MJWarpJoint] = []

        self._bodyid_to_link_local: Dict[int, int] = {}
        self._geom_ids: List[int] = []

        self._dof_ids: List[int] = []
        self._dofid_to_local: Dict[int, int] = {}
        self._segments: List[_DofQposSegment] = []
        self._qpos_indices: List[int] = []
        self._scalar_local_dof_to_qpos: Dict[int, int] = {}

        # Per-DOF limits for scalar joints (torch tensors, allocated on finalize).
        self._dof_limit_lo: Optional[torch.Tensor] = None
        self._dof_limit_hi: Optional[torch.Tensor] = None

        self._kp: Optional[torch.Tensor] = None
        self._kv: Optional[torch.Tensor] = None
        self._pos_target: Optional[torch.Tensor] = None

        # Pre-build metadata (so callers can inspect links/joints before scene.build()).
        self._expected_link_names: Optional[List[str]] = None
        self._expected_joint_names: Optional[List[str]] = None
        # Desired (client-facing) joint order for scalar joints, matching KinCharModel/Genesis.
        # Excludes the root ("root") and excludes the floating base joint.
        self._kin_joint_order: Optional[List[str]] = None

    def _apply_kin_order_to_primed_metadata(self) -> None:
        if not self._kin_joint_order:
            return
        if not self._joints or not self._segments or not self._dof_ids:
            return

        # Build joint name -> mujo local dof idx for scalar joints.
        name_to_mjc_dof: Dict[str, int] = {}
        for j in self._joints:
            if not j.dofs_idx_local:
                continue
            # In the primed model, scalar joints are 1-DOF.
            if len(j.dofs_idx_local) == 1 and j.name not in (
                "floating_base_joint",
                "root_joint",
            ):
                name_to_mjc_dof[j.name] = int(j.dofs_idx_local[0])

        # Only reorder when we have a complete mapping.
        if not all(n in name_to_mjc_dof for n in self._kin_joint_order):
            return

        # Client DOF indices: base [0..5] + scalar joints in kin order.
        kin_to_client: Dict[int, int] = {}
        client_scalar_qpos: Dict[int, int] = {}
        for i, name in enumerate(self._kin_joint_order):
            client_idx = 6 + i
            kin_to_client[name_to_mjc_dof[name]] = client_idx
            qposadr = self._scalar_local_dof_to_qpos.get(name_to_mjc_dof[name])
            if qposadr is None:
                return
            client_scalar_qpos[client_idx] = int(qposadr)

        # Remap scalar joint wrapper indices.
        remapped_joints: List[MJWarpJoint] = []
        for j in self._joints:
            if j.name in name_to_mjc_dof:
                remapped_joints.append(
                    MJWarpJoint(
                        name=j.name,
                        dofs_idx_local=[kin_to_client[name_to_mjc_dof[j.name]]],
                        dofs_limit=j.dofs_limit,
                    )
                )
            else:
                # floating base joint or non-scalar joints keep their indices
                remapped_joints.append(j)
        self._joints = remapped_joints

        # Remap segments to use client dof indices for scalar joints.
        remapped_segments: List[_DofQposSegment] = []
        for seg in self._segments:
            if seg.kind == "scalar":
                mjc_idx = int(seg.dof_local_adr)
                if mjc_idx in kin_to_client:
                    remapped_segments.append(
                        _DofQposSegment(
                            kind=seg.kind,
                            dof_local_adr=int(kin_to_client[mjc_idx]),
                            dof_local_num=seg.dof_local_num,
                            qpos_adr=seg.qpos_adr,
                            qpos_num=seg.qpos_num,
                        )
                    )
                else:
                    remapped_segments.append(seg)
            else:
                remapped_segments.append(seg)
        self._segments = remapped_segments

        # Update scalar fast-path mapping.
        self._scalar_local_dof_to_qpos = dict(client_scalar_qpos)

    def _prime_from_child_model(self, mjm_child: mujoco.MjModel) -> None:
        """Populate links/joints/DOF metadata from an isolated model compile.

        This is intentionally lightweight: global ids are unknown until the full
        scene is compiled, so we use placeholder global ids and only expose names,
        limits and local indices.
        """

        # Links from child bodies (exclude world body 0).
        body_names = [mjm_child.body(i).name for i in range(1, int(mjm_child.nbody))]
        self._expected_link_names = body_names
        self._links = [
            MJWarpLink(idx=-1, idx_local=i, name=n) for i, n in enumerate(body_names)
        ]

        # DOFs in child model are contiguous [0, nv).
        self._dof_ids = list(range(int(mjm_child.nv)))
        self._dofid_to_local = {int(d): int(d) for d in self._dof_ids}

        # Joint wrappers + mapping segments for set_qpos ordering.
        joint_names: List[str] = []
        joints: List[MJWarpJoint] = []
        segments: List[_DofQposSegment] = []

        for jid in range(int(mjm_child.njnt)):
            jview = mjm_child.joint(jid)
            jname = jview.name
            joint_names.append(jname)

            jtype = int(mjm_child.jnt_type[jid])
            dofadr = int(mjm_child.jnt_dofadr[jid])
            qposadr = int(mjm_child.jnt_qposadr[jid])

            if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
                dofnum, qposnum, kind = 6, 7, "free"
            elif jtype == int(mujoco.mjtJoint.mjJNT_BALL):
                dofnum, qposnum, kind = 3, 4, "ball"
            else:
                dofnum, qposnum, kind = 1, 1, "scalar"

            local_dofs = list(range(dofadr, dofadr + dofnum))

            if bool(jview.limited[0]) and kind == "scalar":
                lo, hi = float(jview.range[0]), float(jview.range[1])
                limits = [(lo, hi)]
            else:
                limits = [(-float("inf"), float("inf")) for _ in range(dofnum)]

            joints.append(
                MJWarpJoint(name=jname, dofs_idx_local=local_dofs, dofs_limit=limits)
            )
            segments.append(
                _DofQposSegment(
                    kind=kind,
                    dof_local_adr=min(local_dofs) if local_dofs else 0,
                    dof_local_num=dofnum,
                    qpos_adr=qposadr,
                    qpos_num=qposnum,
                )
            )

        self._expected_joint_names = joint_names
        self._joints = joints
        self._segments = sorted(segments, key=lambda s: int(s.qpos_adr))

        self._scalar_local_dof_to_qpos = {}
        for seg in self._segments:
            if seg.kind == "scalar":
                # Scalar joints map 1:1 between dof position and qpos.
                self._scalar_local_dof_to_qpos[int(seg.dof_local_adr)] = int(
                    seg.qpos_adr
                )

        # Reorder scalar joint DOFs to match KinCharModel/Genesis ordering.
        self._apply_kin_order_to_primed_metadata()

        # Build qpos indices in the ordering expected by this codebase:
        # root pos (3) + root quat (4) + scalar joints (KinChar/Genesis order).
        qpos_indices: List[int] = []
        free = next((s for s in self._segments if s.kind == "free"), None)
        if free is not None:
            qpos_indices.extend(list(range(int(free.qpos_adr), int(free.qpos_adr + 7))))
        for dof_idx in sorted(self._scalar_local_dof_to_qpos.keys()):
            qpos_indices.append(int(self._scalar_local_dof_to_qpos[dof_idx]))
        self._qpos_indices = qpos_indices

    def _finalize_from_model(self) -> None:
        mjm = self._scene._mjm
        if mjm is None:
            raise RuntimeError("Scene not built")

        # Reset derived mappings (build() may re-run in some workflows).
        self._bodyid_to_link_local = {}

        # Links: one per body in this entity (excluding world body 0).
        self._links = []
        body_ids = [b for b in self._body_ids if b != 0]
        # Optionally skip the per-entity root body (if present).
        if self._root_body_name is not None:
            root_id = mujoco.mj_name2id(
                mjm, mujoco.mjtObj.mjOBJ_BODY, self._root_body_name
            )
            if root_id != -1:
                body_ids = [b for b in body_ids if b != int(root_id)]

        # If the entity has bodies but we filtered everything out, fall back.
        if not body_ids and self._body_ids:
            body_ids = [b for b in self._body_ids if b != 0]

        # Preserve stable local ordering if we were primed from an isolated compile.
        stripped_to_bid: Dict[str, int] = {}
        for bid in body_ids:
            nm = mjm.body(bid).name
            if self._name_prefix and nm.startswith(self._name_prefix):
                nm = nm[len(self._name_prefix) :]
            stripped_to_bid[nm] = int(bid)

        ordered_names = self._expected_link_names
        if ordered_names:
            ordered_bids = [
                stripped_to_bid[n] for n in ordered_names if n in stripped_to_bid
            ]
        else:
            ordered_bids = sorted(body_ids)

        for i, bid in enumerate(ordered_bids):
            name = mjm.body(bid).name
            if self._name_prefix and name.startswith(self._name_prefix):
                name = name[len(self._name_prefix) :]
            self._links.append(MJWarpLink(idx=int(bid), idx_local=i, name=name))
            self._bodyid_to_link_local[int(bid)] = int(i)

        # DOFs: all dofs belonging to bodies in the entity.
        # NOTE: We expose DOFs in a client-facing order matching KinCharModel/Genesis,
        # which is a breadth-first traversal of the kinematic tree. MuJoCo's internal
        # dof indexing can differ from that order.
        dof_ids_mjc = [
            int(i) for i, b in enumerate(mjm.dof_bodyid) if int(b) in self._body_id_set
        ]
        dof_ids_mjc = sorted(dof_ids_mjc)

        # Geoms for AABB and contacts
        geom_ids = [
            int(i) for i, b in enumerate(mjm.geom_bodyid) if int(b) in self._body_id_set
        ]
        self._geom_ids = sorted(geom_ids)

        # Build joint list and dof->qpos conversion segments.
        joint_ids_sorted = sorted(self._joint_ids, key=lambda j: int(mjm.jnt_dofadr[j]))
        self._segments = []

        # Joint wrappers (limits are used for action bounds).
        joints_by_name: Dict[str, MJWarpJoint] = {}
        segs_by_name: Dict[str, _DofQposSegment] = {}
        scalar_name_to_global_dof: Dict[str, int] = {}
        scalar_name_to_qposadr: Dict[str, int] = {}
        base_global_dofs: Optional[List[int]] = None
        for jid in joint_ids_sorted:
            jview = mjm.joint(jid)
            jname = jview.name
            if self._name_prefix and jname.startswith(self._name_prefix):
                jname = jname[len(self._name_prefix) :]
            jtype = int(mjm.jnt_type[jid])
            dofadr = int(mjm.jnt_dofadr[jid])
            qposadr = int(mjm.jnt_qposadr[jid])

            if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
                dofnum, qposnum, kind = 6, 7, "free"
                base_global_dofs = list(range(dofadr, dofadr + 6))
            elif jtype == int(mujoco.mjtJoint.mjJNT_BALL):
                dofnum, qposnum, kind = 3, 4, "ball"
            else:
                dofnum, qposnum, kind = 1, 1, "scalar"

            # Local DOF indices are assigned later once we pick client-facing order.
            global_dofs = list(range(dofadr, dofadr + dofnum))
            local_dofs_placeholder = list(global_dofs)

            # Limits
            if bool(jview.limited[0]) and kind == "scalar":
                lo, hi = float(jview.range[0]), float(jview.range[1])
                limits = [(lo, hi)]
            else:
                limits = [(-float("inf"), float("inf")) for _ in range(dofnum)]

            joints_by_name[jname] = MJWarpJoint(
                name=jname, dofs_idx_local=local_dofs_placeholder, dofs_limit=limits
            )

            if kind == "scalar":
                scalar_name_to_global_dof[jname] = int(dofadr)
                scalar_name_to_qposadr[jname] = int(qposadr)

            # Segment to map dof positions to qpos (dof_local_adr filled later).
            segs_by_name[jname] = _DofQposSegment(
                kind=kind,
                dof_local_adr=0,
                dof_local_num=dofnum,
                qpos_adr=qposadr,
                qpos_num=qposnum,
            )

        # Preserve stable local ordering if we were primed from an isolated compile.
        ordered_joint_names = self._expected_joint_names
        if ordered_joint_names:
            self._joints = [
                joints_by_name[n] for n in ordered_joint_names if n in joints_by_name
            ]
        else:
            # Fallback: dof-adr ordering.
            self._joints = list(joints_by_name.values())

        # Determine the client-facing scalar joint ordering.
        kin_order = self._kin_joint_order
        if (
            kin_order
            and all(n in scalar_name_to_global_dof for n in kin_order)
            and base_global_dofs is not None
        ):
            client_global_dof_ids = list(base_global_dofs) + [
                scalar_name_to_global_dof[n] for n in kin_order
            ]
        else:
            # Fallback to MuJoCo dof order within this entity.
            client_global_dof_ids = list(dof_ids_mjc)

        self._dof_ids = client_global_dof_ids
        self._dofid_to_local = {int(d): int(i) for i, d in enumerate(self._dof_ids)}

        # Build segments with correct client dof indices.
        segments: List[_DofQposSegment] = []
        for jname, seg in segs_by_name.items():
            if seg.kind == "free":
                segments.append(
                    _DofQposSegment(
                        kind="free",
                        dof_local_adr=0,
                        dof_local_num=6,
                        qpos_adr=seg.qpos_adr,
                        qpos_num=seg.qpos_num,
                    )
                )
            elif seg.kind == "scalar":
                gd = scalar_name_to_global_dof.get(jname)
                if gd is None or gd not in self._dofid_to_local:
                    continue
                segments.append(
                    _DofQposSegment(
                        kind="scalar",
                        dof_local_adr=int(self._dofid_to_local[gd]),
                        dof_local_num=1,
                        qpos_adr=seg.qpos_adr,
                        qpos_num=seg.qpos_num,
                    )
                )
            else:
                # This codebase expects scalar joints only.
                continue

        # Keep segments sorted by qpos address for set_qpos mapping.
        self._segments = sorted(segments, key=lambda s: int(s.qpos_adr))

        # Fast mapping for scalar joints: client local dof index -> qpos index.
        self._scalar_local_dof_to_qpos = {}
        for seg in self._segments:
            if seg.kind == "scalar":
                self._scalar_local_dof_to_qpos[int(seg.dof_local_adr)] = int(
                    seg.qpos_adr
                )

        # Build qpos indices in the ordering expected by this codebase:
        # root pos (3) + root quat (4) + scalar joints (client DOF order).
        qpos_indices: List[int] = []
        free = next((s for s in self._segments if s.kind == "free"), None)
        if free is not None:
            qpos_indices.extend(list(range(int(free.qpos_adr), int(free.qpos_adr + 7))))
        # Append scalar joints in ascending local DOF index (matches client order).
        for dof_idx in range(6, int(self.n_dofs)):
            qadr = self._scalar_local_dof_to_qpos.get(int(dof_idx))
            if qadr is not None:
                qpos_indices.append(int(qadr))
        self._qpos_indices = qpos_indices

        # Rewrite joint wrapper indices to client local indices.
        global_to_client = self._dofid_to_local
        remapped: List[MJWarpJoint] = []
        for j in self._joints:
            if j.name in scalar_name_to_global_dof:
                gd = scalar_name_to_global_dof[j.name]
                if gd in global_to_client:
                    remapped.append(
                        MJWarpJoint(
                            name=j.name,
                            dofs_idx_local=[int(global_to_client[gd])],
                            dofs_limit=j.dofs_limit,
                        )
                    )
                else:
                    remapped.append(j)
            elif j.name in ("floating_base_joint", "root_joint"):
                remapped.append(
                    MJWarpJoint(
                        name=j.name,
                        dofs_idx_local=list(range(0, 6)),
                        dofs_limit=j.dofs_limit,
                    )
                )
            else:
                remapped.append(j)
        self._joints = remapped

        # Build per-DOF position limits (used to clamp PD targets).
        n = int(self.n_dofs)
        limit_lo = torch.full(
            (n,),
            -float("inf"),
            device=self._scene._engine.device,
            dtype=self._scene._engine.tc_float,
        )
        limit_hi = torch.full(
            (n,),
            float("inf"),
            device=self._scene._engine.device,
            dtype=self._scene._engine.tc_float,
        )
        for j in self._joints:
            for dof_idx, (lo, hi) in zip(j.dofs_idx_local, j.dofs_limit):
                di = int(dof_idx)
                if 0 <= di < n:
                    limit_lo[di] = float(lo)
                    limit_hi[di] = float(hi)
        self._dof_limit_lo = limit_lo
        self._dof_limit_hi = limit_hi

        # Initialize gains/targets
        n = self.n_dofs
        dev = self._scene._engine.device
        dtype = self._scene._engine.tc_float
        self._kp = torch.zeros((n,), device=dev, dtype=dtype)
        self._kv = torch.zeros((n,), device=dev, dtype=dtype)
        self._pos_target = torch.zeros(
            (self._scene._nworld, n), device=dev, dtype=dtype
        )
        self._pos_target[:] = self.get_dofs_position()

    # ========== Position and Orientation ==========
    def get_pos(self) -> torch.Tensor:
        qpos = self._scene._torch_qpos()
        if not self._segments:
            return torch.zeros(
                (self._scene._nworld, 3),
                device=self._scene._engine.device,
                dtype=self._scene._engine.tc_float,
            )

        free = next((s for s in self._segments if s.kind == "free"), None)
        if free is None:
            return torch.zeros(
                (self._scene._nworld, 3), device=qpos.device, dtype=qpos.dtype
            )
        return qpos[:, free.qpos_adr : free.qpos_adr + 3]

    def set_pos(
        self, pos: torch.Tensor, envs_idx: Optional[torch.Tensor] = None
    ) -> None:
        qpos = self._scene._torch_qpos()
        free = next((s for s in self._segments if s.kind == "free"), None)
        if free is None:
            return
        if envs_idx is None:
            qpos[:, free.qpos_adr : free.qpos_adr + 3] = pos
        else:
            qpos[envs_idx, free.qpos_adr : free.qpos_adr + 3] = pos
        self._scene._mark_dirty()

    def get_quat(self) -> torch.Tensor:
        qpos = self._scene._torch_qpos()
        free = next((s for s in self._segments if s.kind == "free"), None)
        if free is None:
            quat = torch.zeros(
                (self._scene._nworld, 4), device=qpos.device, dtype=qpos.dtype
            )
            quat[:, 0] = 1.0
            return quat
        return qpos[:, free.qpos_adr + 3 : free.qpos_adr + 7]

    def set_quat(
        self, quat: torch.Tensor, envs_idx: Optional[torch.Tensor] = None
    ) -> None:
        qpos = self._scene._torch_qpos()
        free = next((s for s in self._segments if s.kind == "free"), None)
        if free is None:
            return
        if envs_idx is None:
            qpos[:, free.qpos_adr + 3 : free.qpos_adr + 7] = quat
        else:
            qpos[envs_idx, free.qpos_adr + 3 : free.qpos_adr + 7] = quat
        self._scene._mark_dirty()

    # ========== Velocities ==========
    def get_vel(self) -> torch.Tensor:
        qvel = self._scene._torch_qvel()
        free = next((s for s in self._segments if s.kind == "free"), None)
        if free is None:
            return torch.zeros(
                (self._scene._nworld, 3), device=qvel.device, dtype=qvel.dtype
            )
        dof_ids = self._dof_ids
        if len(dof_ids) < 6:
            return torch.zeros(
                (self._scene._nworld, 3), device=qvel.device, dtype=qvel.dtype
            )
        base = qvel[:, dof_ids[0:3]]
        return base

    def get_ang(self) -> torch.Tensor:
        qvel = self._scene._torch_qvel()
        free = next((s for s in self._segments if s.kind == "free"), None)
        if free is None:
            return torch.zeros(
                (self._scene._nworld, 3), device=qvel.device, dtype=qvel.dtype
            )
        dof_ids = self._dof_ids
        if len(dof_ids) < 6:
            return torch.zeros(
                (self._scene._nworld, 3), device=qvel.device, dtype=qvel.dtype
            )
        ang = qvel[:, dof_ids[3:6]]
        return ang

    # ========== Degrees of Freedom ==========
    def get_dofs_position(self) -> torch.Tensor:
        qpos = self._scene._torch_qpos()
        out = torch.zeros(
            (self._scene._nworld, self.n_dofs), device=qpos.device, dtype=qpos.dtype
        )
        for seg in self._segments:
            if seg.kind == "free":
                # Base orientation is represented as a quaternion via get_quat()/set_quat()/set_qpos().
                # The DOF vector keeps the conventional 6D base indexing (3 linear + 3 angular).
                out[:, seg.dof_local_adr : seg.dof_local_adr + 3] = qpos[
                    :, seg.qpos_adr : seg.qpos_adr + 3
                ]
            elif seg.kind == "ball":
                # This codebase expects scalar joints; a MuJoCo ball joint is a 3-DOF joint
                # with quaternion qpos storage. Keep zeros here and require users to use
                # set_qpos() / quaternion math if they truly need ball joints.
                continue
            else:
                out[:, seg.dof_local_adr] = qpos[:, seg.qpos_adr]
        return out

    def set_dofs_position(
        self,
        position: torch.Tensor,
        envs_idx: Optional[torch.Tensor] = None,
        dofs_idx_local: Optional[List[int]] = None,
    ) -> None:
        if dofs_idx_local is not None:
            # Common case in this codebase: only non-root hinge joints are set.
            # Avoid exp-map<->quat roundtripping of the floating base by directly
            # writing scalar joint qpos entries when possible.
            if dofs_idx_local and all(
                int(i) in self._scalar_local_dof_to_qpos for i in dofs_idx_local
            ):
                qpos = self._scene._torch_qpos()
                qidx = torch.tensor(
                    [self._scalar_local_dof_to_qpos[int(i)] for i in dofs_idx_local],
                    device=qpos.device,
                    dtype=torch.long,
                )
                if envs_idx is None:
                    qpos[:, qidx] = position
                else:
                    qpos[envs_idx[:, None], qidx] = position
                self._scene._mark_dirty()
                return

            # Fallback: update subset via read-modify-write.
            full = self.get_dofs_position()
            if envs_idx is None:
                full[:, dofs_idx_local] = position
            else:
                full[
                    envs_idx[:, None], torch.tensor(dofs_idx_local, device=full.device)
                ] = position
            position = full
            envs_idx = None

        qpos = self._scene._torch_qpos()

        for seg in self._segments:
            if seg.kind == "free":
                pos = position[:, seg.dof_local_adr : seg.dof_local_adr + 3]
                if envs_idx is None:
                    qpos[:, seg.qpos_adr : seg.qpos_adr + 3] = pos
                else:
                    qpos[envs_idx, seg.qpos_adr : seg.qpos_adr + 3] = pos
            elif seg.kind == "ball":
                raise NotImplementedError(
                    "MJWarpEntity.set_dofs_position does not support MuJoCo ball joints in this codebase. "
                    "Use set_qpos() / quaternions instead."
                )
            else:
                val = position[:, seg.dof_local_adr]
                if envs_idx is None:
                    qpos[:, seg.qpos_adr] = val
                else:
                    qpos[envs_idx, seg.qpos_adr] = val

        self._scene._mark_dirty()

    def get_dofs_velocity(self) -> torch.Tensor:
        qvel = self._scene._torch_qvel()
        return qvel[:, self._dof_ids]

    def set_dofs_velocity(
        self, velocity: torch.Tensor, envs_idx: Optional[torch.Tensor] = None
    ) -> None:
        qvel = self._scene._torch_qvel()
        if envs_idx is None:
            qvel[:, self._dof_ids] = velocity
        else:
            qvel[envs_idx, self._dof_ids] = velocity
        self._scene._mark_dirty()

    def control_dofs_position(
        self, position: torch.Tensor, dofs_idx_local: Optional[List[int]] = None
    ) -> None:
        if self._pos_target is None:
            raise RuntimeError("Entity not finalized")

        # Clamp position targets to joint limits to match Genesis-like behavior.
        # This is especially important because the policy's Gaussian output is unbounded.
        limit_margin = float(
            self._scene._engine._config.get("position_limit_margin", 1e-4)
        )
        max_target_delta = self._scene._engine._config.get("max_target_delta")
        max_target_delta = (
            float(max_target_delta) if max_target_delta is not None else None
        )

        if dofs_idx_local is None:
            idx = torch.arange(
                int(self.n_dofs), device=position.device, dtype=torch.long
            )
            pos_in = position
        else:
            idx = torch.tensor(dofs_idx_local, device=position.device, dtype=torch.long)
            pos_in = position

        pos = pos_in
        if self._dof_limit_lo is not None and self._dof_limit_hi is not None:
            lo = self._dof_limit_lo.to(device=pos.device, dtype=pos.dtype)[idx]
            hi = self._dof_limit_hi.to(device=pos.device, dtype=pos.dtype)[idx]
            finite = torch.isfinite(lo) & torch.isfinite(hi)
            if finite.any():
                lo2 = torch.where(finite, lo + limit_margin, lo)
                hi2 = torch.where(finite, hi - limit_margin, hi)
                pos = torch.maximum(torch.minimum(pos, hi2), lo2)

        if max_target_delta is not None and max_target_delta > 0:
            prev = self._pos_target[:, idx]
            pos = torch.maximum(
                torch.minimum(pos, prev + max_target_delta), prev - max_target_delta
            )

        if dofs_idx_local is None:
            self._pos_target[:, :] = pos
        else:
            self._pos_target[:, idx] = pos

    def set_dofs_kp(self, kp: torch.Tensor) -> None:
        if self._kp is None:
            raise RuntimeError("Entity not finalized")
        self._kp[:] = kp

    def set_dofs_kv(self, kv: torch.Tensor) -> None:
        if self._kv is None:
            raise RuntimeError("Entity not finalized")
        self._kv[:] = kv

    def zero_all_dofs_velocity(self, envs_idx: Optional[torch.Tensor] = None) -> None:
        zeros = torch.zeros(
            (self._scene._nworld, self.n_dofs),
            device=self._scene._engine.device,
            dtype=self._scene._engine.tc_float,
        )
        if envs_idx is None:
            self.set_dofs_velocity(zeros)
        else:
            self.set_dofs_velocity(zeros[envs_idx], envs_idx=envs_idx)

    # ========== Links ==========
    def get_links_pos(self) -> torch.Tensor:
        self._scene._ensure_kinematics()
        xpos = self._scene._torch_xpos()  # (nworld, nbody, 3)
        body_ids = [l.idx for l in self._links]
        return xpos[:, body_ids, :]

    def get_links_quat(self) -> torch.Tensor:
        self._scene._ensure_kinematics()
        xquat = self._scene._torch_xquat()  # (nworld, nbody, 4)
        body_ids = [l.idx for l in self._links]
        return xquat[:, body_ids, :]

    def get_links_net_contact_force(self) -> torch.Tensor:
        # Not currently used by termination logic (v2 uses get_contacts). Return zeros for now.
        return torch.zeros(
            (self._scene._nworld, len(self._links), 3),
            device=self._scene._engine.device,
            dtype=self._scene._engine.tc_float,
        )

    # ========== Contacts ==========
    def get_contacts(
        self, with_entity: BaseEntity, exclude_self_contact: bool = True
    ) -> Dict[str, torch.Tensor]:
        other = with_entity
        if not isinstance(other, MJWarpEntity):
            raise TypeError("MJWarpEntity.get_contacts expects another MJWarpEntity")

        self._scene._ensure_kinematics()
        mjm = self._scene._mjm
        d = self._scene._d
        if mjm is None or d is None:
            raise RuntimeError("Scene not built")

        nacon = int(wp.to_torch(d.nacon).cpu().item())
        if nacon <= 0:
            empty = torch.empty(
                (self._scene._nworld, 0),
                device=self._scene._engine.device,
                dtype=torch.long,
            )
            mask = torch.empty(
                (self._scene._nworld, 0),
                device=self._scene._engine.device,
                dtype=torch.bool,
            )
            return {"link_a": empty, "link_b": empty, "valid_mask": mask}

        geom_pairs = wp.to_torch(d.contact.geom)[:nacon].cpu().numpy().astype(np.int32)
        world_ids = (
            wp.to_torch(d.contact.worldid)[:nacon].cpu().numpy().astype(np.int32)
        )
        geom_bodyid = np.asarray(mjm.geom_bodyid, dtype=np.int32)

        per_world_a: List[List[int]] = [[] for _ in range(self._scene._nworld)]
        per_world_b: List[List[int]] = [[] for _ in range(self._scene._nworld)]

        for i in range(nacon):
            w = int(world_ids[i])
            g0, g1 = int(geom_pairs[i, 0]), int(geom_pairs[i, 1])
            if g0 < 0 or g1 < 0:
                continue
            b0, b1 = int(geom_bodyid[g0]), int(geom_bodyid[g1])

            if exclude_self_contact and (self is other):
                if b0 in self._body_id_set and b1 in self._body_id_set:
                    continue

            a_in_self = b0 in self._body_id_set
            b_in_self = b1 in self._body_id_set
            a_in_other = b0 in other._body_id_set
            b_in_other = b1 in other._body_id_set

            if a_in_self and b_in_other:
                per_world_a[w].append(b0)
                per_world_b[w].append(b1)
            elif b_in_self and a_in_other:
                per_world_a[w].append(b1)
                per_world_b[w].append(b0)

        max_len = max((len(x) for x in per_world_a), default=0)
        link_a = torch.full(
            (self._scene._nworld, max_len),
            -1,
            device=self._scene._engine.device,
            dtype=torch.long,
        )
        link_b = torch.full(
            (self._scene._nworld, max_len),
            -1,
            device=self._scene._engine.device,
            dtype=torch.long,
        )
        valid = torch.zeros(
            (self._scene._nworld, max_len),
            device=self._scene._engine.device,
            dtype=torch.bool,
        )

        for w in range(self._scene._nworld):
            n = len(per_world_a[w])
            if n == 0:
                continue
            link_a[w, :n] = torch.tensor(
                per_world_a[w], device=link_a.device, dtype=torch.long
            )
            link_b[w, :n] = torch.tensor(
                per_world_b[w], device=link_b.device, dtype=torch.long
            )
            valid[w, :n] = True

        return {"link_a": link_a, "link_b": link_b, "valid_mask": valid}

    # ========== Geometry ==========
    def get_AABB(self) -> torch.Tensor:
        self._scene._ensure_kinematics()

        if not self._geom_ids:
            zeros = torch.zeros(
                (self._scene._nworld, 2, 3),
                device=self._scene._engine.device,
                dtype=self._scene._engine.tc_float,
            )
            return zeros

        geom_ids = torch.tensor(
            self._geom_ids, device=self._scene._engine.device, dtype=torch.long
        )
        gxpos = self._scene._torch_geom_xpos()[:, geom_ids, :]  # (nworld, ngeom_e, 3)
        gxmat = self._scene._torch_geom_xmat()[
            :, geom_ids, :, :
        ]  # (nworld, ngeom_e, 3,3)

        gtype = self._scene._geom_type[geom_ids]
        gsize = self._scene._geom_size[geom_ids]
        grb = self._scene._geom_rbound[geom_ids]

        ext = torch.zeros_like(gsize)

        # Sphere
        mask = gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        if mask.any():
            r = gsize[mask, 0:1]
            ext[mask] = r.expand(-1, 3)

        # Box
        mask = gtype == int(mujoco.mjtGeom.mjGEOM_BOX)
        if mask.any():
            ext[mask] = gsize[mask]

        # Capsule
        mask = gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        if mask.any():
            r = gsize[mask, 0]
            h = gsize[mask, 1]
            ext[mask, 0] = r
            ext[mask, 1] = r
            ext[mask, 2] = r + h

        # Cylinder
        mask = gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        if mask.any():
            r = gsize[mask, 0]
            h = gsize[mask, 1]
            ext[mask, 0] = r
            ext[mask, 1] = r
            ext[mask, 2] = h

        # Ellipsoid
        mask = gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
        if mask.any():
            ext[mask] = gsize[mask]

        # Fallback for mesh/hfield/etc: bounding sphere
        mask = ext.abs().sum(dim=1) == 0
        if mask.any():
            r = grb[mask].unsqueeze(-1)
            ext[mask] = r.expand(-1, 3)

        Rabs = gxmat.abs()
        world_ext = torch.einsum("wgij,gj->wgi", Rabs, ext)
        aabb_min = (gxpos - world_ext).amin(dim=1)
        aabb_max = (gxpos + world_ext).amax(dim=1)
        return torch.stack([aabb_min, aabb_max], dim=1)

    # ========== Structure Access ==========
    def get_joint(self, name: str) -> BaseJoint:
        for j in self._joints:
            if j.name == name:
                return j
        raise KeyError(f"Joint not found: {name}")

    def get_link(self, name: str) -> BaseLink:
        for l in self._links:
            if l.name == name:
                return l
        raise KeyError(f"Link not found: {name}")

    @property
    def joints(self) -> List[BaseJoint]:
        return list(self._joints)

    @property
    def links(self) -> List[BaseLink]:
        return list(self._links)

    @property
    def n_dofs(self) -> int:
        return len(self._dof_ids)

    # ========== Special Methods ==========
    def set_qpos(
        self, qpos: torch.Tensor, envs_idx: Optional[torch.Tensor] = None
    ) -> None:
        if not self._qpos_indices:
            return
        tq = self._scene._torch_qpos()
        idx = torch.tensor(self._qpos_indices, device=tq.device, dtype=torch.long)
        if envs_idx is None:
            tq[:, idx] = qpos
        else:
            tq[envs_idx[:, None], idx] = qpos
        self._scene._mark_dirty()


class MJWarpScene(BaseScene):
    def __init__(
        self,
        engine: "MJWarpEngine",
        *,
        show_viewer: bool,
        sim_options: Dict,
        rigid_options: Dict,
        vis_options: Optional[Dict],
        viewer_options: Optional[Dict],
    ):
        self._engine = engine
        self._show_viewer = bool(show_viewer)
        self._sim_options = dict(sim_options or {})
        self._rigid_options = dict(rigid_options or {})
        self._vis_options = dict(vis_options or {}) if vis_options is not None else None
        self._viewer_options = (
            dict(viewer_options or {}) if viewer_options is not None else None
        )

        self._entities_pending: List[Dict] = []
        self._entities: List[MJWarpEntity] = []
        self._cameras: List[MJWarpCamera] = []

        self._built = False
        self._t = 0
        self._nworld = 1

        self._spec: Optional[mujoco.MjSpec] = None
        self._mjm: Optional[mujoco.MjModel] = None
        self._mjd: Optional[mujoco.MjData] = None
        self._m: Optional[mjw.Model] = None
        self._d: Optional[mjw.Data] = None
        self._dirty_kinematics = True

        self._geom_type: Optional[torch.Tensor] = None
        self._geom_size: Optional[torch.Tensor] = None
        self._geom_rbound: Optional[torch.Tensor] = None

        self._viewer = None
        self._viewer_last_sync_time: Optional[float] = None
        self._viewer_max_fps: Optional[float] = None

        # Physics substeps per control step (dt in the MJCF is divided by this).
        self._substeps: int = 1

    def _maybe_init_viewer(self) -> None:
        if not self._show_viewer:
            return
        if self._mjm is None or self._mjd is None:
            return

        # Avoid crashing in headless environments.
        if (
            os.environ.get("DISPLAY") is None
            and os.environ.get("WAYLAND_DISPLAY") is None
        ):
            msg = "MJWarp viewer requested but no DISPLAY/WAYLAND_DISPLAY set; skipping viewer"
            print(msg)
            logger.warning(msg)
            self._show_viewer = False
            return

        try:
            import mujoco.viewer  # type: ignore
        except Exception as e:
            msg = f"Failed to import mujoco.viewer; skipping viewer ({e})"
            print(msg)
            logger.warning(msg)
            self._show_viewer = False
            return

        try:
            print("Launching MuJoCo viewer (MJWarp)...")
            self._viewer = mujoco.viewer.launch_passive(self._mjm, self._mjd)
        except Exception as e:
            msg = f"Failed to launch MuJoCo viewer; skipping viewer ({e})"
            print(msg)
            logger.warning(msg)
            self._show_viewer = False
            self._viewer = None
            return

        # Optional throttling
        if self._viewer_options and "max_FPS" in self._viewer_options:
            try:
                self._viewer_max_fps = float(self._viewer_options["max_FPS"])
            except Exception:
                self._viewer_max_fps = None

        # Best-effort camera placement
        if self._viewer_options and hasattr(self._viewer, "cam"):
            cam_pos = self._viewer_options.get("camera_pos")
            cam_lookat = self._viewer_options.get("camera_lookat")
            if cam_pos is not None and cam_lookat is not None:
                try:
                    px, py, pz = [float(x) for x in cam_pos]
                    lx, ly, lz = [float(x) for x in cam_lookat]
                    vx, vy, vz = (px - lx), (py - ly), (pz - lz)
                    dist = float(math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-9)
                    azimuth = float(math.degrees(math.atan2(vy, vx)))
                    elevation = float(
                        math.degrees(math.asin(max(-1.0, min(1.0, vz / dist))))
                    )

                    self._viewer.cam.lookat[:] = [lx, ly, lz]
                    self._viewer.cam.distance = dist
                    self._viewer.cam.azimuth = azimuth
                    self._viewer.cam.elevation = elevation
                except Exception:
                    pass

    def _viewer_sync(self) -> None:
        if not self._viewer or self._mjm is None or self._mjd is None:
            return
        if hasattr(self._viewer, "is_running") and (not self._viewer.is_running()):
            return

        # Copy env0 state from MJWarp back to host mjData for visualization.
        try:
            qpos0 = self._torch_qpos()[0].detach().to("cpu").numpy()
            qvel0 = self._torch_qvel()[0].detach().to("cpu").numpy()

            # Guard against NaNs/infs (can happen during unstable rollouts).
            # Passing non-finite values into MuJoCo's mj_forward can segfault.
            if (not np.isfinite(qpos0).all()) or (not np.isfinite(qvel0).all()):
                return

            self._mjd.qpos[:] = qpos0
            self._mjd.qvel[:] = qvel0
            mujoco.mj_forward(self._mjm, self._mjd)
        except Exception:
            return

        # Throttle if requested.
        if self._viewer_max_fps and self._viewer_max_fps > 0:
            now = time.time()
            if self._viewer_last_sync_time is not None:
                dt = now - self._viewer_last_sync_time
                min_dt = 1.0 / self._viewer_max_fps
                if dt < min_dt:
                    time.sleep(min_dt - dt)
                    now = time.time()
            self._viewer_last_sync_time = now

        try:
            self._viewer.sync()
        except Exception:
            return

    def _mark_dirty(self) -> None:
        self._dirty_kinematics = True

    def _ensure_kinematics(self) -> None:
        if not self._built or self._m is None or self._d is None:
            return
        if self._dirty_kinematics:
            mjw.kinematics(self._m, self._d)
            self._dirty_kinematics = False

    # torch views into warp arrays
    def _torch_qpos(self) -> torch.Tensor:
        if self._d is None:
            raise RuntimeError("Scene not built")
        return wp.to_torch(self._d.qpos)

    def _torch_qvel(self) -> torch.Tensor:
        if self._d is None:
            raise RuntimeError("Scene not built")
        return wp.to_torch(self._d.qvel)

    def _torch_xpos(self) -> torch.Tensor:
        if self._d is None:
            raise RuntimeError("Scene not built")
        return wp.to_torch(self._d.xpos)

    def _torch_xquat(self) -> torch.Tensor:
        if self._d is None:
            raise RuntimeError("Scene not built")
        return wp.to_torch(self._d.xquat)

    def _torch_geom_xpos(self) -> torch.Tensor:
        if self._d is None:
            raise RuntimeError("Scene not built")
        return wp.to_torch(self._d.geom_xpos)

    def _torch_geom_xmat(self) -> torch.Tensor:
        if self._d is None:
            raise RuntimeError("Scene not built")
        return wp.to_torch(self._d.geom_xmat)

    def add_entity(
        self,
        morph_type: str,
        morph_file: Optional[str] = None,
        morph_pos: Optional[Tuple[float, float, float]] = None,
        morph_quat: Optional[Tuple[float, float, float, float]] = None,
        material_type: str = "rigid",
        visualize_contact: bool = True,
    ) -> BaseEntity:
        if self._built:
            raise RuntimeError("Cannot add entities after build()")
        if material_type != "rigid":
            raise NotImplementedError(
                "MJWarp backend currently supports rigid entities only"
            )

        morph_pos = morph_pos or (0.0, 0.0, 0.0)
        morph_quat = morph_quat or (1.0, 0.0, 0.0, 0.0)

        entry = {
            "morph_type": morph_type,
            "morph_file": morph_file,
            "pos": tuple(float(x) for x in morph_pos),
            "quat": tuple(float(x) for x in morph_quat),
            "visualize_contact": bool(visualize_contact),
        }
        self._entities_pending.append(entry)

        # Placeholder entity; finalized in build.
        entity = MJWarpEntity(
            self,
            name=f"entity_{len(self._entities_pending)-1}",
            body_ids=[],
            joint_ids=[],
        )

        # Prime link/joint metadata early so callers can inspect entity.links/joints
        # before build() (the training code does this when building lookups/limits).
        if entry["morph_type"] != "plane" and entry["morph_file"]:
            try:
                path = _resolve_asset_path(str(entry["morph_file"]))
                entry["morph_file"] = path
                entity._morph_file = path

                # Match Genesis/KinCharModel DOF ordering (breadth-first over bodies).
                try:
                    from add_gym.anim import kin_char_model as _kin_char_model

                    kcm = _kin_char_model.KinCharModel("cpu")
                    kcm.load_char_file(path)
                    # Exclude the root joint name "root".
                    entity._kin_joint_order = kcm.get_joint_order()[1:]
                except Exception:
                    entity._kin_joint_order = None

                child = mujoco.MjSpec.from_file(path)
                child.modelfiledir = str(Path(path).resolve().parent)
                mjm_child = child.compile()
                entity._prime_from_child_model(mjm_child)
            except Exception:
                # Best-effort: keep placeholder empty if priming fails.
                pass

        self._entities.append(entity)
        return entity

    def add_camera(
        self,
        res: Tuple[int, int],
        pos: Tuple[float, float, float],
        lookat: Tuple[float, float, float],
        fov: float,
    ) -> BaseCamera:
        cam = MJWarpCamera()
        self._cameras.append(cam)
        return cam

    def build(
        self, n_envs: int, env_spacing: Tuple[float, float] = (0.0, 0.0)
    ) -> None:  # noqa: ARG002
        if self._built:
            return

        self._nworld = int(n_envs)
        if self._nworld < 1:
            raise ValueError("n_envs must be >= 1")

        spec = mujoco.MjSpec()

        ctrl_dt = float(
            self._sim_options.get("dt", self._engine._config.get("ctrl_dt", 0.01))
        )
        substeps = self._engine._config.get("substeps", 1)
        try:
            substeps = int(substeps)
        except Exception:
            substeps = 1
        if substeps < 1:
            substeps = 1
        self._substeps = substeps

        # Use a smaller MuJoCo timestep and take multiple physics substeps per control step.
        # This improves stability without changing the control frequency.
        spec.option.timestep = ctrl_dt / float(substeps)

        # Optional integrator selection via config (best-effort).
        integrator = self._engine._config.get("integrator")
        if integrator is not None:
            integrator_map = {
                "euler": mujoco.mjtIntegrator.mjINT_EULER,
                "rk4": mujoco.mjtIntegrator.mjINT_RK4,
                "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            }
            if isinstance(integrator, str) and integrator.lower() in integrator_map:
                spec.option.integrator = integrator_map[integrator.lower()]

        # Optional solver selection/iterations (best-effort).
        # NOTE: mujoco_warp currently does not support all MuJoCo solver types.
        # In particular, mjSOL_PGS (0) will raise in put_model().
        solver = self._engine._config.get("contact_model")
        if solver is None:
            solver = self._rigid_options.get("constraint_solver")
        if isinstance(solver, str):
            s = solver.strip().lower()
            if s == "newton":
                spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
            elif s == "pgs":
                # Keep compatibility with older configs by mapping to NEWTON.
                spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON

        iters = self._engine._config.get("solver_iterations")
        if iters is not None:
            try:
                spec.option.iterations = int(iters)
            except Exception:
                pass

        # Add a ground body if any plane entities exist.
        plane_indices = [
            i
            for i, e in enumerate(self._entities_pending)
            if e["morph_type"] == "plane"
        ]
        ground_bodyid: Optional[int] = None
        if plane_indices:
            ground = spec.worldbody.add_body(name="ground")
            g = ground.add_geom(
                type=mujoco.mjtGeom.mjGEOM_PLANE, size=[100.0, 100.0, 0.1]
            )
            g.friction = [1.0, 0.005, 0.0001]

        # Attach other entities on frames.
        attach_info: List[Tuple[int, str]] = []  # (entity_index, prefix)
        for i, entry in enumerate(self._entities_pending):
            if entry["morph_type"] == "plane":
                continue
            mtype = entry["morph_type"]
            mfile = entry["morph_file"]
            if not mfile:
                raise ValueError(f"morph_file required for morph_type={mtype}")
            path = _resolve_asset_path(mfile)
            child = mujoco.MjSpec.from_file(path)
            child.modelfiledir = str(Path(path).resolve().parent)

            # Disable collisions for purely-visual entities.
            if not entry["visualize_contact"]:
                for geom in child.geoms:
                    geom.contype = 0
                    geom.conaffinity = 0

            frame = spec.worldbody.add_frame(
                pos=list(entry["pos"]), quat=list(entry["quat"])
            )
            prefix = f"e{i}_"
            spec.attach(child, frame=frame, prefix=prefix)
            attach_info.append((i, prefix))

        mjm = spec.compile()
        mjd = mujoco.MjData(mjm)

        # Put model and data on Warp device.
        m = mjw.put_model(mjm)

        # Allocate contact/constraint workspaces generously by default.
        # These can be overridden via engine config.
        # NOTE: Contact-heavy scenes (e.g., humanoids + ground + self-collisions)
        # can easily exceed small defaults.
        nconmax = int(self._engine._config.get("nconmax", 2048))
        njmax = int(self._engine._config.get("njmax", 2048))
        naconmax_cfg = self._engine._config.get("naconmax")
        naconmax = int(naconmax_cfg) if naconmax_cfg is not None else None

        d = mjw.put_data(
            mjm,
            mjd,
            nworld=self._nworld,
            nconmax=nconmax,
            njmax=njmax,
            naconmax=naconmax,
        )

        self._spec = spec
        self._mjm = mjm
        self._mjd = mjd
        self._m = m
        self._d = d
        self._built = True
        self._dirty_kinematics = True

        # Optional: initialize viewer after model/data exist.
        self._maybe_init_viewer()

        # Cache geom metadata tensors (device-side) for AABB.
        dev = self._engine.device
        dtype = self._engine.tc_float
        self._geom_type = torch.tensor(
            np.asarray(mjm.geom_type, dtype=np.int32), device=dev, dtype=torch.int32
        )
        self._geom_size = torch.tensor(
            np.asarray(mjm.geom_size, dtype=np.float32), device=dev, dtype=dtype
        )
        self._geom_rbound = torch.tensor(
            np.asarray(mjm.geom_rbound, dtype=np.float32), device=dev, dtype=dtype
        )

        # Finalize entity mappings.
        for idx, entry in enumerate(self._entities_pending):
            ent = self._entities[idx]
            if entry["morph_type"] == "plane":
                # plane entity maps to ground body
                ground_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "ground")
                ent._body_ids = [ground_id]
                ent._body_id_set = {ground_id}
                ent._joint_ids = []
            else:
                prefix = next((p for (i, p) in attach_info if i == idx), None)
                if prefix is None:
                    raise RuntimeError("Missing attach info")

                body_ids = [
                    int(i)
                    for i in range(mjm.nbody)
                    if mjm.body(i).name.startswith(prefix)
                ]
                body_id_set = set(body_ids)
                joint_ids = [
                    int(j)
                    for j, b in enumerate(mjm.jnt_bodyid)
                    if int(b) in body_id_set
                ]

                ent._name_prefix = prefix
                ent._root_body_name = None
                ent._body_ids = body_ids
                ent._body_id_set = body_id_set
                ent._joint_ids = joint_ids

            ent._finalize_from_model()

        # Run an initial kinematics pass so getters and AABB work immediately.
        self._ensure_kinematics()

    def step(self) -> None:
        if not self._built or self._m is None or self._d is None:
            raise RuntimeError("Scene not built")

        max_torque = self._engine._config.get("max_torque")
        max_torque = float(max_torque) if max_torque is not None else None

        substeps = int(getattr(self, "_substeps", 1))
        if substeps < 1:
            substeps = 1

        for _ in range(substeps):
            # Apply PD control as generalized forces.
            qfrc = wp.to_torch(self._d.qfrc_applied)
            qfrc.zero_()

            for ent in self._entities:
                if ent._kp is None or ent._kv is None or ent._pos_target is None:
                    continue
                if ent.n_dofs == 0:
                    continue

                kp = ent._kp
                kv = ent._kv
                if (
                    float(kp.abs().max().item()) == 0.0
                    and float(kv.abs().max().item()) == 0.0
                ):
                    continue

                pos = ent.get_dofs_position()
                vel = ent.get_dofs_velocity()
                tgt = ent._pos_target
                tau_local = kp.unsqueeze(0) * (tgt - pos) - kv.unsqueeze(0) * vel

                if max_torque is not None and max_torque > 0:
                    tau_local = torch.clamp(tau_local, -max_torque, max_torque)

                # Only apply to dofs with nonzero gains.
                mask = (kp != 0) | (kv != 0)
                # Never apply PD forces to the floating base.
                # In this codebase the first 6 DOFs correspond to base linear/angular motion.
                if mask.numel() >= 6:
                    mask[:6] = False
                if mask.any():
                    dof_ids = torch.tensor(
                        ent._dof_ids, device=qfrc.device, dtype=torch.long
                    )
                    qfrc[:, dof_ids[mask]] += tau_local[:, mask]

            mjw.step(self._m, self._d)

        self._t += 1
        self._dirty_kinematics = False

        # Viewer: sync env0 for visualization.
        if self._show_viewer:
            self._viewer_sync()

    @property
    def t(self) -> int:
        return self._t


class MJWarpEngine(BaseEngine):
    def __init__(self, **kwargs):
        self._config = dict(kwargs)
        self._device: Optional[torch.device] = None
        self._tc_float: Optional[torch.dtype] = None
        self._warp_device: Optional[str] = None

    def init(self, backend: str, precision: str) -> None:
        backend = str(backend).lower()
        precision = str(precision)

        if precision != "32":
            raise NotImplementedError(
                "MJWarp backend currently supports precision='32' only"
            )

        if backend == "gpu":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._warp_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif backend == "cpu":
            self._device = torch.device("cpu")
            self._warp_device = "cpu"
        elif backend == "warp":
            # Treat as GPU if available.
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._warp_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._tc_float = torch.float32

        wp.init()
        wp.set_device(self._warp_device)

    def create_scene(
        self,
        show_viewer: bool,
        sim_options: Dict,
        rigid_options: Dict,
        vis_options: Optional[Dict] = None,
        viewer_options: Optional[Dict] = None,
    ) -> BaseScene:
        if self._device is None:
            raise RuntimeError("Engine not initialized. Call init() first.")
        return MJWarpScene(
            self,
            show_viewer=show_viewer,
            sim_options=sim_options,
            rigid_options=rigid_options,
            vis_options=vis_options,
            viewer_options=viewer_options,
        )

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("Engine not initialized. Call init() first.")
        return self._device

    @property
    def tc_float(self) -> torch.dtype:
        if self._tc_float is None:
            raise RuntimeError("Engine not initialized. Call init() first.")
        return self._tc_float
