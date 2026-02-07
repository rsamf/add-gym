import torch
import torch.nn.functional as F


class AdaptiveSegmentSampler:
    def __init__(self, clip_lengths, dt, num_segments=20, temperature=None, min_start_time=0.0):
        self.num_segments = num_segments
        self.dt = dt
        self.temperature = temperature
        self.min_start_time = min_start_time

        # Calculate segment size for each clip (treating clip_lengths as seconds)
        self.segment_sizes = torch.tensor(
            [clip_length / num_segments for clip_length in clip_lengths]
        )  # (num_clips,)

        num_clips = len(clip_lengths)
        self.errors = torch.ones((num_clips, num_segments))

    @torch.no_grad()
    def update_errors(self, clip_ids, timesteps, tracking_errors):
        """Update difficulty estimates from rollout data."""
        if self.errors.device != tracking_errors.device:
            self.errors = self.errors.to(tracking_errors.device)
            self.segment_sizes = self.segment_sizes.to(tracking_errors.device)

        seg_sizes = self.segment_sizes[clip_ids]
        # Avoid division by zero
        seg_sizes = torch.clamp(seg_sizes, min=1e-6)

        # Calculate segment index
        seg_idxs = (timesteps / seg_sizes).long()
        seg_idxs = torch.clamp(seg_idxs, 0, self.num_segments - 1)

        # Flatten (clip_id, seg_idx) into 1D index for scatter_reduce
        flat_idx = clip_ids * self.num_segments + seg_idxs

        # Compute per-segment mean of tracking errors
        mean_errors = torch.zeros(
            self.errors.numel(), device=self.errors.device, dtype=self.errors.dtype
        ).scatter_reduce(
            0, flat_idx, tracking_errors, reduce="mean", include_self=False
        ).view_as(self.errors)

        # Identify which segments received updates
        update_mask = torch.zeros(
            self.errors.numel(), device=self.errors.device, dtype=self.errors.dtype
        ).scatter_add(
            0, flat_idx, torch.ones_like(tracking_errors)
        ).view_as(self.errors) > 0

        # EMA blend only updated segments
        self.errors = torch.where(
            update_mask, 0.9 * self.errors + 0.1 * mean_errors, self.errors
        )

    def get_probs(self, clip_ids=None):
        """Get sampling probabilities for segments of given clips."""
        if clip_ids is None:
            clip_ids = torch.arange(self.errors.shape[0], device=self.errors.device)

        if self.errors.device != clip_ids.device:
            self.errors = self.errors.to(clip_ids.device)
            self.segment_sizes = self.segment_sizes.to(clip_ids.device)

        clip_errors = self.errors[clip_ids]
        if self.temperature is None:
            temperature = torch.max(clip_errors) + 1e-6
        else: 
            temperature = self.temperature

        probs = F.softmax(clip_errors / temperature, dim=-1)
        return probs

    def sample_start_frame(self, clip_ids=None):
        """Sample start frames weighted by difficulty."""
        probs = self.get_probs(clip_ids)
        segments = torch.multinomial(probs, 1, True).squeeze(-1)

        seg_sizes = self.segment_sizes[clip_ids]
        time = segments * seg_sizes

        # Add random noise within segment
        noise = torch.rand(clip_ids.shape, device=clip_ids.device) * seg_sizes
        time = time + noise

        # Quantize to dt
        time = (time // self.dt) * self.dt

        # Clamp to minimum start time to avoid negative disc-history lookback
        time = torch.clamp(time, min=self.min_start_time)
        return time
