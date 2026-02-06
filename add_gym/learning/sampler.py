import torch
import torch.nn.functional as F


class AdaptiveSegmentSampler:
    def __init__(self, clip_lengths, dt, num_segments=20, temperature=1.0):
        self.num_segments = num_segments
        self.dt = dt
        self.temperature = temperature

        # Calculate segment size for each clip (treating clip_lengths as seconds)
        self.segment_sizes = torch.tensor(
            [clip_length / num_segments for clip_length in clip_lengths]
        )  # (num_clips,)

        num_clips = len(clip_lengths)
        self.errors = torch.ones((num_clips, num_segments))

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

        # Update errors using approximate scatter/indexing (last write wins for duplicates in batch)
        current_vals = self.errors[clip_ids, seg_idxs]
        new_vals = 0.9 * current_vals + 0.1 * tracking_errors
        self.errors.index_put_((clip_ids, seg_idxs), new_vals, accumulate=False)
        
    def get_probs(self, clip_ids=None):
        """Get sampling probabilities for segments of given clips."""
        if clip_ids is None:
            clip_ids = torch.arange(self.errors.shape[0], device=self.errors.device)

        if self.errors.device != clip_ids.device:
            self.errors = self.errors.to(clip_ids.device)
            self.segment_sizes = self.segment_sizes.to(clip_ids.device)

        clip_errors = self.errors[clip_ids]

        probs = F.softmax(clip_errors / self.temperature, dim=-1)
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
        return time
