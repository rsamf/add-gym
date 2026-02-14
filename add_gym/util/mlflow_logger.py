import os

import mlflow

import add_gym.util.logger as logger


class MLflowLogger(logger.Logger):
    MISC_TAG = "Misc"

    def __init__(self):
        super().__init__()

        self._step_var_key = None
        self._collections = dict()
        self._run = None
        self._output_dir = None

    def reset(self):
        super().reset()

    def configure_output_file(self, filename=None):
        super().configure_output_file(filename)

        if logger.Logger.is_root():
            self._output_dir = os.path.dirname(filename)
            mlflow.set_tracking_uri(self._output_dir)
            self._run = mlflow.start_run()

    def set_step_key(self, var_key):
        self._step_key = var_key

    def log(self, key, val, collection=None, quiet=False):
        super().log(key, val, quiet)

        if collection is not None:
            self._add_collection(collection, key)

    def write_log(self):
        row_count = self._row_count

        super().write_log()

        if logger.Logger.is_root() and (self._run is not None):
            if row_count == 0:
                self._key_tags = self._build_key_tags()

            step_val = row_count
            if self._step_key is not None:
                step_val = self.log_current_row.get(self._step_key, "").val
            step_val = int(step_val)

            for i, key in enumerate(self.log_headers):
                if key != self._step_key:
                    entry = self.log_current_row.get(key, "")
                    val = entry.val
                    tag = self._key_tags[i]
                    mlflow.log_metric(tag, val, step=step_val)

    def log_image(self, tag, fig, step):
        """Log a matplotlib figure as an image artifact."""
        if logger.Logger.is_root() and (self._run is not None):
            mlflow.log_figure(fig, f"{tag}/step_{step}.png")

    def _add_collection(self, name, key):
        if name not in self._collections:
            self._collections[name] = []
        self._collections[name].append(key)

    def _build_key_tags(self):
        tags = []
        for key in self.log_headers:
            curr_tag = MLflowLogger.MISC_TAG
            for col_tag, col_keys in self._collections.items():
                if key in col_keys:
                    curr_tag = col_tag

            curr_tags = "{:s}/{:s}".format(curr_tag, key)
            tags.append(curr_tags)

        return tags
