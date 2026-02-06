import os
import time
import atexit
import torch
import torch.distributed as dist


class Logger:
    class Entry:
        def __init__(self, val, quiet=False):
            self.val = val
            self.quiet = quiet

    @staticmethod
    def print(str, end=None):
        if Logger.is_root():
            print(str, end=end)

    @staticmethod
    def is_root():
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def __init__(self):
        self.output_file = None
        self.log_headers = []
        self.log_current_row = {}
        self._dump_str_template = ""
        self._max_key_len = 0
        self._row_count = 0
        self._need_update = True
        self._data_buffer = None

    def reset(self):
        self._row_count = 0
        self.log_headers = []
        self.log_current_row = {}
        self._need_update = True
        self._data_buffer = None

        if self.output_file is not None:
            self.output_file.truncate(0)

    def configure_output_file(self, filename=None):
        """
        Set output directory to d, or to /tmp/somerandomnumber if d is None
        """
        self._row_count = 0
        self.log_headers = []
        self.log_current_row = {}

        output_path = filename or "output/log_%i.txt" % int(time.time())

        out_dir = os.path.dirname(output_path)

        if Logger.is_root():
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            self.output_file = open(output_path, "w")
            assert os.path.exists(output_path)
            atexit.register(self.output_file.close)

            Logger.print("Logging data to " + self.output_file.name)

    def log(self, key, val, quiet=False, **kwargs):
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        """
        if (self._row_count == 0) and key not in self.log_headers:
            self.log_headers.append(key)
            self._max_key_len = max(self._max_key_len, len(key))
        else:
            assert key in self.log_headers, (
                "Trying to introduce a new key %s that you didn't include in the first iteration"
                % key
            )
        self.log_current_row[key] = Logger.Entry(val, quiet)
        self._need_update = True

    def get_num_keys(self):
        return len(self.log_headers)

    def print_log(self):
        """
        Print all of the diagnostics from the current iteration
        """

        if dist.is_initialized() and self._need_update:
            self._mp_aggregate()

        key_spacing = self._max_key_len
        format_str = "| %" + str(key_spacing) + "s | %15s |"

        if Logger.is_root():
            vals = []
            Logger.print("-" * (22 + key_spacing))
            for key in self.log_headers:
                entry = self.log_current_row.get(key, "")
                if not (entry.quiet):
                    val = entry.val

                    if isinstance(val, float):
                        valstr = "%8.3g" % val
                    elif isinstance(val, int):
                        valstr = str(val)
                    else:
                        valstr = val

                    Logger.print(format_str % (key, valstr))
                    vals.append(val)
            Logger.print("-" * (22 + key_spacing))

    def write_log(self):
        """
        Write all of the diagnostics from the current iteration
        """

        if dist.is_initialized() and self._need_update:
            self._mp_aggregate()

        if Logger.is_root():
            if self._row_count == 0:
                self._dump_str_template = self._build_str_template()

            vals = []
            for key in self.log_headers:
                entry = self.log_current_row.get(key, "")
                val = entry.val
                vals.append(val)

            if self.output_file is not None:
                if self._row_count == 0:
                    header_str = self._dump_str_template.format(*self.log_headers)
                    self.output_file.write(header_str + "\r")

                val_str = self._dump_str_template.format(*map(str, vals))
                self.output_file.write(val_str + "\r")
                self.output_file.flush()

        self._row_count += 1

    def has_key(self, key):
        return key in self.log_headers

    def get_current_val(self, key):
        val = None
        if key in self.log_current_row.keys():
            entry = self.log_current_row[key]
            val = entry.val
        return val

    def _build_str_template(self):
        num_keys = self.get_num_keys()
        template = "{:<25}" * num_keys
        return template

    def _mp_aggregate(self):
        if self._data_buffer is None:
            n = len(self.log_headers)
            device = (
                torch.device("cuda")
                if torch.cuda.is_available() and dist.get_backend() == "nccl"
                else torch.device("cpu")
            )
            self._data_buffer = torch.zeros(n, dtype=torch.float64, device=device)

        for i, key in enumerate(self.log_headers):
            entry = self.log_current_row[key]
            val = entry.val
            self._data_buffer[i] = val

        dist.all_reduce(self._data_buffer, op=dist.ReduceOp.SUM)
        self._data_buffer /= dist.get_world_size()

        for i, key in enumerate(self.log_headers):
            entry = self.log_current_row[key]
            val = self._data_buffer[i].item()
            if isinstance(entry.val, int):
                val = int(val)
            entry.val = val
        self._need_update = False
