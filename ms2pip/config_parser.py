import os

import tomlkit


class UnsupportedConfigFormatError(ValueError):
    pass


class NoFilepathError(Exception):
    pass


class ConfigParser:
    """
    MS2PIP Configuration parser

    Parameters
    ----------
    filepath: str
        Path to config file to load from or write to (optional)
    """

    def __init__(self, filepath=None):
        self.filepath = filepath
        self.config = dict()

        if self.filepath:
            self.load()

    def _set_filepath(self, filepath):
        """
        Set config filepath

        Parameters
        ----------
        filepath: str
            Path to config file to load

        Raises
        ------
        NoFilepathError
            If both filepath and self.filepath are None
        """
        if not filepath:
            if not self.filepath:
                raise NoFilepathError()
        else:
            self.filepath = filepath

    def _load_ms2pip_txt(self):
        params = {}
        params["ptm"] = []
        params["sptm"] = []
        params["gptm"] = []

        with open(self.filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line[0] == "#":
                    continue
                (par, val) = line.split("=")
                if par == "ptm":
                    params["ptm"].append(val)
                elif par == "sptm":
                    params["sptm"].append(val)
                elif par == "gptm":
                    params["gptm"].append(val)
                else:
                    params[par] = val

        if "frag_error" in params:
            params["frag_error"] = float(params["frag_error"])

        self.config["ms2pip"] = params

    def _load_toml(self):
        toml_file = ""
        with open(self.filepath, "rt") as f_in:
            for line in f_in:
                toml_file += line
        self.config = tomlkit.loads(toml_file)

    def _write_toml(self):
        self.filepath = os.path.splitext(self.filepath)[0] + ".toml"
        with open(self.filepath, "wt+") as f_out:
            f_out.write(tomlkit.dumps(self.config))

    def load(self, filepath=None, config_format=None):
        """
        Load configuration file.

        Parameters
        ----------
        filepath: str
            Path to config file to load (optional)
        config_format: str
            Config file format, either `txt` or `toml`. If None, the format will be
            inferred from the filename extension. (optional)
        """
        self._set_filepath(filepath)

        if not config_format:
            config_format = os.path.splitext(self.filepath)[1].lower().lstrip(".")

        if config_format == "toml":
            self._load_toml()
        elif config_format in ("txt", "config", "ms2pip"):
            self._load_ms2pip_txt()
        else:
            raise UnsupportedConfigFormatError(
                "Configuration file should have extension `txt`, `config`, or "
                "`ms2pip` (text-based format) or `toml` (TOML-based format), not "
                f"`{config_format}`",
            )

    def write(self, filepath=None, config_format="toml"):
        """
        Write configuration to file.

        Parameters
        ----------
        filepath: str
            Path where config file will be written (optional)
        config_format: str
            Config file format to write, default: `toml` (optional)
        """
        self._set_filepath(filepath)

        if config_format.lower() == "toml":
            self._write_toml()
        else:
            raise UnsupportedConfigFormatError(config_format)
