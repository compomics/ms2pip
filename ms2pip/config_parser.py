import os

import tomlkit


class ConfigParser:
    """
    MS2PIP Configuration parser
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.config = dict()

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
            params["frag_error"] == float(params["frag_error"])
        
        self.config['ms2pip'] = params

    def _load_toml(self):
        with open(self.filepath, 'rt') as f_in:
            self.config = tomlkit.parse(f_in)

    def _write_toml(self, filepath=None):
        if not filepath:
            filepath = self.filepath
        filepath = os.path.splitext(filepath)[0] + '.toml'
        with open(filepath, 'wt+') as f_out:
            f_out.write(tomlkit.dumps(self.config))
            
    def load(self, format=None):
        """
        Load configuration file.

        Parameters
        ----------
        format: str
            Config
        """
        pass
 

def test():
    config_txt_file = 'config.txt'

    config_parser = ConfigParser(config_txt_file)
    config_parser._load_ms2pip_txt()

    print(config_parser.config)

    config_parser._write_toml()


if __name__ == "__main__":
    test()
