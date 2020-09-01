from setuptools.command.develop import develop as st_develop


class develop(st_develop):
    __doc__ = st_develop.__doc__

    def install_for_development(self):
        self.run_command('build_clib')
        st_develop.install_for_development(self)
