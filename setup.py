from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class universal_bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = super().get_tag()
        return ("py3", "none", plat)


setup(
    packages=find_packages(),
    package_data={"tech_rkj": ["bin/*.so"]},
    include_package_data=True,
    cmdclass={"bdist_wheel": universal_bdist_wheel},
    has_ext_modules=lambda: True,
)
