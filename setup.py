"""Build Napa's native C core; all other metadata lives in pyproject.toml."""

from setuptools import Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext

NATIVE = "src/napa_vision/_native"


class BuildNative(build_ext):
    """Compile the dependency-free C core as a plain shared library loaded with ctypes."""

    def build_extensions(self) -> None:
        for extension in self.extensions:
            if self.compiler.compiler_type == "msvc":
                extension.extra_compile_args = ["/O2"]
            else:
                extension.extra_compile_args = ["-O3", "-std=c99", "-fvisibility=hidden"]
                extension.libraries = ["m"]
        super().build_extensions()


class PlatformWheel(bdist_wheel):
    """Tag wheels py3-none-<platform>: ctypes loads the library, so one wheel serves every Python 3."""

    def get_tag(self) -> tuple[str, str, str]:
        return "py3", "none", super().get_tag()[2]


setup(
    ext_modules=[
        Extension(
            "napa_vision._native._napa_core",
            sources=[f"{NATIVE}/napa_core.c"],
            depends=[f"{NATIVE}/napa_core.h", f"{NATIVE}/tag36h11_codes.h"],
            include_dirs=[NATIVE],
            # Napa is the C core: installing without a C compiler fails rather than shipping no detector.
            optional=False,
        )
    ],
    cmdclass={"build_ext": BuildNative, "bdist_wheel": PlatformWheel},
)
