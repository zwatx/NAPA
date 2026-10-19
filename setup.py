"""Build the optional native Napa core; all other metadata lives in pyproject.toml."""

from setuptools import Extension, setup
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


setup(
    ext_modules=[
        Extension(
            "napa_vision._native._napa_core",
            sources=[f"{NATIVE}/napa_core.c"],
            depends=[f"{NATIVE}/napa_core.h", f"{NATIVE}/tag36h11_codes.h"],
            include_dirs=[NATIVE],
            # Without a compiler the install still succeeds and Napa falls back to OpenCV detection.
            optional=True,
        )
    ],
    cmdclass={"build_ext": BuildNative},
)
