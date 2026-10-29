# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Selected exact excerpts from python/ttl/ttl_api.py, not executable as a standalone file.

def _write_kernel_to_tmp(name: str, source: str) -> str:
    """Write generated kernel source and return the path used by TT-Metal JIT."""
    import hashlib
    import os
    import tempfile

    content_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    user = os.environ.get("USER", "default")
    output_dir = Path("/tmp") / user
    worker_name = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_name:
        output_dir = output_dir / worker_name
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"ttlang_kernel_{name}_{content_hash}.cpp"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=output_dir,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(source)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
    if os.environ.get("TTLANG_VERBOSE_KERNELS", "1") != "0":
        print(f"=== {name} kernel written to {path} ===")
        print(source)
        print("=" * 60)
    return str(path)


def _lookup_kernel_func_op(module, kernel_name: str, *, symbol_table=None):

        initial_mlir_path = os.environ.get("TTLANG_INITIAL_MLIR")
        if initial_mlir_path:
            with open(initial_mlir_path, "w") as fd:
                module.operation.print(
                    file=fd,
                    enable_debug_info=print_debug_locations,
                    print_generic_op_form=False,
                )
            print(f"SAVED INITIAL TO {initial_mlir_path}")


        final_mlir_path = os.environ.get("TTLANG_FINAL_MLIR")
        if final_mlir_path:
            with open(final_mlir_path, "w") as fd:
                module.operation.print(
                    file=fd,
                    enable_debug_info=print_debug_locations,
                    print_generic_op_form=False,
                )
            print(f"SAVED FINAL TO {final_mlir_path}")

        # Extract source lines for auto-profiling (use first thread's source)
