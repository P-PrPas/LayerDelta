"""The isolated worker of the `onnx` provider (v0.3-G7): parses an untrusted .onnx file (protobuf and C++ shape inference) away from
the server. Same protocol as export_worker: `--request request.json --output DIR`, result `DIR/ir.json` or `DIR/error.json`."""

from . import export_worker, onnx_provider


def build(request: dict) -> dict:
    return onnx_provider.build_ir(request["spec"], request["tensors"], request["table_signature"])


if __name__ == "__main__":
    export_worker.main(build)
