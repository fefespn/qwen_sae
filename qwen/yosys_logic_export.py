"""Export a learned difflogic toxic circuit JSON to Verilog and Yosys artifacts.

Example:
    python qwen/yosys_logic_export.py qwen/out/qwen_scope_layer6_en_top32_difflogic.json

Outputs by default:
    qwen/out/yosys/<json-stem>/toxic_circuit.v
    qwen/out/yosys/<json-stem>/toxic_circuit_opt.v
    qwen/out/yosys/<json-stem>/toxic_circuit_opt.png
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"\s*(->|AND|OR|XOR|XNOR|NOT|\(|\)|0|1|i\d+)\s*", re.IGNORECASE)
BINARY_PRECEDENCE = {
    "->": 1,
    "OR": 2,
    "XOR": 3,
    "XNOR": 3,
    "AND": 4,
}
VERILOG_BINARY_OP = {
    "AND": "&",
    "OR": "|",
    "XOR": "^",
}


@dataclass(frozen=True)
class Expr:
    kind: str
    value: str | None = None
    left: "Expr | None" = None
    right: "Expr | None" = None


class ExpressionParser:
    def __init__(self, expression: str) -> None:
        self.tokens = self._tokenize(expression)
        self.pos = 0

    @staticmethod
    def _tokenize(expression: str) -> list[str]:
        tokens = []
        pos = 0
        while pos < len(expression):
            match = TOKEN_RE.match(expression, pos)
            if match is None:
                raise ValueError(f"cannot parse expression near: {expression[pos:pos + 30]!r}")
            tokens.append(match.group(1).upper() if not match.group(1).startswith("i") else match.group(1))
            pos = match.end()
        return tokens

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None:
            raise ValueError("unexpected end of expression")
        if expected is not None and token != expected:
            raise ValueError(f"expected {expected!r}, got {token!r}")
        self.pos += 1
        return token

    def parse(self) -> Expr:
        expr = self.parse_binary(min_precedence=1)
        if self.peek() is not None:
            raise ValueError(f"unexpected trailing token {self.peek()!r}")
        return expr

    def parse_binary(self, min_precedence: int) -> Expr:
        left = self.parse_unary()
        while True:
            op = self.peek()
            if op not in BINARY_PRECEDENCE or BINARY_PRECEDENCE[op] < min_precedence:
                break
            precedence = BINARY_PRECEDENCE[op]
            self.consume()
            right = self.parse_binary(precedence + 1)
            left = Expr("binary", op, left, right)
        return left

    def parse_unary(self) -> Expr:
        token = self.peek()
        if token == "NOT":
            self.consume()
            return Expr("not", left=self.parse_unary())
        if token == "(":
            self.consume("(")
            expr = self.parse_binary(min_precedence=1)
            self.consume(")")
            return expr
        if token in {"0", "1"}:
            self.consume()
            return Expr("const", token)
        if token is not None and re.fullmatch(r"i\d+", token):
            self.consume()
            return Expr("input", token)
        raise ValueError(f"unexpected token {token!r}")


def expr_to_verilog(expr: Expr) -> str:
    if expr.kind == "const":
        return f"1'b{expr.value}"
    if expr.kind == "input":
        return str(expr.value)
    if expr.kind == "not":
        return f"(~{expr_to_verilog(required(expr.left))})"
    if expr.kind == "binary":
        left = expr_to_verilog(required(expr.left))
        right = expr_to_verilog(required(expr.right))
        if expr.value in VERILOG_BINARY_OP:
            return f"({left} {VERILOG_BINARY_OP[expr.value]} {right})"
        if expr.value == "XNOR":
            return f"(~({left} ^ {right}))"
        if expr.value == "->":
            return f"((~{left}) | {right})"
    raise ValueError(f"unsupported expression node {expr}")


def required(value: Expr | None) -> Expr:
    if value is None:
        raise ValueError("malformed expression tree")
    return value


def load_binary_circuit(json_path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    circuit = payload.get("logic_classifier", {}).get("hard_circuit", {})
    binary = circuit.get("binary_output", {})
    expression = binary.get("expression")
    inputs = circuit.get("inputs")
    if not expression or not isinstance(expression, str):
        raise ValueError(f"{json_path} does not contain logic_classifier.hard_circuit.binary_output.expression")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError(f"{json_path} does not contain hard_circuit.inputs")
    return expression, inputs


def write_verilog(path: Path, expression: str, inputs: list[dict[str, Any]]) -> None:
    input_names = [str(item["name"]) for item in inputs]
    verilog_expr = expr_to_verilog(ExpressionParser(expression).parse())
    comments = "\n".join(
        f"// {item['name']} = SAE feature {item.get('feature_id', 'unknown')}"
        for item in inputs
    )
    input_decls = ", ".join(input_names)
    path.write_text(
        f"""// Generated from difflogic hard binary toxic circuit.
// Original expression: {expression}
{comments}

module toxic_circuit(
    input wire {input_decls},
    output wire toxic
);
    assign toxic = {verilog_expr};
endmodule
""",
        encoding="utf-8",
    )


def run_yosys(verilog_path: Path, opt_verilog_path: Path, png_prefix: Path) -> None:
    yosys = shutil.which("yosys")
    if yosys is None:
        raise RuntimeError("yosys was not found on PATH. Install/load Yosys, then rerun this script.")
    script = "\n".join(
        [
            f"read_verilog {verilog_path}",
            "hierarchy -top toxic_circuit",
            "proc",
            "opt",
            "techmap",
            "opt",
            f"write_verilog -noattr {opt_verilog_path}",
            f"show -format png -prefix {png_prefix} toxic_circuit",
        ]
    )
    subprocess.run([yosys, "-q", "-p", script], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-yosys", action="store_true", help="Only write the unoptimized Verilog.")
    args = parser.parse_args()

    json_path = args.json_path.resolve()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = json_path.parent / "yosys" / json_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    expression, inputs = load_binary_circuit(json_path)
    verilog_path = out_dir / "toxic_circuit.v"
    opt_verilog_path = out_dir / "toxic_circuit_opt.v"
    png_prefix = out_dir / "toxic_circuit_opt"

    write_verilog(verilog_path, expression, inputs)
    print(f"wrote Verilog: {verilog_path}")

    if args.skip_yosys:
        return

    run_yosys(verilog_path, opt_verilog_path, png_prefix)
    print(f"wrote optimized Verilog: {opt_verilog_path}")
    print(f"wrote PNG: {png_prefix}.png")


if __name__ == "__main__":
    main()
