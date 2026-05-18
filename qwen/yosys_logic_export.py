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


TOKEN_RE = re.compile(r"\s*(->|AND|OR|XOR|XNOR|NOT|\(|\)|0|1|[a-z]\d+)\s*", re.IGNORECASE)
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
            tok = match.group(1)
            tokens.append(tok if re.fullmatch(r"[a-z]\d+", tok) else tok.upper())
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
        if token is not None and re.fullmatch(r"[a-z]\d+", token):
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


def load_distill_circuits(json_path: Path) -> dict[str, Any]:
    """Load circuits from a gemma_sae_distill_results.json file.

    Returns a dict with keys:
      toxicity   — (expression, inputs) for the SAE-level toxicity circuit
      features   — list of (expression, inputs, name, meaning) per feature circuit
      full       — (expression_str, inputs) for the fully inlined circuit (neurons → toxic)
    """
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    hc = payload["hard_circuits"]

    # ── toxicity circuit (i1..i8 = SAE feature bits) ──────────────────────
    tc = hc["toxicity_circuit"]
    tox_inputs = tc["hard_circuit"]["inputs"]
    meanings = payload.get("feature_meanings", {})
    # annotate inputs with SAE meanings for readability
    tox_inputs_ann = [
        {**inp, "meaning": meanings.get(inp["name"], "?")}
        for inp in tox_inputs
    ]
    toxicity = (tc["expression"], tox_inputs_ann)

    # ── per-feature circuits (neuron bits → SAE feature bit) ───────────────
    features = []
    for fc in hc["feature_circuits"]:
        expr  = fc["hard_circuit"]["binary_output"]["expression"]
        inps  = fc["hard_circuit"]["inputs"]
        # rename inputs to n{neuron_id} for unambiguous Verilog port names
        renamed = [
            {**inp, "verilog_name": f"n{inp['feature_id']}"}
            for inp in inps
        ]
        features.append({
            "name":    fc["input_name"],
            "meaning": fc["meaning"],
            "expr":    expr,
            "inputs":  renamed,
        })

    # ── full circuit: inline all feature circuits into the toxicity circuit ─
    # Replace each i{k} in the toxicity expression with (feature_k_expression),
    # after substituting neuron names inside each feature expression.
    def neuron_expr(fc_entry: dict) -> str:
        """Feature circuit expression with neuron port names substituted."""
        expr = fc_entry["expr"]
        for inp in fc_entry["inputs"]:
            # inp["name"] is e.g. "i42" (position in 256 selected neurons)
            # inp["verilog_name"] is e.g. "n1645" (actual hidden-state neuron index)
            pos_num = inp["name"][1:]   # strip leading 'i'
            expr = re.sub(rf"\bi{pos_num}\b", inp["verilog_name"], expr)
        return f"({expr})"

    full_expr = tc["expression"]
    for k, fc_entry in enumerate(features, start=1):
        full_expr = re.sub(
            rf"\bi{k}\b",
            neuron_expr(fc_entry),
            full_expr,
        )

    # collect all neuron inputs (union across all feature circuits)
    neuron_inputs: dict[str, dict] = {}
    for fc_entry in features:
        for inp in fc_entry["inputs"]:
            neuron_inputs[inp["verilog_name"]] = {
                "name":       inp["verilog_name"],
                "feature_id": inp["feature_id"],
                "description": inp.get("description", ""),
            }
    full_inputs = sorted(neuron_inputs.values(), key=lambda x: x["feature_id"])
    full = (full_expr, full_inputs)

    return {"toxicity": toxicity, "features": features, "full": full}


def write_verilog(
    path: Path,
    expression: str,
    inputs: list[dict[str, Any]],
    module_name: str = "toxic_circuit",
    name_key: str = "name",
    comment_fn: Any = None,
) -> None:
    input_names = [str(item[name_key]) for item in inputs]
    verilog_expr = expr_to_verilog(ExpressionParser(expression).parse())
    if comment_fn is None:
        comments = "\n".join(
            f"// {item[name_key]} = SAE feature {item.get('feature_id', 'unknown')}"
            for item in inputs
        )
    else:
        comments = "\n".join(comment_fn(item) for item in inputs)
    input_decls = ", ".join(input_names)
    path.write_text(
        f"""// Generated from difflogic hard binary toxic circuit.
// Original expression: {expression[:200]}{'...' if len(expression) > 200 else ''}
{comments}

module {module_name}(
    input wire {input_decls},
    output wire toxic
);
    assign toxic = {verilog_expr};
endmodule
""",
        encoding="utf-8",
    )


def run_yosys(
    verilog_path: Path,
    opt_verilog_path: Path,
    png_prefix: Path,
    top_module: str = "toxic_circuit",
) -> None:
    yosys = shutil.which("yosys")
    if yosys is None:
        raise RuntimeError("yosys was not found on PATH. Install/load Yosys, then rerun this script.")
    script = "\n".join(
        [
            f"read_verilog {verilog_path}",
            f"hierarchy -top {top_module}",
            "proc",
            "opt",
            "techmap",
            "opt",
            f"write_verilog -noattr {opt_verilog_path}",
            f"show -format png -prefix {png_prefix} {top_module}",
        ]
    )
    subprocess.run([yosys, "-q", "-p", script], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-yosys", action="store_true",
                        help="Only write the unoptimized Verilog.")
    parser.add_argument("--distill", action="store_true",
                        help="Load from gemma_sae_distill_results.json format.")
    parser.add_argument("--circuit", default="toxicity",
                        choices=("toxicity", "full", "all"),
                        help="Which circuit(s) to export (distill mode only). "
                             "toxicity=SAE-level (i1..i8 → toxic); "
                             "full=end-to-end (neurons → toxic); "
                             "all=toxicity + all 8 feature circuits + full.")
    args = parser.parse_args()

    json_path = args.json_path.resolve()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = json_path.parent / "yosys" / json_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.distill:
        circuits = load_distill_circuits(json_path)

        to_export = []
        if args.circuit in ("toxicity", "all"):
            expr, inps = circuits["toxicity"]
            to_export.append(("toxic_circuit", expr, inps, "name",
                lambda item: f"// {item['name']} = SAE feat meaning: {item.get('meaning', '?')}"))
        if args.circuit in ("full", "all"):
            expr, inps = circuits["full"]
            to_export.append(("toxic_circuit_full", expr, inps, "name",
                lambda item: f"// {item['name']} = hidden neuron {item['feature_id']}"))
        if args.circuit == "all":
            for fc in circuits["features"]:
                safe = fc["name"].replace("/", "_")
                # substitute position names (i42) → neuron port names (n1645) in expression
                ne = fc["expr"]
                for inp in fc["inputs"]:
                    pos_num = inp["name"][1:]  # "i42" -> "42"
                    ne = re.sub(rf"\bi{pos_num}\b", inp["verilog_name"], ne)
                to_export.append((f"feature_{safe}", ne, fc["inputs"], "verilog_name",
                    lambda item: f"// {item['verilog_name']} = neuron {item['feature_id']}"))

        for mod_name, expr, inps, name_key, cfn in to_export:
            v_path    = out_dir / f"{mod_name}.v"
            opt_path  = out_dir / f"{mod_name}_opt.v"
            png_pfx   = out_dir / f"{mod_name}_opt"
            write_verilog(v_path, expr, inps, module_name=mod_name,
                          name_key=name_key, comment_fn=cfn)
            print(f"wrote Verilog: {v_path}")
            if not args.skip_yosys:
                run_yosys(v_path, opt_path, png_pfx, top_module=mod_name)
                print(f"wrote optimized Verilog: {opt_path}")
                print(f"wrote PNG: {png_pfx}.png")
        return

    # ── original SAE classifier JSON mode ─────────────────────────────────
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
