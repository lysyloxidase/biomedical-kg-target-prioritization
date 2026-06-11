"""Assemble markdown and LaTeX ablation tables."""

from __future__ import annotations

from pathlib import Path

from kgtp.ablation.ablation1_nokg_vs_kg import KG_ARM, NO_KG_ARM
from kgtp.ablation.ablation2_kg_vs_kgtext import KG_TEXT_ARM
from kgtp.ablation.ablation3_homo_rel_hetero import GRAPH_SAGE_ARM, HGT_ARM, RGCN_ARM
from kgtp.ablation.common import TABLE_METRICS, AblationArm, AblationReport
from kgtp.data.common import PathLike


def ablation12_markdown(ablation1: AblationReport, ablation2: AblationReport) -> str:
    """Rows: no-KG, KG, KG+text; columns: AUROC/AUPRC/Hits@10/MRR."""

    arms = [
        _find_arm(ablation1, NO_KG_ARM),
        _find_arm(ablation1, KG_ARM),
        _find_arm(ablation2, KG_TEXT_ARM),
    ]
    return markdown_table("Ablations 1-2", arms)


def ablation3_markdown(ablation3: AblationReport) -> str:
    """Rows: GraphSAGE, R-GCN, HGT."""

    arms = [
        _find_arm(ablation3, GRAPH_SAGE_ARM),
        _find_arm(ablation3, RGCN_ARM),
        _find_arm(ablation3, HGT_ARM),
    ]
    return markdown_table("Ablation 3", arms)


def ablation4_markdown(ablation4: AblationReport) -> str:
    """Rows: full design-knob grid."""

    return markdown_table("Ablation 4 Design Grid", ablation4.arms)


def markdown_table(title: str, arms: list[AblationArm]) -> str:
    """Build a compact markdown table with mean ± std cells."""

    header = ["row", *TABLE_METRICS]
    lines = [
        f"### {title}",
        "",
        "|" + "|".join(header) + "|",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for arm in arms:
        cells = [arm.name, *[_format_cell(arm, metric) for metric in TABLE_METRICS]]
        lines.append("|" + "|".join(cells) + "|")
    return "\n".join(lines) + "\n"


def latex_table(title: str, arms: list[AblationArm]) -> str:
    """Build a LaTeX tabular with the same cells."""

    columns = "l" + "r" * len(TABLE_METRICS)
    lines = [
        "\\begin{table}",
        "\\centering",
        f"\\caption{{{title}}}",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        "Row & " + " & ".join(TABLE_METRICS) + " \\\\",
        "\\midrule",
    ]
    for arm in arms:
        cells = [
            arm.name,
            *[_format_cell(arm, metric, latex=True) for metric in TABLE_METRICS],
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def write_ablation_tables(
    ablation1: AblationReport,
    ablation2: AblationReport,
    ablation3: AblationReport,
    ablation4: AblationReport,
    output_dir: PathLike,
) -> dict[str, Path]:
    """Write markdown + LaTeX tables and an honest narrative file."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    md_path = output / "ablation_tables.md"
    tex_path = output / "ablation_tables.tex"
    narrative_path = output / "honest_narrative.md"

    md = "\n".join(
        [
            ablation12_markdown(ablation1, ablation2),
            ablation3_markdown(ablation3),
            ablation4_markdown(ablation4),
        ]
    )
    tex = "\n".join(
        [
            latex_table(
                "Ablations 1-2",
                [
                    _find_arm(ablation1, NO_KG_ARM),
                    _find_arm(ablation1, KG_ARM),
                    _find_arm(ablation2, KG_TEXT_ARM),
                ],
            ),
            latex_table(
                "Ablation 3",
                [
                    _find_arm(ablation3, GRAPH_SAGE_ARM),
                    _find_arm(ablation3, RGCN_ARM),
                    _find_arm(ablation3, HGT_ARM),
                ],
            ),
            latex_table("Ablation 4 Design Grid", ablation4.arms),
        ]
    )
    narrative = "\n\n".join(
        [
            "# Honest Ablation Narrative",
            f"## Ablation 1\n{ablation1.narrative}",
            f"## Ablation 2\n{ablation2.narrative}",
            f"## Ablation 3\n{ablation3.narrative}",
            f"## Ablation 4\n{ablation4.narrative}",
        ]
    )
    md_path.write_text(md, encoding="utf-8")
    tex_path.write_text(tex, encoding="utf-8")
    narrative_path.write_text(narrative, encoding="utf-8")
    return {"markdown": md_path, "latex": tex_path, "narrative": narrative_path}


def _format_cell(arm: AblationArm, metric: str, *, latex: bool = False) -> str:
    summary = arm.summary[metric]
    plus_minus = "$\\pm$" if latex else "±"
    marker = _significance_marker(arm)
    return f"{summary['mean']:.3f} {plus_minus} {summary['std']:.3f}{marker}"


def _significance_marker(arm: AblationArm) -> str:
    sig = arm.significance_vs_hgt or {}
    p_value = sig.get("p_value")
    if isinstance(p_value, float) and p_value < 0.05:
        return "*"
    return ""


def _find_arm(report: AblationReport, name: str) -> AblationArm:
    for arm in report.arms:
        if arm.name == name:
            return arm
    msg = f"Missing arm {name} in {report.name}"
    raise KeyError(msg)
