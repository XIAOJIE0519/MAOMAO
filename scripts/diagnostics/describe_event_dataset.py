#!/usr/bin/env python3
"""Write a human-readable event-vocabulary and provenance report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path,
                        default=Path("data/perioperative_event_sequences_v5_full"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/MAOMAO_V5_EVENT_VOCABULARY.md"))
    args = parser.parse_args()
    meta = json.loads((args.data_dir / "event_sequence_meta.json").read_text())
    selection = meta.get("outcome_selection", {})
    families = meta.get("outcome_family_vocabulary", [])
    family_ids = meta.get("outcome_to_family", [])
    counts = meta.get("audit_counts", {})
    patient_counts = selection.get("patient_counts", {})
    rows = []
    for index, name in enumerate(meta["outcome_vocabulary"], 1):
        family = families[family_ids[index - 1]] if family_ids else "unknown"
        definition = meta.get("outcome_definitions", {}).get(name, "")
        rows.append(f"| {index} | `{name}` | `{family}` | {counts.get(name, 0)} | "
                    f"{patient_counts.get(name, 'n/a')} | {definition} |")
    text = f"""# MAOMAO v5 事件词表

数据目录：`{args.data_dir}`  
版本：{meta.get('version')}；训练来源：`{meta.get('source_root')}`；MIMIC 使用：`{meta.get('mimic_used')}`

## 规模

- 手术 episode：{meta.get('num_operation_episodes', meta.get('num_admissions')):,}
- token：{meta.get('num_tokens', 0):,}
- 候选事件：{selection.get('candidate_count', 'n/a')}
- 观察到候选事件：{selection.get('observed_candidate_count', 'n/a')}
- 合格输出事件：{len(meta.get('outcome_vocabulary', []))}
- 语义族：{len(families)}
- 选择阈值：至少 {selection.get('minimum_occurrences', 'n/a')} 次，至少 {selection.get('minimum_independent_patients', 'n/a')} 名独立患者；无固定 cap
- 静态变量：{', '.join(item['name'] for item in meta.get('static_features', []))}
- clock：每 {meta.get('clock_token_policy', {}).get('interval_minutes', 'n/a')} 分钟的静息时间摘要，最多 {meta.get('clock_token_policy', {}).get('maximum_tokens_per_inter_record_gap', 'n/a')} 个/间隔

## 输出事件

| # | event token | family | occurrences | independent patients | definition |
|---:|---|---|---:|---:|---|
{chr(10).join(rows)}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
