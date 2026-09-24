"""项目权重临界点分析（只读）。

以当前校准范围与逐项裁决形成的**项目最终均分**（``eff_avg``）为基础：选定一个
项目与倍率上限 ``upper``（1～5），令该项目的倍率 ``w`` 从 0 连续变化到
``upper``，其余项目倍率恒为 1。选手 ``c`` 的总分为关于 ``w`` 的一次函数::

    T_c(w) = base_c + w * value_c

其中 ``value_c`` 是该选手在选定项目上的最终均分（在该项目无评分时按 0 贡献，
``value_c = 0``），``base_c`` 是其余项目最终均分之和。

任意两名选手的总分差也是一次函数，在闭区间 ``[0, upper]`` 内至多有一个交点：

- 斜率与截距都相同（``Δvalue = Δbase = 0``）的两名选手**始终并列**，不产生交点；
- 否则求解 ``T_a(w) = T_b(w)``，交点落在 ``[0, upper]`` 内才记为临界点，
  同一位置的交点（浮点容差内）合并，交点处并列的选手一并标出。

端点 0 与 upper 本身也作为临界点展示（区间为闭区间）；相邻临界点之间取开区间
中点，给出该区间内的竞赛名次与总分。全部计算为纯函数，不读写任何存储；
评分、裁决或校准范围变化后由调用方重新请求即可。
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

# 与 scoring.TIE_EPS 保持同一量级；同时允许随数据尺度轻微放大，避免在大数值
# 总分上把浮点误差误判为交点，或把同一位置的交点误拆成两个。
TIE_EPS = 1e-9
MAX_UPPER = 5.0
MIN_UPPER = 1.0


class WeightAnalysisError(ValueError):
    """项目权重分析请求参数不合法（API 层转为 400）。"""


# ---------- 并查集：合并「在同一倍率处总分相等」的选手 ----------

class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # 固定合并到编号较小的根，保证输出稳定。
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _merge_tol(scale: float) -> float:
    """交点合并 / 同分判定容差：随数据尺度放大，但不小于 1e-9。"""
    return TIE_EPS * max(1.0, scale)


def weight_analysis(analysis: Mapping[str, Any], item: str, upper: float) -> dict[str, Any]:
    """计算选定项目的权重临界点分析。

    ``analysis`` 必须是 ``scoring.analyze`` 的完整输出（使用其中的最终项目均分
    ``eff_avg``、选手列表与项目列表）。``item`` 必须存在于 ``analysis["items"]``。
    """
    if not isinstance(upper, (int, float)) or isinstance(upper, bool) or not math.isfinite(upper):
        raise WeightAnalysisError("倍率上限必须是数字。")
    upper = float(upper)
    if not (MIN_UPPER <= upper <= MAX_UPPER):
        raise WeightAnalysisError(f"倍率上限必须在 {MIN_UPPER:g}～{MAX_UPPER:g} 之间。")
    items = list(analysis["items"])
    if item not in items:
        raise WeightAnalysisError(
            f"项目「{item}」不存在，可选：{', '.join(items) or '(无)'}。"
        )

    names = [b["contestant"] for b in analysis["contestants"]]
    if not names:
        raise WeightAnalysisError("当前会话没有评分数据，无法进行权重临界点分析。")
    index = {name: i for i, name in enumerate(names)}

    # 从分析结果提取每个选手的最终项目均分；缺失项目不在 items 块中，按 0 贡献。
    values: dict[str, float] = {name: 0.0 for name in names}
    base: dict[str, float] = {name: 0.0 for name in names}
    has_item: dict[str, bool] = {name: False for name in names}
    for block in analysis["contestants"]:
        name = block["contestant"]
        for ib in block["items"]:
            if ib["item"] == item:
                values[name] = float(ib["eff_avg"])
                has_item[name] = True
            else:
                base[name] += float(ib["eff_avg"])

    n = len(names)
    scale = max(
        [1.0]
        + [abs(base[name]) for name in names]
        + [abs(values[name]) for name in names]
        + [abs(base[name] + upper * values[name]) for name in names]
    )
    tol = _merge_tol(scale)

    # 1) 始终并列：base 与 value 都相同的选手对在任意倍率下总分恒等，不产生交点。
    always_dsu = _DSU(n)
    for i in range(n):
        ni = names[i]
        for j in range(i + 1, n):
            nj = names[j]
            if (
                abs(values[ni] - values[nj]) <= tol
                and abs(base[ni] - base[nj]) <= tol
            ):
                always_dsu.union(i, j)

    def group_members(dsu: _DSU) -> list[list[int]]:
        buckets: dict[int, list[int]] = {}
        for i in range(n):
            buckets.setdefault(dsu.find(i), []).append(i)
        return [sorted(members) for members in buckets.values()]

    always_groups = [
        {"contestants": [names[i] for i in members], "size": len(members)}
        for members in group_members(always_dsu)
        if len(members) > 1
    ]
    always_groups.sort(key=lambda g: g["contestants"][0])
    always_pairs = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if always_dsu.find(i) == always_dsu.find(j)
    ]

    def new_cluster_dsu() -> _DSU:
        # 交点处的并列必须带上始终并列关系：若 A≡B（始终并列）且 A 与 C 相交，
        # 则该临界点上 B 同样与 C 并列。
        dsu = _DSU(n)
        for i, j in always_pairs:
            dsu.union(i, j)
        return dsu

    # 2) 枚举全部选手对，求闭区间内的交点；始终并列的选手对跳过。
    raw_events: list[tuple[float, int, int]] = []
    for i in range(n):
        ni = names[i]
        for j in range(i + 1, n):
            nj = names[j]
            dv = values[ni] - values[nj]
            db = base[ni] - base[nj]
            if abs(dv) <= tol and abs(db) <= tol:
                continue  # 始终并列，不产生交点
            if abs(dv) <= tol:
                continue  # 总分差恒为非零常数，不存在交点
            w = -db / dv
            # 闭区间 [0, upper]；端点附近的误差吸附到端点。
            if w < -tol or w > upper + tol:
                continue
            if abs(w) <= tol:
                w = 0.0
            elif abs(w - upper) <= tol:
                w = upper
            elif w < 0.0 or w > upper:
                continue
            raw_events.append((w, i, j))

    # 3) 同位置交点合并（容差聚类），用并查集汇总该位置并列的全部选手。
    raw_events.sort(key=lambda e: e[0])
    clusters: list[dict[str, Any]] = []
    for w, i, j in raw_events:
        if clusters and abs(w - clusters[-1]["w"]) <= tol:
            clusters[-1]["dsu"].union(i, j)
        else:
            dsu = new_cluster_dsu()
            dsu.union(i, j)
            clusters.append({"w": w, "dsu": dsu})

    def snapshot(weight: float, dsu: _DSU) -> dict[str, Any]:
        """给定倍率与并列关系，并查集输出竞赛名次与总分（1224 排名）。"""
        totals = [base[names[i]] + weight * values[names[i]] for i in range(n)]
        roots = {i: dsu.find(i) for i in range(n)}
        # 每个并列组取组内最小数值总分作为排序键（组内数值理论相等，取最小兜底）。
        group_key: dict[int, float] = {}
        for i in range(n):
            r = roots[i]
            t = totals[i]
            if r not in group_key or t < group_key[r]:
                group_key[r] = t
        ordered_groups = sorted(
            group_key.keys(), key=lambda r: (-group_key[r], names[min(
                k for k in range(n) if roots[k] == r
            )])
        )
        rank_of_group: dict[int, int] = {}
        last_score: float | None = None
        last_rank = 0
        for position, r in enumerate(ordered_groups, start=1):
            score = group_key[r]
            if last_score is not None and abs(score - last_score) <= tol:
                rank_of_group[r] = last_rank
            else:
                rank_of_group[r] = position
                last_rank = position
            last_score = score
        group_size = {r: sum(1 for k in range(n) if roots[k] == r) for r in set(roots.values())}
        always_root = {i: always_dsu.find(i) for i in range(n)}
        rows = [
            {
                "contestant": names[i],
                "total": totals[i],
                "rank": rank_of_group[roots[i]],
                "tied": group_size[roots[i]] > 1,
                "always_tied": any(
                    always_root[i] == always_root[k] and k != i for k in range(n)
                ),
            }
            for i in range(n)
        ]
        rows.sort(key=lambda row: (row["rank"], row["contestant"]))
        return {"weight": weight, "standings": rows}

    def crossing_snapshot(weight: float, dsu: _DSU) -> dict[str, Any]:
        snap = snapshot(weight, dsu)
        groups = [
            {
                "contestants": [names[i] for i in members],
                "size": len(members),
                "total": base[names[members[0]]] + weight * values[names[members[0]]],
            }
            for members in group_members(dsu)
            if len(members) > 1
        ]
        groups.sort(key=lambda g: (-g["total"], g["contestants"][0]))
        snap["tie_groups"] = groups
        return snap

    # 4) 组装时间线：临界点（含端点）与相邻开区间中点交错按倍率排列。
    grid = [0.0]
    for cl in clusters:
        w = cl["w"]
        if all(abs(w - g) > tol for g in grid):
            grid.append(w)
    if all(abs(upper - g) > tol for g in grid):
        grid.append(upper)
    grid.sort()

    def cluster_at(weight: float) -> dict[str, Any] | None:
        # 交点吸附到端点后，网格上用端点值查找，需按容差匹配回原交点簇。
        for cl in clusters:
            if abs(cl["w"] - weight) <= tol:
                return cl
        return None

    timeline: list[dict[str, Any]] = []
    for k, w in enumerate(grid):
        cl = cluster_at(w)
        if cl is not None:
            snap = crossing_snapshot(w, cl["dsu"])
        else:
            # 端点若没有任何选手对在此相交，仍需按始终并列关系给名次。
            snap = crossing_snapshot(w, always_dsu)
        if k == 0 and abs(w) <= tol:
            kind, label = "endpoint", "起点 w=0"
        elif k == len(grid) - 1 and abs(w - upper) <= tol:
            kind, label = "endpoint", f"上限 w={upper:g}"
        else:
            kind, label = "critical", "临界倍率"
        snap.update(
            {
                "kind": kind,
                "label": label,
                "tie_count": sum(g["size"] for g in snap["tie_groups"]),
            }
        )
        timeline.append(snap)

        if k + 1 < len(grid):
            w_next = grid[k + 1]
            if w_next - w > 2 * tol:
                mid = (w + w_next) / 2.0
                mid_snap = snapshot(mid, always_dsu)
                mid_snap.update(
                    {
                        "kind": "midpoint",
                        "label": f"开区间 ({w:g}, {w_next:g}) 中点",
                        "tie_groups": [],
                        "tie_count": sum(1 for row in mid_snap["standings"] if row["tied"]),
                    }
                )
                timeline.append(mid_snap)

    critical_points = [p for p in timeline if p["kind"] in ("critical", "endpoint")]
    midpoints = [p for p in timeline if p["kind"] == "midpoint"]
    crossing_count = sum(1 for p in critical_points if p["kind"] == "critical")

    contestants = [
        {
            "contestant": name,
            "base": base[name],
            "item_value": values[name],
            "has_item": has_item[name],
        }
        for name in names
    ]

    return {
        "scope": analysis["scope"],
        "scope_label": analysis["scope_label"],
        "item": item,
        "upper": upper,
        "tie_eps": tol,
        "formula": "总分(w) = 其余项目最终均分之和 + w × 该项目最终均分",
        "contestants": contestants,
        "always_tied_groups": always_groups,
        "critical_count": crossing_count,
        "timeline": timeline,
        "critical_points": critical_points,
        "midpoints": midpoints,
    }
