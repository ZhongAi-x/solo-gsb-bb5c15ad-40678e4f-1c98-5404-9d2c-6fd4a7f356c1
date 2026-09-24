"""项目权重临界点分析的单元测试。"""
import os
import sys
import tempfile
import unittest

_tmp = tempfile.mkdtemp(prefix="calib_test_")
os.environ.setdefault("CALIB_DATA_DIR", _tmp)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.scoring import analyze  # noqa: E402
from app.weight_analysis import WeightAnalysisError, weight_analysis  # noqa: E402


def raw_analysis(rows):
    """rows: (contestant, item, score)，单一评委/项目；裁决全部保留原分。

    保留原分后最终分即原始分，便于用精确数值验证临界点。
    """
    judge_for_item = {}
    records = []
    for idx, (c, item, score) in enumerate(rows, start=1):
        judge = judge_for_item.setdefault(item, f"J-{item}")
        records.append(
            {"id": idx, "contestant": c, "judge": judge, "item": item, "score": float(score)}
        )
    decisions = {r["id"]: (True, "") for r in records}
    return analyze(records, decisions, scope="judge")


def by_name(point):
    return {r["contestant"]: r for r in point["standings"]}


class TestWeightAnalysis(unittest.TestCase):
    def test_crossings_ordering_and_ranks(self):
        # value(X)/base(Y): A=10/2, B=4/8, C=6/5
        # 交点：A-C w=0.75，A-B w=1，B-C w=1.5（均在 [0,3] 内）。
        analysis = raw_analysis(
            [
                ("A", "X", 10), ("B", "X", 4), ("C", "X", 6),
                ("A", "Y", 2), ("B", "Y", 8), ("C", "Y", 5),
            ]
        )
        res = weight_analysis(analysis, "X", 3)
        self.assertEqual(res["critical_count"], 3)
        self.assertEqual(res["always_tied_groups"], [])
        weights = [p["weight"] for p in res["critical_points"]]
        self.assertEqual(weights[0], 0.0)
        self.assertAlmostEqual(weights[1], 0.75)
        self.assertAlmostEqual(weights[2], 1.0)
        self.assertAlmostEqual(weights[3], 1.5)
        self.assertAlmostEqual(weights[4], 3.0)
        # 4 个相邻开区间中点：(0,.75) (.75,1) (1,1.5) (1.5,3)
        self.assertEqual(len(res["midpoints"]), 4)
        # 时间线按倍率严格递增、临界点与中点交错。
        tw = [p["weight"] for p in res["timeline"]]
        self.assertEqual(tw, sorted(tw))
        kinds = [p["kind"] for p in res["timeline"]]
        self.assertEqual(kinds[0], "endpoint")
        self.assertEqual(kinds[-1], "endpoint")
        self.assertTrue(all(k in ("critical", "midpoint") for k in kinds[1:-1]))

        # w=0：B(8) > C(5) > A(2)
        start = by_name(res["critical_points"][0])
        self.assertEqual([start[c]["rank"] for c in "ABC"], [3, 1, 2])
        self.assertFalse(any(r["tied"] for r in start.values()))

        # w=0.75：A=C=9.5 并列第 2，B=11 第 1
        p75 = next(p for p in res["critical_points"] if abs(p["weight"] - 0.75) < 1e-9)
        rows = by_name(p75)
        self.assertEqual(rows["B"]["rank"], 1)
        self.assertEqual(rows["A"]["rank"], 2)
        self.assertEqual(rows["C"]["rank"], 2)
        self.assertTrue(rows["A"]["tied"] and rows["C"]["tied"])
        self.assertFalse(rows["B"]["tied"])
        self.assertEqual(
            [g["contestants"] for g in p75["tie_groups"]],
            [["A", "C"]],
        )

        # w=1：A=B=12 并列第 1，C=11 第 2
        p1 = next(p for p in res["critical_points"] if abs(p["weight"] - 1.0) < 1e-9)
        rows = by_name(p1)
        self.assertEqual(rows["A"]["rank"], 1)
        self.assertEqual(rows["B"]["rank"], 1)
        self.assertEqual(rows["C"]["rank"], 2)
        self.assertEqual(
            [g["contestants"] for g in p1["tie_groups"]],
            [["A", "B"]],
        )

        # w=1.5：A=17 第 1，B=C=14 并列第 2
        p15 = next(p for p in res["critical_points"] if abs(p["weight"] - 1.5) < 1e-9)
        rows = by_name(p15)
        self.assertEqual(rows["A"]["rank"], 1)
        self.assertEqual(rows["B"]["rank"], 2)
        self.assertEqual(rows["C"]["rank"], 2)

        # 中点无交点（且无始终并列选手），不应出现并列。
        for mid in res["midpoints"]:
            self.assertFalse(any(r["tied"] for r in mid["standings"]))
        # 最后一个中点 w=2.25：A=24.5 > C=18.5 > B=17
        last = by_name(res["midpoints"][-1])
        self.assertAlmostEqual(res["midpoints"][-1]["weight"], 2.25)
        self.assertEqual([last[c]["rank"] for c in "ABC"], [1, 3, 2])

        # 上限端点 w=3：A=32 > C=23 > B=20
        end = by_name(res["critical_points"][-1])
        self.assertEqual([end[c]["rank"] for c in "ABC"], [1, 3, 2])
        self.assertAlmostEqual(end["A"]["total"], 32.0)

    def test_crossing_outside_range_excluded(self):
        # A=10/0, B=0/8：唯一交点 w=0.8，在 [0,1] 内。
        analysis = raw_analysis([("A", "X", 10), ("B", "X", 0), ("B", "Y", 8)])
        res = weight_analysis(analysis, "X", 1)
        self.assertEqual(res["critical_count"], 1)
        self.assertAlmostEqual(res["critical_points"][1]["weight"], 0.8)
        # A=2/0, B=0/5：交点 w=2.5，upper=2 时落在区间外，区间内无内部临界点。
        analysis2 = raw_analysis([("A", "X", 2), ("B", "X", 0), ("B", "Y", 5)])
        res3 = weight_analysis(analysis2, "X", 2)
        self.assertEqual(res3["critical_count"], 0)
        weights = [p["weight"] for p in res3["critical_points"]]
        self.assertEqual(weights, [0.0, 2.0])
        # 区间内无交点 → 唯一开区间 (0,2) 的中点 w=1。
        self.assertEqual(len(res3["midpoints"]), 1)
        self.assertAlmostEqual(res3["midpoints"][0]["weight"], 1.0)

    def test_same_position_intersections_merged(self):
        # 三条线在 w=1 同时相交：A=6/4, B=4/6, C=2/8（w=1 时总分都是 10）。
        analysis = raw_analysis(
            [
                ("A", "X", 6), ("B", "X", 4), ("C", "X", 2),
                ("A", "Y", 4), ("B", "Y", 6), ("C", "Y", 8),
            ]
        )
        res = weight_analysis(analysis, "X", 3)
        self.assertEqual(res["critical_count"], 1)
        point = next(p for p in res["critical_points"] if p["kind"] == "critical")
        self.assertAlmostEqual(point["weight"], 1.0)
        self.assertEqual(
            sorted(g["contestants"] for g in point["tie_groups"]),
            [["A", "B", "C"]],
        )
        rows = by_name(point)
        self.assertEqual({rows[c]["rank"] for c in "ABC"}, {1})
        self.assertTrue(all(rows[c]["tied"] for c in "ABC"))

    def test_always_tied_produces_no_intersection(self):
        # A、B 的两个项目分完全相同 → 始终并列；D 与它们在 w=1 相交。
        analysis = raw_analysis(
            [
                ("A", "X", 6), ("B", "X", 6), ("D", "X", 2),
                ("A", "Y", 4), ("B", "Y", 4), ("D", "Y", 8),
            ]
        )
        res = weight_analysis(analysis, "X", 3)
        self.assertEqual(res["critical_count"], 1)
        self.assertEqual(
            res["always_tied_groups"],
            [{"contestants": ["A", "B"], "size": 2}],
        )
        point = res["critical_points"][1]
        self.assertAlmostEqual(point["weight"], 1.0)
        # A≡B（始终并列），A 与 D 相交，故交点处三人全部并列。
        self.assertEqual(
            sorted(g["contestants"] for g in point["tie_groups"]),
            [["A", "B", "D"]],
        )
        # 中点处只有 A/B 因始终并列而并列，且不被当作交点。
        mid = res["midpoints"][0]
        rows = by_name(mid)
        self.assertTrue(rows["A"]["tied"] and rows["B"]["tied"] and rows["A"]["always_tied"])
        self.assertFalse(rows["D"]["tied"])
        self.assertEqual(mid["tie_groups"], [])
        # w=0：D=8 第 1，A=B=4 并列第 2。
        start = by_name(res["critical_points"][0])
        self.assertEqual(start["D"]["rank"], 1)
        self.assertEqual(start["A"]["rank"], 2)
        self.assertEqual(start["B"]["rank"], 2)

    def test_missing_item_contributes_zero(self):
        # M 在项目 X 无任何评分：value=0，总分不随 w 变化。
        analysis = raw_analysis(
            [
                ("A", "X", 9), ("B", "X", 6),
                ("A", "Y", 5), ("B", "Y", 8), ("M", "Y", 7),
            ]
        )
        res = weight_analysis(analysis, "X", 3)
        contestants = {c["contestant"]: c for c in res["contestants"]}
        self.assertTrue(contestants["A"]["has_item"])
        self.assertFalse(contestants["M"]["has_item"])
        self.assertEqual(contestants["M"]["item_value"], 0.0)
        self.assertEqual(contestants["M"]["base"], 7.0)
        # M 各点总分恒为 7。
        for point in res["timeline"]:
            self.assertAlmostEqual(by_name(point)["M"]["total"], 7.0)
        # A(9w+5) 与 M(7) 的交点 w=2/9≈0.2222；B(6w+8) 与 M 交点 w=-1/6（区间外）。
        crits = [p for p in res["critical_points"] if p["kind"] == "critical"]
        self.assertEqual(len(crits), 2)  # A-M 与 A-B(w=1)
        am = next(p for p in crits if any("M" in g["contestants"] for g in p["tie_groups"]))
        self.assertAlmostEqual(am["weight"], 2 / 9)

    def test_endpoint_ties_included(self):
        # A=2/8, B=6/4：w=1 相交；再构造在 w=0 就同分的一对（base 相等）。
        analysis = raw_analysis(
            [
                ("A", "X", 2), ("B", "X", 6),
                ("A", "Y", 8), ("B", "Y", 4),
                ("C", "X", 5), ("C", "Y", 8),  # base 与 A 相同 → w=0 处 A=C
            ]
        )
        res = weight_analysis(analysis, "X", 1)
        start = res["critical_points"][0]
        self.assertEqual(start["weight"], 0.0)
        self.assertIn(
            ["A", "C"],
            [sorted(g["contestants"]) for g in start["tie_groups"]],
        )
        # 上限 w=1：A=B=10 相交于端点，应作为 endpoint 临界点展示。
        end = res["critical_points"][-1]
        self.assertAlmostEqual(end["weight"], 1.0)
        self.assertIn(
            ["A", "B"],
            [sorted(g["contestants"]) for g in end["tie_groups"]],
        )
        # 上限即交点时不再额外产生 critical 点（同位置合并到端点）。
        # w=0 的 A=C 与 w=1 的 A=B 都吸附到端点，故区间内部临界点数为 0。
        self.assertEqual(res["critical_count"], 0)
        self.assertEqual(
            [p["kind"] for p in res["critical_points"]],
            ["endpoint", "endpoint"],
        )
        self.assertEqual(len(res["midpoints"]), 1)
        self.assertAlmostEqual(res["midpoints"][0]["weight"], 0.5)
        mid_rows = by_name(res["midpoints"][0])
        # w=0.5：C=10.5 > A=9 > B=7
        self.assertEqual([mid_rows[c]["rank"] for c in "ABC"], [2, 3, 1])

    def test_validation_errors(self):
        analysis = raw_analysis([("A", "X", 9), ("A", "Y", 5)])
        with self.assertRaises(WeightAnalysisError):
            weight_analysis(analysis, "Z", 2)
        with self.assertRaises(WeightAnalysisError):
            weight_analysis(analysis, "X", 0.9)
        with self.assertRaises(WeightAnalysisError):
            weight_analysis(analysis, "X", 5.1)
        with self.assertRaises(WeightAnalysisError):
            weight_analysis(analysis, "X", float("nan"))
        with self.assertRaises(WeightAnalysisError):
            weight_analysis(analysis, "X", float("inf"))
        with self.assertRaises(WeightAnalysisError):
            weight_analysis(analysis, "X", "2")  # type: ignore[arg-type]

    def test_uses_effective_item_averages(self):
        # 不带裁决时用校准分；全部保留原分时用原分，两者结果应不同。
        records = [
            {"id": 1, "contestant": "A", "judge": "J1", "item": "X", "score": 9.0},
            {"id": 2, "contestant": "B", "judge": "J1", "item": "X", "score": 4.0},
            {"id": 3, "contestant": "A", "judge": "J2", "item": "Y", "score": 5.0},
            {"id": 4, "contestant": "B", "judge": "J2", "item": "Y", "score": 8.0},
        ]
        cal = analyze(records, {}, scope="judge")
        raw = analyze(records, {r["id"]: (True, "") for r in records}, scope="judge")
        r_cal = weight_analysis(cal, "X", 3)
        r_raw = weight_analysis(raw, "X", 3)
        self.assertNotEqual(
            [p["weight"] for p in r_cal["timeline"]],
            [p["weight"] for p in r_raw["timeline"]],
        )
        # 原始分下 A(9w+5)、B(4w+8) 交点 w=0.6。
        crits = [p["weight"] for p in r_raw["critical_points"] if p["kind"] == "critical"]
        self.assertEqual(len(crits), 1)
        self.assertAlmostEqual(crits[0], 0.6)


if __name__ == "__main__":
    unittest.main()
