#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TopologyComputer - TDA 拓扑计算核心模块

使用持续同调 (Persistent Homology) 计算点云的拓扑特征:
- beta_0 (Betti-0): 连通分量数量
- beta_1 (Betti-1): 一维洞/循环数量

依赖: gudhi (pip install gudhi)

注: 此模块与模型无关，可同时用于 MIDILM 和 MeloFormer。
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class PersistenceDiagram:
    """持续同调图"""
    dimension: int
    pairs: np.ndarray  # shape: (n_pairs, 2) - [(birth, death), ...]

    @property
    def betti_number(self) -> int:
        """Betti 数 (infinite bars 的数量)"""
        if len(self.pairs) == 0:
            return 0
        return int(np.sum(self.pairs[:, 1] == np.inf))

    @property
    def total_persistence(self) -> float:
        """总持续度 (排除 infinite bars)"""
        if len(self.pairs) == 0:
            return 0.0
        finite_pairs = self.pairs[self.pairs[:, 1] != np.inf]
        if len(finite_pairs) == 0:
            return 0.0
        return float(np.sum(finite_pairs[:, 1] - finite_pairs[:, 0]))

    @property
    def max_persistence(self) -> float:
        """最大持续度"""
        if len(self.pairs) == 0:
            return 0.0
        finite_pairs = self.pairs[self.pairs[:, 1] != np.inf]
        if len(finite_pairs) == 0:
            return 0.0
        return float(np.max(finite_pairs[:, 1] - finite_pairs[:, 0]))

    def count_significant(self, threshold: float = 0.1) -> int:
        """统计显著特征数量 (持续度 > threshold)"""
        if len(self.pairs) == 0:
            return 0
        finite_pairs = self.pairs[self.pairs[:, 1] != np.inf]
        if len(finite_pairs) == 0:
            return 0
        persistence = finite_pairs[:, 1] - finite_pairs[:, 0]
        return int(np.sum(persistence > threshold))


class TopologyComputer:
    """
    TDA 拓扑计算核心模块

    使用 gudhi 计算持续同调。如果 gudhi 不可用，
    使用简化的近似算法作为 fallback。
    """

    def __init__(
        self,
        max_dimension: int = 1,
        max_edge_length: float = 2.0,
        use_fallback: bool = False,
    ):
        """
        Args:
            max_dimension: 最大同调维度 (0=连通性, 1=循环)
            max_edge_length: Rips 复形最大边长
            use_fallback: 强制使用 fallback 算法
        """
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self._gudhi = None
        self._use_fallback = use_fallback

        if not use_fallback:
            try:
                import gudhi
                self._gudhi = gudhi
            except ImportError:
                print("[警告] gudhi 未安装，使用 fallback 近似算法")
                print("       安装: pip install gudhi")
                self._use_fallback = True

    def compute_persistence(
        self,
        point_cloud: np.ndarray,
    ) -> List[PersistenceDiagram]:
        """
        计算点云的持续同调

        Args:
            point_cloud: (n_points, n_features) 点云数据

        Returns:
            各维度的持续同调图列表
        """
        if len(point_cloud) < 2:
            return [
                PersistenceDiagram(dim, np.empty((0, 2)))
                for dim in range(self.max_dimension + 1)
            ]

        if self._use_fallback:
            return self._compute_fallback(point_cloud)
        return self._compute_with_gudhi(point_cloud)

    def _compute_with_gudhi(self, point_cloud: np.ndarray) -> List[PersistenceDiagram]:
        """使用 gudhi 计算"""
        rips = self._gudhi.RipsComplex(
            points=point_cloud.tolist(),
            max_edge_length=self.max_edge_length
        )
        simplex_tree = rips.create_simplex_tree(max_dimension=self.max_dimension + 1)
        simplex_tree.compute_persistence()

        diagrams = []
        for dim in range(self.max_dimension + 1):
            intervals = simplex_tree.persistence_intervals_in_dimension(dim)
            pairs = np.array(intervals) if len(intervals) > 0 else np.empty((0, 2))
            diagrams.append(PersistenceDiagram(dimension=dim, pairs=pairs))

        return diagrams

    def _compute_fallback(self, point_cloud: np.ndarray) -> List[PersistenceDiagram]:
        """
        简化的 fallback 算法

        使用距离矩阵的特征值来近似拓扑特征
        """
        from scipy.spatial.distance import pdist, squareform

        dist_matrix = squareform(pdist(point_cloud))

        # H0: 使用并查集估计连通分量
        h0_pairs = self._estimate_h0(dist_matrix)

        # H1: 使用三角形检测估计循环
        h1_pairs = self._estimate_h1(dist_matrix) if self.max_dimension >= 1 else []

        diagrams = [
            PersistenceDiagram(
                dimension=0,
                pairs=np.array(h0_pairs) if h0_pairs else np.empty((0, 2))
            )
        ]

        if self.max_dimension >= 1:
            diagrams.append(
                PersistenceDiagram(
                    dimension=1,
                    pairs=np.array(h1_pairs) if h1_pairs else np.empty((0, 2))
                )
            )

        return diagrams

    def _estimate_h0(self, dist_matrix: np.ndarray) -> List[Tuple[float, float]]:
        """估计 H0 (连通分量)"""
        n = len(dist_matrix)

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((dist_matrix[i, j], i, j))
        edges.sort()

        # 并查集
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        pairs = []
        for dist, i, j in edges:
            if dist > self.max_edge_length:
                break
            if union(i, j):
                pairs.append((0.0, dist))

        # 最后一个连通分量活到无穷
        pairs.append((0.0, np.inf))

        return pairs

    def _estimate_h1(self, dist_matrix: np.ndarray) -> List[Tuple[float, float]]:
        """估计 H1 (循环) - 简化版"""
        n = len(dist_matrix)
        if n < 3:
            return []

        pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    d_ij = dist_matrix[i, j]
                    d_jk = dist_matrix[j, k]
                    d_ik = dist_matrix[i, k]

                    edges = sorted([d_ij, d_jk, d_ik])

                    if edges[2] <= self.max_edge_length:
                        birth = edges[2]
                        death = edges[2] * 1.1

                        if death - birth > 0.05:
                            pairs.append((birth, death))

        if pairs:
            pairs = list(set(pairs))
            pairs.sort(key=lambda x: x[1] - x[0], reverse=True)
            pairs = pairs[:10]

        return pairs

    def compute_betti_numbers(
        self,
        diagrams: List[PersistenceDiagram],
        threshold: float = 0.1,
    ) -> Tuple[int, ...]:
        """
        计算 Betti 数

        Args:
            diagrams: 持续同调图列表
            threshold: 过滤短特征的阈值

        Returns:
            (beta_0, beta_1, ...) Betti 数元组
        """
        betti = []
        for diag in diagrams:
            if len(diag.pairs) == 0:
                betti.append(0)
                continue

            count = np.sum(
                (diag.pairs[:, 0] <= threshold) &
                ((diag.pairs[:, 1] > threshold) | (diag.pairs[:, 1] == np.inf))
            )
            betti.append(int(count))

        return tuple(betti)

    def compute_distance(
        self,
        diag1: PersistenceDiagram,
        diag2: PersistenceDiagram,
        metric: str = 'wasserstein',
    ) -> float:
        """
        计算两个持续同调图的距离

        Args:
            diag1, diag2: 持续同调图
            metric: 'wasserstein' 或 'bottleneck'

        Returns:
            距离值
        """
        pairs1 = diag1.pairs[diag1.pairs[:, 1] != np.inf] \
            if len(diag1.pairs) > 0 else np.empty((0, 2))
        pairs2 = diag2.pairs[diag2.pairs[:, 1] != np.inf] \
            if len(diag2.pairs) > 0 else np.empty((0, 2))

        if len(pairs1) == 0 and len(pairs2) == 0:
            return 0.0

        if self._gudhi is not None and not self._use_fallback:
            try:
                if metric == 'wasserstein':
                    return self._gudhi.wasserstein.wasserstein_distance(
                        pairs1, pairs2, order=2
                    )
                else:
                    return self._gudhi.bottleneck_distance(pairs1, pairs2)
            except Exception:
                pass

        return self._simple_diagram_distance(pairs1, pairs2)

    def _simple_diagram_distance(
        self, pairs1: np.ndarray, pairs2: np.ndarray
    ) -> float:
        """简单的持续图距离近似"""
        pers1 = np.sum(pairs1[:, 1] - pairs1[:, 0]) if len(pairs1) > 0 else 0.0
        pers2 = np.sum(pairs2[:, 1] - pairs2[:, 0]) if len(pairs2) > 0 else 0.0

        n1, n2 = len(pairs1), len(pairs2)

        return abs(pers1 - pers2) + 0.1 * abs(n1 - n2)
