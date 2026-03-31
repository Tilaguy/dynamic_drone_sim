import math
import numpy as np
from collections import defaultdict

class FormationManager:
    """
    Manages the initial positions and desired formation references
    for a hierarchical tethered multi-robot system.
    """
    def __init__(self, N: int, edges: list[tuple[int, int]]):
        if N < 1:
            raise ValueError("The number of robots <N> must be at least 1.")

        self.N = N
        self.num_agents = N + 1
        self.edges = edges
        self.depth = int(math.log2(self.num_agents))

        self.cables = []

        # Pre-compute the level grouping ONCE to save CPU time during simulation
        self.levels = defaultdict(list)
        for i in range(1, self.num_agents): # Skip index 0 (the load)
            lvl = int(math.log2(i + 1))
            self.levels[lvl].append(i)

    def exponential_cables(self, base_length, alpha=0.7):
        for lvl in range(1, len(self.levels) + 1):
            count = 2**lvl
            self.cables.extend([base_length*(alpha**lvl)]*count)

        return self.cables

    def estimate_max_depth(self, min_sep):
        """
        Rough physical limit for a binary tree without cable crossing.
        """
        L = np.mean(self.cables)

        if min_sep <= 0:
            return np.inf

        return int(math.log2(L / min_sep))

    def generate_initial_position(self, load_position, base_sep=0.5):
        agent_pos = np.zeros((self.num_agents, 2))
        agent_pos[0] = load_position

        # adjacency
        children = defaultdict(list)
        for p, c in self.edges:
            children[p].append(c)

        # ---------- compute subtree width ----------
        width = {}

        def subtree_width(node):

            if node not in children:
                width[node] = base_sep
                return width[node]

            w = sum(subtree_width(c) for c in children[node])

            width[node] = max(w, base_sep)
            return width[node]

        subtree_width(0)

        # ---------- recursive placement ----------
        def place(node):

            if node not in children:
                return

            xp, yp = agent_pos[node]

            kids = children[node]
            total_w = sum(width[c] for c in kids)

            offset = -total_w / 2

            for c in kids:

                L = self.cables[c-1]

                d = offset + width[c]/2

                # enforce geometric feasibility
                d = np.clip(d, -0.95*L, 0.95*L)

                xc = xp + d
                yc = yp + math.sqrt(L**2 - d**2)

                agent_pos[c] = [xc, yc]

                place(c)

                offset += width[c]

        place(0)

        return agent_pos
