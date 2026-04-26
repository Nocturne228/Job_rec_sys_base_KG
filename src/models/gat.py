"""
Graph Attention Network (GAT) for Skill Node Importance Estimation.

Architecture:
- Multi-head self-attention over skill nodes in the Knowledge Graph
- Edge attributes (skill prerequisite relations) incorporated via edge embedding
- Outputs node importance scores for interpretable skill-gap diagnostics

Usage scenarios:
1. Skill Importance Scoring: compute centrality-aware scores for skill nodes
2. Skill Gap Prioritization: combine importance with user-skill delta to rank learning paths
3. Explainability: attention weights serve as interpretable edge importance signals

References:
- Velickovic et al., "Graph Attention Networks", ICLR 2018
- Brody et al., "How Attentive are Graph Attention Networks?", ICLR 2022 (GATv2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, softmax


# ============================================================================
# Single-Head Graph Attention Layer
# ============================================================================

class GATLayer(MessagePassing):
    """
    Single attention head for skill node importance estimation.

    Message passing formula (GATv2 variant):
        e_ij = a^T · LeakyReLU(W · [x_i || x_j || edge_attr_ij])
        α_ij = softmax_j(e_ij)
        h_i' = Σ_j α_ij · W · x_j

    Parameters:
        in_dim: input feature dimension per node
        out_dim: output feature dimension
        edge_attr_dim: dimension of edge attributes (prerequisite encoding)
        negative_slope: LeakyReLU slope (default 0.2)
        dropout: attention dropout rate (default 0.6)
    """

    def __init__(self, in_dim, out_dim, edge_attr_dim=16, negative_slope=0.2, dropout=0.6):
        super(GATLayer, self).__init__(aggr='add', flow='source_to_target')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_attr_dim = edge_attr_dim
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Learnable weight matrix W
        self.W = nn.Parameter(torch.Tensor(in_dim, out_dim))

        # Edge attribute projection: map edge_attr_dim → out_dim
        self.W_edge = nn.Parameter(torch.Tensor(edge_attr_dim, out_dim))

        # Attention vector a: dimension must match concatenated features
        # [x_i (out_dim) || x_j (out_dim) || edge_proj (out_dim)] = 3 * out_dim
        self.att = nn.Parameter(torch.Tensor(3 * out_dim, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.W_edge)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass for single attention head.

        Args:
            x: node features, shape [num_nodes, in_dim]
            edge_index: graph connectivity, shape [2, num_edges]
            edge_attr: edge features (skill prerequisite encoding), shape [num_edges, edge_attr_dim]

        Returns:
            out: attention-weighted node representations, shape [num_nodes, out_dim]
            attention_weights: (edge_index, α), for explainability
        """
        # Linear transform node features
        x = torch.matmul(x, self.W)

        # Compute attention scores via message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        return out

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        """
        Compute attention coefficients α_ij for each edge.

        Args:
            x_i: target node features after W transform, shape [num_edges, out_dim]
            x_j: source node features after W transform, shape [num_edges, out_dim]
            edge_attr: edge features, shape [num_edges, edge_attr_dim] or None
            index: edge target indices for softmax, shape [num_edges]
        """
        # Concatenate target + source + edge features
        if edge_attr is not None:
            edge_proj = torch.matmul(edge_attr, self.W_edge)
            cat = torch.cat([x_i, x_j, edge_proj], dim=-1)  # [num_edges, 2*out_dim + edge_attr_dim]
        else:
            cat = torch.cat([x_i, x_j], dim=-1)  # [num_edges, 2*out_dim]

        # Attention score: LeakyReLU(a^T · cat)
        e = F.leaky_relu(torch.matmul(cat, self.att).squeeze(-1), self.negative_slope)

        # Softmax over all neighbors of node i
        alpha = softmax(e, index, ptr, size_i)

        # Apply dropout to attention weights (for regularization)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Message = α_ij · W · x_j
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_dim} → {self.out_dim})'


# ============================================================================
# Multi-Head Graph Attention Layer
# ============================================================================

class MultiHeadGATLayer(nn.Module):
    """
    Multi-head GAT layer: concatenates outputs from H independent attention heads,
    then applies output projection.

    Concatenation → Linear projection avoids capacity bottlenecks that arise with
    averaging in the hidden layers (see GAT paper §2.2).

    Parameters:
        in_dim: input feature dimension per node
        hid_dim: per-head output dimension
        out_dim: final projection dimension
        num_heads: number of attention heads (default 4)
        edge_attr_dim: dimension of edge attributes (default 16)
        negative_slope: LeakyReLU slope (default 0.2)
        dropout: attention dropout (default 0.6)
        concat_heads: if True, concat all heads; if False, average them
    """

    def __init__(self, in_dim, hid_dim, out_dim, num_heads=4, edge_attr_dim=16,
                 negative_slope=0.2, dropout=0.6, concat_heads=True):
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads

        # All heads share the same input dimension — this is the core design
        # of multi-head attention: parallel heads operating on identical input,
        # then concatenated/averaged at the output side.
        self.heads = nn.ModuleList([
            GATLayer(in_dim, hid_dim, edge_attr_dim, negative_slope, dropout)
            for _ in range(num_heads)
        ])

        if concat_heads:
            self.proj = nn.Linear(hid_dim * num_heads, out_dim)
        else:
            # When averaging, all heads output same dim
            self.proj = nn.Linear(hid_dim, out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward: multi-head attention + concat + projection.

        Returns:
            out: projected multi-head representation, shape [num_nodes, out_dim]
        """
        if self.concat_heads:
            # Concatenate outputs from all heads: [num_nodes, num_heads * hid_dim]
            outs = [head(x, edge_index, edge_attr) for head in self.heads]
            out = torch.cat(outs, dim=-1)
        else:
            # Average: each head outputs [num_nodes, hid_dim], stack then mean
            outs = torch.stack(
                [head(x, edge_index, edge_attr) for head in self.heads],
                dim=0
            )
            out = outs.mean(dim=0)

        out = F.elu(self.proj(out))
        return out


# ============================================================================
# Full GAT for Skill Node Importance
# ============================================================================

class GraphAttentionNetwork(nn.Module):
    """
    2-layer GAT for skill importance scoring and explainable skill-gap analysis.

    Architecture:
        Input features → MultiHead GAT (concat, H heads) → ELU →
        Linear projection (single head) → Skill importance score

    Training objective:
        - Skill importance is learned via regression against a pseudo-label
          combining graph centrality (PageRank) + job skill frequency
        - Loss = MSE(ŝ, s_target) + λ * entropy(α) to encourage focused attention

    Parameters:
        num_skill_features: input feature dimension per skill node (default 16)
        hidden_dim: per-head hidden dimension (default 32)
        num_heads: number of attention heads in first layer (default 4)
        edge_attr_dim: edge feature dimension (default 16)
        dropout: attention dropout rate (default 0.6)
    """

    def __init__(self, num_skill_features=16, hidden_dim=32, num_heads=4,
                 edge_attr_dim=16, dropout=0.6):
        super(GraphAttentionNetwork, self).__init__()

        self.num_skill_features = num_skill_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Layer 1: Multi-head GAT (concatenation)
        self.gat1 = MultiHeadGATLayer(
            in_dim=num_skill_features,
            hid_dim=hidden_dim,
            out_dim=hidden_dim * num_heads,  # Project to match concat output
            num_heads=num_heads,
            edge_attr_dim=edge_attr_dim,
            dropout=dropout,
            concat_heads=True
        )

        # Layer 2: Single-head GAT (averaging) for final score
        self.gat2 = MultiHeadGATLayer(
            in_dim=hidden_dim * num_heads,
            hid_dim=hidden_dim,
            out_dim=1,  # Single scalar: skill importance score
            num_heads=1,
            edge_attr_dim=edge_attr_dim,
            dropout=dropout,
            concat_heads=False  # Average for stability
        )

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for head in self.gat1.heads:
            head.reset_parameters()
        self.gat1.proj.reset_parameters()
        for head in self.gat2.heads:
            head.reset_parameters()
        self.gat2.proj.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass: 2-layer multi-head GAT.

        Args:
            x: skill node features, shape [num_skills, num_skill_features]
            edge_index: prerequisite graph edges, shape [2, num_edges]
            edge_attr: prerequisite encoding (e.g., [required, nice-to-have]),
                       shape [num_edges, edge_attr_dim]

        Returns:
            importance: skill importance scores, shape [num_skills, 1]
        """
        # Layer 1: Multi-head attention with concat + ELU
        h = self.gat1(x, edge_index, edge_attr)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2: Single-head attention for scalar score
        importance = self.gat2(h, edge_index, edge_attr)

        return importance  # [num_skills, 1]

    def get_attention_weights(self, x, edge_index, edge_attr=None):
        """
        Extract attention weights from the first layer for explainability.
        Used to visualize which skill prerequisites the model considers most important.

        Returns:
            Dict mapping (node_i, node_j) → α_ij for top-k attention edges
        """
        # Re-project node features for layer 1
        x_transformed = []
        for head in self.gat1.heads:
            x_transformed.append(torch.matmul(x, head.W))

        # Extract attention from each head
        attention_dict = {}
        for head_idx, head in enumerate(self.gat1.heads):
            # Recompute attention scores (same logic as message())
            cat = self._assemble_cat(x_transformed[head_idx], edge_index, x, edge_attr, head)
            e = F.leaky_relu(torch.matmul(cat, head.att).squeeze(-1), head.negative_slope)

            # Store raw attention scores for this head
            for idx, (i, j) in enumerate(zip(edge_index[1].tolist(), edge_index[0].tolist())):
                key = (i, j, head_idx)
                attention_dict[key] = torch.sigmoid(e[idx]).item()

        return attention_dict

    def _assemble_cat(self, x_w, edge_index, x, edge_attr, head):
        """Helper to build concatenated feature tensors for attention computation."""
        x_j = x[edge_index[0]]  # source nodes
        x_i = x[edge_index[1]]  # target nodes
        # Re-apply head's W to get transformed features
        x_w_i = torch.matmul(x_i, head.W)
        x_w_j = torch.matmul(x_j, head.W)
        if edge_attr is not None:
            edge_proj = torch.matmul(edge_attr, head.W_edge)
            return torch.cat([x_w_i, x_w_j, edge_proj], dim=-1)
        return torch.cat([x_w_i, x_w_j], dim=-1)

    def compute_node_importance(self, x, edge_index, edge_attr=None):
        """
        Convenience wrapper: returns normalized importance scores [0, 1].

        Usage in ranking pipeline:
            scores = gat.compute_node_importance(skill_features, kg_edge_index, kg_edge_attr)
            top_skills = torch.topk(scores.squeeze(), k=5)
        """
        raw_scores = self.forward(x, edge_index, edge_attr)
        normalized = torch.sigmoid(raw_scores.squeeze(-1))  # [num_skills]
        return normalized

    @staticmethod
    def bpr_loss(y_pos, y_neg, weight_decay=0.0, params=None):
        """
        Bayesian Personalized Ranking loss (for training GAT on skill pairs).

        L = -Σ_(i,j,k) ln σ(ŷ_ji - ŷ_ki) + λ||Θ||²
        where j is a required skill, k is a non-required skill for item i.

        Args:
            y_pos: positive skill scores, shape [batch_size]
            y_neg: negative skill scores, shape [batch_size]
            weight_decay: λ regularization strength
            params: model parameters for L2 penalty
        """
        diff = y_pos - y_neg
        bpr = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

        if weight_decay > 0 and params is not None:
            l2 = sum(p.pow(2).sum() for p in params)
            bpr += weight_decay * l2

        return bpr

    def train_with_pseudo_labels(self, x, edge_index, edge_attr,
                                 pseudo_labels, n_epochs=200,
                                 lr=1e-3, weight_decay=1e-4,
                                 device="cpu", verbose=True):
        """
        Train GAT using pseudo-labels derived from graph centrality + job frequency.

        Pseudo-label construction (done by caller):
            score(s) = α * PageRank(s) + β * normalized_job_freq(s)
        This creates a regression target: skills that are central in the KG
        and frequently mentioned in jobs should receive higher importance.

        Loss = MSE(sigmoid(GAT(x)) - pseudo_labels) + λ * ||Θ||²

        Args:
            x: skill node features, shape [num_skills, num_features]
            edge_index: prerequisite edges, shape [2, num_edges]
            edge_attr: edge attributes, shape [num_edges, edge_attr_dim]
            pseudo_labels: target importance scores, shape [num_skills], range [0, 1]
            n_epochs: training epochs
            lr: learning rate
            weight_decay: L2 regularization strength
            device: "cpu" or "cuda"
            verbose: print training progress

        Returns:
            dict with training history
        """
        self.train()
        self.to(device)
        x = x.to(device)
        edge_index = edge_index.to(device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        pseudo_labels = pseudo_labels.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        history = {"loss": [], "mse": [], "reg": []}
        best_loss = float("inf")
        best_state = None

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            raw_scores = self.forward(x, edge_index, edge_attr)  # [num_skills, 1]
            pred = torch.sigmoid(raw_scores.squeeze(-1))  # [num_skills]

            mse = F.mse_loss(pred, pseudo_labels)

            # L2 regularization
            l2 = sum(p.pow(2).sum() for p in self.parameters())
            reg = weight_decay * l2

            total_loss = mse + reg
            total_loss.backward()
            optimizer.step()

            history["loss"].append(total_loss.item())
            history["mse"].append(mse.item())
            history["reg"].append(reg.item())

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_state = {k: v.clone() for k, v in self.state_dict().items()}

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  [GAT] Epoch {epoch+1}/{n_epochs}, "
                      f"Loss={total_loss.item():.4f} (MSE={mse.item():.4f}, "
                      f"Reg={reg.item():.4f})")

        # Restore best weights
        if best_state is not None:
            self.load_state_dict(best_state)

        self.eval()
        if verbose:
            print(f"  [GAT] Training done. Best loss={best_loss:.4f}")

        return history


# ============================================================================
# Skill Node Feature Builder
# ============================================================================

class SkillFeatureBuilder:
    """
    Constructs feature vectors for skill nodes in the KG.

    Feature engineering (num_skill_features=16 by default):
        [0:4]   — Job Frequency: how many jobs mention this skill
        [4:8]   — Prerequisite Count: in-degree in the prerequisite subgraph
        [8:10]  — Centrality Scores: power_iteration PageRank + degree ratio
        [10:12] — Level Difficulty: category-based difficulty encoding
        [12:14] — Trend Score: frequency growth proxy + SO mention proxy
        [14:16] — Salary Correlation: difficulty × frequency proxy + salary band

    For the demo, all features are derived deterministically from the mock KG
    data (skill frequency, degree, and category heuristics). In production,
    these are replaced by real Neo4j aggregation queries and external data.
    """

    # Category → difficulty (heuristic proxy for Level Difficulty)
    CATEGORY_DIFFICULTY = {
        "soft": 0.2,
        "data": 0.4,
        "database": 0.5,
        "frontend": 0.5,
        "backend": 0.6,
        "programming": 0.6,
        "cloud": 0.7,
        "devops": 0.75,
        "ml": 0.8,
    }

    # Category → market demand proxy (Trend Score)
    CATEGORY_TREND = {
        "soft": 0.5,
        "data": 0.7,
        "database": 0.5,
        "frontend": 0.6,
        "backend": 0.7,
        "programming": 0.7,
        "cloud": 0.85,
        "devops": 0.9,
        "ml": 0.95,
    }

    # Category → salary impact proxy (higher = stronger correlation with salary)
    CATEGORY_SALARY_IMPACT = {
        "soft": 0.3,
        "data": 0.6,
        "database": 0.4,
        "frontend": 0.3,
        "backend": 0.5,
        "programming": 0.4,
        "cloud": 0.7,
        "devops": 0.8,
        "ml": 0.9,
    }

    def __init__(self, num_features=16):
        self.num_features = num_features

    def build_from_kg_data(self, kg_data):
        """
        Build feature matrix from Knowledge Graph data.

        Args:
            kg_data: dict with structure:
                {
                    "skills": [{"name": str, "level": int, "domain": str}, ...],
                    "prerequisites": [(skill_a, skill_b, weight), ...],
                    "job_associations": {job_id: [skill_names]},
                }

        Returns:
            x: feature matrix, shape [num_skills, num_features]
            edge_index: prerequisite edges, shape [2, num_edges]
            edge_attr: edge weights, shape [num_edges, 1]
            skill_id_map: {skill_name: node_id} mapping
        """
        skills = kg_data["skills"]
        prereqs = kg_data["prerequisites"]
        job_assocs = kg_data["job_associations"]

        n_skills = len(skills)
        skill_id_map = {s["name"]: i for i, s in enumerate(skills)}

        # --- Job frequency ---
        skill_freq = {name: 0 for name in skill_id_map}
        for job_skills in job_assocs.values():
            for skill_name in job_skills:
                if skill_name in skill_freq:
                    skill_freq[skill_name] += 1
        max_freq = max(skill_freq.values(), default=1) + 1e-8

        # --- Degree statistics ---
        in_degree = {name: 0 for name in skill_id_map}
        out_degree = {name: 0 for name in skill_id_map}

        edge_list_src = []
        edge_list_dst = []
        edge_weights = []

        for src, dst, weight in prereqs:
            if src in skill_id_map and dst in skill_id_map:
                edge_list_src.append(skill_id_map[src])
                edge_list_dst.append(skill_id_map[dst])
                edge_weights.append([weight])
                in_degree[dst] += 1
                out_degree[src] += 1

        if edge_list_src:
            edge_index = torch.tensor([edge_list_src, edge_list_dst], dtype=torch.long)
            edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        max_in = max(max(in_degree.values()), 1)
        max_out = max(max(out_degree.values()), 1)

        # --- Power-iteration PageRank (lightweight, no NetworkX dependency) ---
        n = n_skills
        if edge_list_src:
            # Build normalized column-stochastic transition matrix as dicts
            out_counts = {i: 0 for i in range(n)}
            for s in edge_list_src:
                out_counts[s] += 1

            pr = [1.0 / n] * n
            damping = 0.85
            for _ in range(50):
                new_pr = [damping * pr[j] / out_counts[j] if out_counts[j] > 0 else 0.0
                          for j in range(n)]
                pr_next = [0.0] * n
                for s_val, d_val in zip(edge_list_src, edge_list_dst):
                    pr_next[d_val] += new_pr[s_val]
                teleport = (1.0 - damping) / n
                pr = [pr_next[i] + teleport for i in range(n)]
                # Normalize
                total = sum(pr)
                if total > 0:
                    pr = [p / total for p in pr]
        else:
            pr = [1.0 / n] * n

        max_pr = max(pr) + 1e-8
        # Betweenness proxy: total degree (in + out) / max_total_degree
        max_total_deg = max(max(in_degree[s] + out_degree[s] for s in skill_id_map), 1)

        # --- Feature matrix ---
        x = torch.zeros(n_skills, self.num_features)
        # Build skill category lookup: domain → lower-case string
        skill_category = {s["name"]: s.get("domain", "programming").lower()
                          for s in skills}

        for skill_name, idx in skill_id_map.items():
            freq = skill_freq[skill_name]
            ind = in_degree[skill_name]
            opd = out_degree[skill_name]
            cat = skill_category.get(skill_name, "programming")

            # [0:4] Job Frequency (raw + capped + per-relation-type variants)
            x[idx, 0] = freq / max_freq
            x[idx, 1] = min(freq, 10) / 10.0
            # Frequency split by relation type: skills that are more often "required"
            # vs "nice-to-have" — proxy: normalize in-degree vs out-degree ratio
            total_deg = ind + opd
            x[idx, 2] = ind / max_in       # in-degree as "prerequisite fan-in"
            x[idx, 3] = opd / max_out       # out-degree as "prerequisite fan-out"

            # [4:8] Centrality Scores (PageRank + betweenness proxy + variants)
            x[idx, 4] = pr[idx] / max_pr     # Normalized PageRank
            x[idx, 5] = (ind + opd) / max_total_deg  # Degree centrality
            # Harmonic centrality proxy: skills with balanced in/out are more central
            if ind + opd > 0:
                x[idx, 6] = min(ind, opd) / (ind + opd)  # Balance ratio
            else:
                x[idx, 6] = 0.0
            x[idx, 7] = x[idx, 4] * x[idx, 5]  # PR × degree interaction

            # [8:10] Level Difficulty (category-based + frequency-adjusted)
            base_diff = self.CATEGORY_DIFFICULTY.get(cat, 0.5)
            # Frequently referenced skills tend to be harder (more prerequisites)
            freq_adj = min(freq / max_freq * 0.2, 0.2)
            x[idx, 8] = min(base_diff + freq_adj, 1.0)
            x[idx, 9] = freq / max_freq  # frequency as difficulty variance proxy

            # [10:14] Trend Score (category demand + frequency growth proxy)
            base_trend = self.CATEGORY_TREND.get(cat, 0.5)
            x[idx, 10] = base_trend
            # Emerging skills proxy: high degree / high frequency but not yet ubiquitous
            x[idx, 11] = min(base_trend * x[idx, 4], 1.0)  # trend × PR interaction
            x[idx, 12] = x[idx, 2] * (1.0 - x[idx, 0])  # in-degree × (1-freq) → emerging
            x[idx, 13] = base_trend * 0.5  # baseline SO mention proxy

            # [14:16] Salary Correlation (category impact + interaction)
            base_salary = self.CATEGORY_SALARY_IMPACT.get(cat, 0.4)
            x[idx, 14] = base_salary
            x[idx, 15] = base_salary * (0.5 + 0.5 * x[idx, 4])  # salary × PR

        return x, edge_index, edge_attr, skill_id_map
