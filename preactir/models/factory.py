from __future__ import annotations

from preactir.config import Config
from preactir.models.belief_encoder import SpatialBeliefEncoder
from preactir.models.world_model import ToolConditionedWorldModel
from preactir.models.verifier_model import LearnedTransitionVerifier


def build_belief_model(cfg: Config) -> SpatialBeliefEncoder:
    model_cfg = cfg.belief_model
    return SpatialBeliefEncoder(
        num_degradations=len(cfg.degradations.names),
        base_channels=int(model_cfg.base_channels),
        dropout=float(model_cfg.dropout),
        logvar_min=float(model_cfg.severity_logvar_min),
        logvar_max=float(model_cfg.severity_logvar_max),
        global_backbone=str(model_cfg.get("global_backbone", "none")),
        clip_checkpoint=model_cfg.get("clip_checkpoint"),
        clip_fusion_mode=str(model_cfg.get("clip_fusion_mode", "concat")),
    )


def build_world_model(cfg: Config) -> ToolConditionedWorldModel:
    model_cfg = cfg.world_model
    return ToolConditionedWorldModel(
        num_degradations=len(cfg.degradations.names),
        num_tools=len(cfg.tools.names),
        num_quality_metrics=4,
        base_channels=int(model_cfg.base_channels),
        feature_dim=int(model_cfg.feature_dim),
        action_embed_dim=int(model_cfg.action_embed_dim),
        hidden_dim=int(model_cfg.hidden_dim),
        dropout=float(model_cfg.dropout),
        logvar_min=float(model_cfg.logvar_min),
        logvar_max=float(model_cfg.logvar_max),
        predict_harm=bool(model_cfg.get("predict_harm", False)),
        predict_severe_harm=bool(model_cfg.get("predict_severe_harm", False)),
        predict_paper_quality=bool(model_cfg.get("predict_paper_quality", False)),
        trajectory_context_dim=int(model_cfg.get("trajectory_context_dim", 0)),
    )


def build_verifier_model(cfg: Config) -> LearnedTransitionVerifier:
    model_cfg = cfg.verifier_model
    return LearnedTransitionVerifier(
        num_degradations=len(cfg.degradations.names),
        num_tools=len(cfg.tools.names),
        base_channels=int(model_cfg.base_channels),
        feature_dim=int(model_cfg.feature_dim),
        action_embed_dim=int(model_cfg.action_embed_dim),
        hidden_dim=int(model_cfg.hidden_dim),
        dropout=float(model_cfg.dropout),
        predict_paper_quality=bool(model_cfg.get("predict_paper_quality", False)),
        trajectory_context_dim=int(model_cfg.get("trajectory_context_dim", 0)),
    )
