"""Checkpoint compatibility between legacy and current DBEFNet parameter names."""

FINAL_EXPERIMENT_PREFIXES = {
    "rs_branch.cnn_branch.": "rsi_encoder.cnn.features.",
    "rs_branch.transformer_branch.encoder.backbone.":
        "rsi_encoder.transformer.encoder.backbone.",
    "rs_branch.transformer_branch.decoder.":
        "rsi_encoder.transformer.aggregation.",
    "rs_branch.align_conv.": "rsi_encoder.projection.",
    "multi_scale_feature.branch1.": "mfem.branches.0.",
    "multi_scale_feature.branch2.": "mfem.branches.1.",
    "multi_scale_feature.branch3.": "mfem.branches.2.",
    "multi_scale_feature.fusion.": "mfem.fusion.",
    "multi_scale_feature.attention.": "mfem.attention.",
    "building_branch.cnn.": "building_encoder.features.",
    "building_branch.cbr_avg.": "afwm.avg_path.",
    "building_branch.cbr_max.": "afwm.max_path.",
    "building_branch.fusion.": "afwm.fusion.",
    "feature_fusion.rs_queries_bfs_attention.": "hafm.rs_queries_bf.",
    "feature_fusion.bfs_queries_rs_attention.": "hafm.bf_queries_rs.",
    "feature_fusion.bfs_self_attention.": "hafm.bf_self_attention.",
    "feature_fusion.rs_proj.": "hafm.rs_projection.",
    "feature_fusion.bfs_proj.": "hafm.bf_projection.",
    "feature_fusion.fusion_conv.": "hafm.fusion.",
    "feature_fusion.channel_attention.": "hafm.channel_attention.",
    "feature_fusion.norm.": "hafm.norm.",
    "seghead.layer4.": "segmentation_head.layer4.",
    "seghead.layer3.": "segmentation_head.layer3.refine.",
    "seghead.layer2.": "segmentation_head.layer2.refine.",
    "seghead.layer1.": "segmentation_head.layer1.refine.",
    "seghead.classifier.": "segmentation_head.classifier.",
}


def convert_final_experiment_state(state):
    """Convert `model_Fusion_v2.SemanticSegmentationModel` parameter names.

    Tensor values and ordering are retained. An unknown source name is rejected
    so a checkpoint from another experiment cannot be loaded silently.
    """
    converted = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        matches = [prefix for prefix in FINAL_EXPERIMENT_PREFIXES if key.startswith(prefix)]
        if len(matches) != 1:
            raise KeyError(f"Unrecognized model_Fusion_v2 parameter: {key}")
        prefix = matches[0]
        converted[FINAL_EXPERIMENT_PREFIXES[prefix] + key[len(prefix):]] = value
    return converted


def normalize_state_dict(state):
    """Return a state dictionary using public DBEFNet parameter names."""
    if not state:
        raise ValueError("Checkpoint state dictionary is empty")
    keys = [key.removeprefix("module.") for key in state]
    if any(key.startswith(("rs_branch.", "multi_scale_feature.", "feature_fusion."))
           for key in keys):
        return convert_final_experiment_state(state)
    if all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state
