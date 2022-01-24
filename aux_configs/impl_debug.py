def get_auxiliary_configuration():
    # Adding extra keys here for logging:
    aux_config = {
        'show_progress': True,
        'jax_debug_nans': True,
        'save_logger_keys': ['Q', 'max_reward_rate', 'reward_rate', 'state_estimate_r', 'state_estimate_gamma', 'state_estimate_goal_prob', 'goal_values', 
        'goal_baseline', 'goal_r', 'goal_gamma', 'goal_transition', ]
        }
    return aux_config