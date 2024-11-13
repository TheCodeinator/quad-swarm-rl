python -m swarm_rl.train \
--env=quadrotor_multi --train_for_env_steps=100000000 --algo=APPO --use_rnn=False \
--num_workers=6 --num_envs_per_worker=4 --device=cpu --learning_rate=0.001 --ppo_clip_value=5.0 --recurrence=1 \
--nonlinearity=relu --actor_critic_share_weights=False --policy_initialization=xavier_uniform \
--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --rnn_size=256 \
--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 \
--with_pbt=False --normalize_input=False --normalize_returns=False --reward_clip=10 \
--quads_use_numba=True --save_milestones_sec=3600 --anneal_collision_steps=300000000 \
--replay_buffer_sample_prob=0.75 \
--quads_num_agents=6 \
--quads_mode=mix --quads_episode_duration=15.0 \
--quads_obs_repr=xyz_vxyz_R_omega \
--quads_neighbor_hidden_size=256 --quads_neighbor_obs_type=pos_vel --quads_collision_hitbox_radius=2.0 \
--quads_collision_falloff_radius=4.0 --quads_collision_reward=5.0 --quads_collision_smooth_max_penalty=10.0 \
--quads_neighbor_encoder_type=mean_embed --quads_neighbor_visible_num=5 \
--quads_sim2real=True \
--quads_use_obstacles=False --quads_obst_density=0.0 \
--quads_use_downwash=True \
--quads_view_mode=global --quads_render=True \
--experiment=train_multi_drone_real_256_256_extra_decoder_256_mean_embed_smaller_env \
--with_wandb=True --wandb_user=codeinator-personal \
--wandb_project=neuralfly \
