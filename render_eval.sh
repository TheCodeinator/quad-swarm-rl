python -m swarm_rl.enjoy \
--algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 \
--quads_use_numba=True --train_dir=./train_dir --experiment=neuralfly_rsp_not_all_visible \
--quads_view_mode=global --quads_render=True --save_video --fps=25 --max_num_episodes=10 \

