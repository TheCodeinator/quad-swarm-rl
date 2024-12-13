python -m swarm_rl.enjoy \
--algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 \
--quads_use_numba=True --train_dir=./train_dir --experiment=train_rnn_small_test \
--quads_view_mode=global --quads_render=True --save_video --fps=25 --max_num_episodes=10 \
--visualize_v_value --generate_quantization_samples

