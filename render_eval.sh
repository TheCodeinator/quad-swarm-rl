python -m swarm_rl.enjoy \
--algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 \
--quads_use_numba=True --train_dir=./train_dir --experiment=neuralfly_rnn \
--quads_view_mode=global --quads_render=True \

