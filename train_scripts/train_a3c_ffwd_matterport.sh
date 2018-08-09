 #!/bin/sh

xvfb-run -a -s "-ac -screen 0 1280x1024x24" python main.py --env_type indoor --env_name pointgoal_mp3d_s --nouse_pixel_change --nouse_value_replay --nouse_lstm  --nouse_reward_prediction false --checkpoint_dir "/local/mircea/checkpoints/a3cff_mp3d_small_pointgoal" --log_dir "/local/mircea/logs/ac3ff_mp3d_small_pointgoal"
