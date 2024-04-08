#!/bin/bash

declare -A tasks=(
	[0]="bigfish"
	[1]="bossfight"
	[2]="caveflyer"
	[3]="chaser"
	[4]="climber"
	[5]="coinrun"
	[6]="dodgeball"
	[7]="fruitbot"
	[8]="heist"
	[9]="jumper"
	[10]="leaper"
	[11]="maze"
	[12]="miner"
	[13]="ninja"
	[14]="plunder"
	[15]="starpilot"
)

sweep_name=$(python -c "import doy; print(doy.random_proquint(2))")

for ind in {0..15}; do
	# generate a random experiment name that's the same across stages 1-3
	exp_name="${ind}_${sweep_name}"
	python stage1_idm.py env_name="${tasks[${ind}]}" exp_name="${exp_name}" &&
	python stage2_bc.py env_name="${tasks[${ind}]}" exp_name="${exp_name}" &&
	python stage3_decoding.py env_name="${tasks[${ind}]}" exp_name="${exp_name}"
done