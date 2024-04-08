#!/bin/bash

# uncomment all the tasks that you want to download the data for
# note: some tasks's datasets are much large than others due to compression differences
download_tasks=(
    bigfish
    # bossfight
    # caveflyer
    # chaser
    # climber
    # coinrun
    # dodgeball
    # fruitbot
    # heist
    # jumper
    # leaper
    # maze
    # miner
    # ninja
    # plunder
    # starpilot
)

# maps task names to google drive file ids
declare -A task_to_file_id=(
    ["bigfish"]="10xwWejz1ZwccxV7O4fqbDTP_260OsBU5"
    ["bossfight"]="11x3-8eh0M4KsnDYW3pVRHjlOIGVCeuMz"
    ["caveflyer"]="1YdSG3Y3Tyf-UnL4I-ZXGGlJjOIdungU3"
    ["chaser"]="17c8NyuT-_IU7ymZYh3qGy_zOgm3XwcHb"
    ["climber"]="1qiomp1WF_KYEqbL5SWgXhZOLoaFYjoJl"
    ["coinrun"]="14Dfs1xy4u2IOfzLISm8adTA5IwtCsAo7"
    ["dodgeball"]="1q7DZkTwys0yWEWWkd7ddGczTSwGyJVXG"
    ["fruitbot"]="1zJw1MNxeSxjeS78Btzqo3c6oq4kXqpKH"
    ["heist"]="1FMCKSxCOQFjNOlUOHpARK9u0w0BFaXlx"
    ["jumper"]="1lv1QGv06AJMClX_Iyw9HdXUEiHgnIWX3"
    ["leaper"]="1S8pR6heQVGixbJ0f1u4kKPji1m481l2a"
    ["maze"]="18eeqTAZg1OeyN3qQFdyC-Y0l-6niM5Me"
    ["miner"]="177IBL8ByPBlyCKCc-1jbiwVZklc_2ga0"
    ["ninja"]="10sXxsoOU2Jcczu91dio7ScpZIAo6sGGP"
    ["plunder"]="1KzaDi7D8cc_k7M1JyrWtkk88kaS_upRy"
    ["starpilot"]="1AH6fO7zIp_SQYxJS0aoc81OA-UkWFp1W"
)


for i in "${!download_tasks[@]}"; do
    task=${download_tasks[$i]}
    file_id=${task_to_file_id[$task]}
    echo "---> [$i/${#download_tasks[@]}] Downloading *$task* (id=$file_id)"

    gdown --continue --output lapo/expert_data/$task.zip $file_id
    unzip -q -o lapo/expert_data/$task.zip -d lapo/expert_data && rm lapo/expert_data/$task.zip
done