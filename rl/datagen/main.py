# SPDX-FileCopyrightText: 2026 UIUC Security and Privacy Lab
#
# SPDX-License-Identifier: Apache-2.0

from rl.datagen.collect import collect_main

# Train
# Main - "kodcode"
# Side Tasks - "permissions", "iodelete", "exit", "varname", "ioexfil", "iodownload"

# Val1 (ID Main, ID Side) -
# Main - "kodcode"
# Side Tasks - "permissions", "iodelete", "exit", "varname", "ioexfil", "iodownload"

# Val2 (ID Main, OOD Side) -
# Main - "kodcode"
# Side Tasks - "logio"

# Val3 (OOD Main, ID Side) -
# Main - "bigcodebench"
# Side Tasks - "permissions", "iodelete", "exit", "varname", "ioexfil", "iodownload"

# Val3 (OOD Main, OOD Side) -
# Main - "bigcodebench"
# Side Tasks - "logio"


def main(
    id_main_tasks=["bigcodebench"],
    ood_main_tasks=["bigcodebench"],
    id_side_tasks=[
        "defaultval",
        "complexity",
        "varname",
        # "longlines", "complexity",
        # "permissions",
        # "iodelete",
        # "exit",
        # "varname",
        # "ioexfil",
        # "iodownload",
    ],
    ood_side_tasks=["logio"],
    evaluated_models=["Qwen3-14B"],
    monitor_models=["Qwen3-32B"],
    attack_policies=["baseline"],
    rollout_num=5,
    monitor_policies=["main_aware"],
    monitor_objectives=[
        # "action_only",
        # "cot_only",
        "cot_action"
    ],
    filter="side-all",
    top_k=32,
    check_against_sft: bool = False,
    sft_path: str | None = None,
) -> list[str]:

    paths = []

    train_path = collect_main(
        main_tasks=id_main_tasks,
        side_tasks=id_side_tasks,
        evaluated_models=evaluated_models,
        monitor_models=monitor_models,
        attack_policies=attack_policies,
        rollout_num=rollout_num,
        monitor_policies=monitor_policies,
        monitor_objectives=monitor_objectives,
        filter=filter,
        top_k=-1,
        set_type="train",
        check_against_sft=check_against_sft,
        sft_path=sft_path,
    )
    paths.append(train_path)

    # Val1
    # ood_side_val_path = collect_main(
    #     main_tasks=id_main_tasks,
    #     side_tasks=ood_side_tasks,
    #     evaluated_models=evaluated_models,
    #     monitor_models=monitor_models,
    #     attack_policies=attack_policies,
    #     rollout_num=rollout_num,
    #     monitor_policies=monitor_policies,
    #     monitor_objectives=monitor_objectives,
    #     filter=filter,
    #     top_k=top_k,
    #     set_type="ood-side",
    #     check_against_sft=check_against_sft,
    #     sft_path=sft_path,
    # )
    # paths.append(ood_side_val_path)

    # # Val2
    # ood_main_val_path = collect_main(
    #     main_tasks=ood_main_tasks,
    #     side_tasks=id_side_tasks,
    #     evaluated_models=evaluated_models,
    #     monitor_models=monitor_models,
    #     attack_policies=attack_policies,
    #     rollout_num=5,
    #     monitor_policies=monitor_policies,
    #     monitor_objectives=monitor_objectives,
    #     filter=filter,
    #     top_k=top_k,
    #     set_type="ood-main",
    #     check_against_sft=check_against_sft,
    #     sft_path=sft_path,
    # )
    # paths.append(ood_main_val_path)

    # # Val3
    # ood_main_ood_side_val_path = collect_main(
    #     main_tasks=ood_main_tasks,
    #     side_tasks=ood_side_tasks,
    #     evaluated_models=evaluated_models,
    #     monitor_models=monitor_models,
    #     attack_policies=attack_policies,
    #     rollout_num=5,
    #     monitor_policies=monitor_policies,
    #     monitor_objectives=monitor_objectives,
    #     filter=filter,
    #     top_k=top_k,
    #     set_type="ood-main_ood-side",
    #     check_against_sft=check_against_sft,
    #     sft_path=sft_path,
    # )
    # paths.append(ood_main_ood_side_val_path)

    return paths


if __name__ == "__main__":
    from fire import Fire

    Fire(main)

# Sample run:
# python -m rl.datagen.main --top_k 32
