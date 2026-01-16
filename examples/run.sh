#############################################
# interactive session
#############################################

BEAKER_USERNAME_CAPS=$(beaker account whoami | tail -1 | awk '{print $2}' | tr [:lower:] [:upper:])
beaker session create \
--budget ai2/prior \
--mount weka://oe-training-default=/weka/oe-training-default \
--mount weka://prior-default=/weka/prior \
--bare \
--workspace ai2/robo-molmo \
--gpus 1 \
--priority high \
--secret-env BEAKER_TOKEN="${BEAKER_USERNAME_CAPS}_BEAKER_TOKEN" \
--secret-env AWS_ACCESS_KEY_ID="${BEAKER_USERNAME_CAPS}_AWS_ACCESS_KEY_ID" \
--secret-env AWS_SECRET_ACCESS_KEY="${BEAKER_USERNAME_CAPS}_AWS_SECRET_ACCESS_KEY" \
--secret-env GITHUB_TOKEN="${BEAKER_USERNAME_CAPS}_GITHUB_TOKEN" \
--env AWS_DEFAULT_REGION=us-west-2 \
--env PYTHONPATH=/weka/prior/ainaze/code/robo_mm_olmo


#############################################
# preprocess the datasets
#############################################

cd /weka/prior/ainaze/code/robo_mm_olmo/
curl -LsSf https://astral.sh/uv/install.sh | sh
export MOLMO_DATA_DIR=/weka/oe-training-default/mm-olmo

PYTHONPATH=. uv run python olmo/data/robot_datasets.py --datasets ObjectNavSceneMemoryObjectPointingRoomCount ObjectNavDoneEvalSceneMemoryObjectPointingRoomCount HardObjectNavSceneMemoryObjectPointingRoomCount HardObjectNavDoneEvalSceneMemoryObjectPointingRoomCount \
--include_eval 

PYTHONPATH=. uv run python olmo/data/robot_datasets.py --datasets ExploreHouseSceneMemorySceneDescriptionRoomCount ExploreHouseSceneMemoryDoneEvalSceneDescriptionRoomCount \
--include_eval


#############################################
# train the model
#############################################

python examples/ae_run_gantry.py \
"torchrun --nproc-per-node 8 launch_scripts/train_multitask_model.py robot-multitask /weka/oe-training-default/roseh/molmo_pretrained_checkpoints/Molmo-7B-D-0924-Pretrained --robot_memory_setting SceneMemory --robot_prompt_style scene_description --robot_done_behavior ObjectPointing --robot_room_count_behavior RoomCount" \
--name stretch_ft_vit_open_type \
--group robomolmo \
--preemptible


#############################################
# upload to modal
#############################################

cd /weka/prior/ainaze/code/robo_mm_olmo/
curl -LsSf https://astral.sh/uv/install.sh | sh
export MOLMO_DATA_DIR=/weka/oe-training-default/mm-olmo

pip install modal
modal setup

(do the token flow)
python scripts/serving/deploy_molmo.py --checkpoint-dir /weka/oe-training-default/ainaze/mm_olmo/robomolmo_checkpoints/robomolmo/new_procthor_houses_fpin_ft_vit/step6000 --model-name robo-scenemem-fpin-ft-vit-6000
python scripts/serving/deploy_molmo.py --checkpoint-dir /weka/oe-training-default/ainaze/mm_olmo/robomolmo_checkpoints/robomolmo/new_procthor_houses_fpin_no_ft_vit_v2/step6000 --model-name robo-scenemem-fpin-no-ft-vit-6000
python scripts/serving/deploy_molmo.py --checkpoint-dir /weka/oe-training-default/ainaze/mm_olmo/robomolmo_checkpoints/robomolmo/new_procthor_houses_stretch_ft_vit/step6000 --model-name robo-scenemem-stretch-ft-vit-6000
python scripts/serving/deploy_molmo.py --checkpoint-dir /weka/oe-training-default/ainaze/mm_olmo/robomolmo_checkpoints/robomolmo/new_procthor_houses_stretch_no_ft_vit_v2/step6000 --model-name robo-scenemem-stretch-no-ft-vit-6000
python scripts/serving/deploy_molmo.py --checkpoint-dir /weka/oe-training-default/ainaze/mm_olmo/robomolmo_checkpoints/robomolmo/stretch_ft_vit_open_type/step9000 --model-name stretch-ft-vit-open-type-9000


