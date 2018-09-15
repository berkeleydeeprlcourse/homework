NUM_ROLLOUTS := 50

pip_install:
	pip install -r requirements.txt


behavior_cloning_setup: pip_install
	mkdir -p behavior_cloning_data

behavior_cloning_train_eval: behavior_cloning_setup
	python3 run_behavioral_cloning.py $(env_name) --num_epochs=15 --batch_size=512 > $(EXPORT_FILE) 2>&1

behavior_cloning_data/Hopper-v2.txt: EXPORT_FILE = behavior_cloning_data/Hopper-v2.txt
behavior_cloning_data/Hopper-v2.txt: env_name := Hopper-v2
behavior_cloning_data/Hopper-v2.txt: behavior_cloning_train_eval

behavior_cloning_data/Ant-v2.txt: EXPORT_FILE = behavior_cloning_data/Ant-v2.txt
behavior_cloning_data/Ant-v2.txt: env_name := Ant-v2
behavior_cloning_data/Ant-v2.txt: behavior_cloning_train_eval

behavior_cloning_data/HalfCheetah-v2.txt: EXPORT_FILE = behavior_cloning_data/HalfCheetah-v2.txt
behavior_cloning_data/HalfCheetah-v2.txt: env_name := HalfCheetah-v2
behavior_cloning_data/HalfCheetah-v2.txt: behavior_cloning_train_eval

behavior_cloning_data/Humanoid-v2.txt: EXPORT_FILE = behavior_cloning_data/Humanoid-v2.txt
behavior_cloning_data/Humanoid-v2.txt: env_name := Humanoid-v2
behavior_cloning_data/Humanoid-v2.txt: behavior_cloning_train_eval

behavior_cloning_data/Reacher-v2.txt: EXPORT_FILE = behavior_cloning_data/Reacher-v2.txt
behavior_cloning_data/Reacher-v2.txt: env_name := Reacher-v2
behavior_cloning_data/Reacher-v2.txt: behavior_cloning_train_eval

behavior_cloning_data/Walker2d-v2.txt: EXPORT_FILE = behavior_cloning_data/Walker2d-v2.txt
behavior_cloning_data/Walker2d-v2.txt: env_name := Walker2d-v2
behavior_cloning_data/Walker2d-v2.txt: behavior_cloning_train_eval



gen_expert_data:
	$(MAKE) -B expert_data/Hopper-v2.pkl
	$(MAKE) -B expert_data/Ant-v2.pkl
	$(MAKE) -B expert_data/HalfCheetah-v2.pkl
	$(MAKE) -B expert_data/Humanoid-v2.pkl
	$(MAKE) -B expert_data/Reacher-v2.pkl
	$(MAKE) -B expert_data/Walker2d-v2.pkl


expert_data_dir: pip_install
	mkdir -p expert_data

expert_data/Hopper-v2.pkl: EXPORT_FILE = expert_data/Hopper-v2.txt
expert_data/Hopper-v2.pkl: expert_data_dir
	python run_expert.py experts/Hopper-v2.pkl Hopper-v2 --num_rollouts=$(NUM_ROLLOUTS) > $(EXPORT_FILE) 2>&1

expert_data/Ant-v2.pkl: EXPORT_FILE = expert_data/Ant-v2.txt
expert_data/Ant-v2.pkl: expert_data_dir
	python run_expert.py experts/Ant-v2.pkl Ant-v2 --num_rollouts=$(NUM_ROLLOUTS) > $(EXPORT_FILE) 2>&1

expert_data/HalfCheetah-v2.pkl: EXPORT_FILE = expert_data/HalfCheetah-v2.txt
expert_data/HalfCheetah-v2.pkl: expert_data_dir
	python run_expert.py experts/HalfCheetah-v2.pkl HalfCheetah-v2 --num_rollouts=$(NUM_ROLLOUTS) > $(EXPORT_FILE) 2>&1

expert_data/Humanoid-v2.pkl: EXPORT_FILE = expert_data/Humanoid-v2.txt
expert_data/Humanoid-v2.pkl: expert_data_dir
	python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts=$(NUM_ROLLOUTS) > $(EXPORT_FILE) 2>&1

expert_data/Reacher-v2.pkl: EXPORT_FILE = expert_data/Reacher-v2.txt
expert_data/Reacher-v2.pkl: expert_data_dir
	python run_expert.py experts/Reacher-v2.pkl Reacher-v2 --num_rollouts=$(NUM_ROLLOUTS) > $(EXPORT_FILE) 2>&1

expert_data/Walker2d-v2.pkl: EXPORT_FILE = expert_data/Walker2d-v2.txt
expert_data/Walker2d-v2.pkl: expert_data_dir
	python run_expert.py experts/Walker2d-v2.pkl Walker2d-v2 --num_rollouts=$(NUM_ROLLOUTS) > $(EXPORT_FILE) 2>&1



behavior_cloning_hyperparamter_tune: expert_data_dir behavior_cloning_setup
	python run_expert.py experts/Ant-v2.pkl Ant-v2 --num_rollouts=$(NUM_ROLLOUTS)
	python3 run_behavioral_cloning.py Ant-v2 --num_epochs=15 --batch_size=512 --num_rollouts=5 > behavior_cloning_data/Ant-v2_r$(NUM_ROLLOUTS).txt 2>&1



dagger_dir: pip_install
	mkdir -p dagger_data

run_dagger_experiment:
	python run_dagger.py $(env_name)


gen_behavior_cloning_data_chart:
	mkdir -p behavior_cloning_data/table_data/hopper/expert
	mkdir -p behavior_cloning_data/table_data/hopper/behavior_cloning
	mkdir -p behavior_cloning_data/table_data/ant/expert
	mkdir -p behavior_cloning_data/table_data/ant/behavior_cloning
	$(MAKE) -B expert_data/Hopper-v2.pkl NUM_ROLLOUTS=10 EXPORT_FILE=behavior_cloning_data/table_data/hopper/expert/10
	$(MAKE) -B behavior_cloning_data/Hopper-v2.txt EXPORT_FILE=behavior_cloning_data/table_data/hopper/behavior_cloning/10
	$(MAKE) -B expert_data/Hopper-v2.pkl NUM_ROLLOUTS=20 EXPORT_FILE=behavior_cloning_data/table_data/hopper/expert/20
	$(MAKE) -B behavior_cloning_data/Hopper-v2.txt EXPORT_FILE=behavior_cloning_data/table_data/hopper/behavior_cloning/20
	$(MAKE) -B expert_data/Hopper-v2.pkl NUM_ROLLOUTS=30 EXPORT_FILE=behavior_cloning_data/table_data/hopper/expert/30
	$(MAKE) -B behavior_cloning_data/Hopper-v2.txt EXPORT_FILE=behavior_cloning_data/table_data/hopper/behavior_cloning/30
	$(MAKE) -B expert_data/Hopper-v2.pkl NUM_ROLLOUTS=40 EXPORT_FILE=behavior_cloning_data/table_data/hopper/expert/40
	$(MAKE) -B behavior_cloning_data/Hopper-v2.txt EXPORT_FILE=behavior_cloning_data/table_data/hopper/behavior_cloning/40
	$(MAKE) -B expert_data/Hopper-v2.pkl NUM_ROLLOUTS=50 EXPORT_FILE=behavior_cloning_data/table_data/hopper/expert/50
	$(MAKE) -B behavior_cloning_data/Hopper-v2.txt EXPORT_FILE=behavior_cloning_data/table_data/hopper/behavior_cloning/50
	$(MAKE) -B expert_data/Ant-v2.pkl NUM_ROLLOUTS=10 EXPORT_FILE=behavior_cloning_data/table_data/ant/expert/10
	$(MAKE) -B behavior_cloning_data/Ant-v2.txt EXPORT_FILE=behavior_cloning_data/table_data/ant/behavior_cloning/10
	$(MAKE) -B expert_data/Ant-v2.pkl NUM_ROLLOUTS=20 EXPORT_FILE=behavior_cloning_data/table_data/ant/expert/20
	$(MAKE) -B behavior_cloning_data/Ant-v2.txt EXPORT_FILE=behavior_cloning_data/table_data/ant/behavior_cloning/20
	$(MAKE) -B expert_data/Ant-v2.pkl NUM_ROLLOUTS=30 EXPORT_FILE=behavior_cloning_data/table_data/ant/expert/30
	$(MAKE) -B behavior_cloning_data/Ant-v2.txt EXPORT_FILE=behavior_cloning_data/table_data/ant/behavior_cloning/30
	$(MAKE) -B expert_data/Ant-v2.pkl NUM_ROLLOUTS=40 EXPORT_FILE=behavior_cloning_data/table_data/ant/expert/40
	$(MAKE) -B behavior_cloning_data/Ant-v2.txt EXPORT_FILE=behavior_cloning_data/table_data/ant/behavior_cloning/40
	$(MAKE) -B expert_data/Ant-v2.pkl NUM_ROLLOUTS=50 EXPORT_FILE=behavior_cloning_data/table_data/ant/expert/50
	$(MAKE) -B behavior_cloning_data/Ant-v2.txt EXPORT_FILE=behavior_cloning_data/table_data/ant/behavior_cloning/50



gen_behavior_cloning_hyperparamter_data:
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=1
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=2
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=3
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=4
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=5
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=10
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=15
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=20
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=25
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=30
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=40
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=50
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=60
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=70
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=80
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=90
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=100
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=150
	$(MAKE) behavior_cloning_hyperparamter_tune NUM_ROLLOUTS=200
	

gen_behavior_cloning_data: behavior_cloning_setup gen_expert_data
	$(MAKE) behavior_cloning_data/Hopper-v2.txt
	$(MAKE) behavior_cloning_data/Ant-v2.txt
	$(MAKE) behavior_cloning_data/HalfCheetah-v2.txt
	$(MAKE) behavior_cloning_data/Humanoid-v2.txt
	$(MAKE) behavior_cloning_data/Reacher-v2.txt
	$(MAKE) behavior_cloning_data/Walker2d-v2.txt


gen_dagger_data: dagger_dir
	$(MAKE) run_dagger_experiment env_name=Humanoid-v2
	$(MAKE) run_dagger_experiment env_name=Walker2d-v2
	$(MAKE) run_dagger_experiment env_name=Hopper-v2
	$(MAKE) run_dagger_experiment env_name=Reacher-v2
	$(MAKE) run_dagger_experiment env_name=HalfCheetah-v2
	$(MAKE) run_dagger_experiment env_name=Ant-v2


clean_data:
	rm -rf behavior_cloning_data dagger_data expert_data results.txt

.PHONY: clean_data pip_install behavior_cloning_setup behavior_cloning_train_eval gen_expert_data expert_data_dir behavior_cloning_hyperparamter_tune gen_behavior_cloning_data_chart dagger_dir run_dagger_experiment gen_behavior_cloning_data_chart gen_behavior_cloning_hyperparamter_data gen_behavior_cloning_data gen_dagger_data
