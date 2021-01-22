venv: model_venv/bin/activate

model_venv/bin/activate:
	python3 -m venv model_venv
	. model_venv/bin/activate; pip install pip --upgrade

libhts.a: Dependencies/htslib-1.11
	cd Dependencies/htslib-1.11; chmod +x ./configure ./version.sh
	. model_venv/bin/activate; cd Dependencies/htslib-1.11; ./configure CFLAGS=-fpic && make

gpu: venv requirements.txt libhts.a generate_features.cpp models.cpp gen.cpp setup.py
	. model_venv/bin/activate; pip install -r requirements.txt;
	. model_venv/bin/activate; python3 setup.py build_ext; python3 setup.py install
	rm -rf build

cpu: venv requirements_cpu.txt libhts.a generate_features.cpp models.cpp gen.cpp setup.py
	. model_venv/bin/activate; pip install -r requirements_cpu.txt; pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html; pip install pytorch-lightning;
	. model_venv/bin/activate; python3 setup.py build_ext; python3 setup.py install
	rm -rf build

clean:
	cd Dependencies/htslib-1.11 && make clean || exit 0
	rm -rf model_venv
