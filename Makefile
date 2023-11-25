install: venv
	. venv/bin/activate;pip install -Ur requirements.txt
venv:
	test -d venv || python3 -m venv venv
run1:
	. venv/bin/activate;python3 Classifier_1.py
run2:
	. venv/bin/activate;python3 Classifier_2.py
run3:
	. venv/bin/activate;python3 Classifier_3.py
run4:
	. venv/bin/activate;python3 Classifier_4.py

clean:
	rm -rf venv
	find -iname "*.pyc" -delete
