develop:
	rm -f *.so
	python setup.py clean --all
	python setup.py develop