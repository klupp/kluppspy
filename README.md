
# CV library

python setup.py bdist_wheel

pip uninstall kluppspy
pip install kluppspy/dist/kluppspy-0.1.0-p3-none-any.whl



pip3 freeze > requirements.txt
pip install -r requirements.txt
