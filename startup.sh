cd /home/site/wwwroot
pip install -r requirements.txt
gunicorn app.main:app --workers 1 --threads 4 --timeout 120 -b 0.0.0.0:8000