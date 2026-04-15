#!/bin/bash
PORT=${PORT:-5000}
exec gunicorn --bind 0.0.0.0:$PORT app:app
