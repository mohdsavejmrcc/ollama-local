"""
ASGI config for chatapi project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os
import django
import socketio
from django.core.asgi import get_asgi_application
from socket_server import sio
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatapi.settings')
django.setup()

django_app = get_asgi_application()
application = socketio.ASGIApp(sio, django_app)
