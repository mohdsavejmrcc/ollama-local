"""
URL configuration for chatapi project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# from django.urls import path
# from myapi import views
# urlpatterns = [
#     path('query/', views.query_document, name='query_document'),
#     path('process/', views.process_document, name='process_document'),
# ]
# from django.urls import path
# from myapi import views

# urlpatterns = [
#     path("process_query/", views.process_query, name="process_query"),
# ]
from django.http import HttpResponseRedirect
from django.urls import path
from myapi import views

urlpatterns = [
    path("", lambda request: HttpResponseRedirect("/process_query/")),  # Redirect base URL
    path("process_query/", views.process_query, name="process_query"),
     path("process_query_stream/", views.stream_query_view,name='stream_query_view')
]
