
from django.contrib import admin
from django.urls import path,include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('app1/',include('django.contrib.auth.urls')),
    path('', include("app1.urls")),
]
