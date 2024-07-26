from . import views 
from django.urls import path

urlpatterns=[
    path("",views.home,name="home"),
    path("register/",views.register,name="register"),
    path("login/",views.login_user,name="login"),
    path("chat/",views.chat,name="chat"),
    path('specific',views.specific,name='specific'),
    path('getResponse',views.getResponse,name='getResponse'),
    path('get_intent_response',views.getResponse,name='get_intent_response'),
    path('get_gpt_response',views.getResponse,name='get_gpt_response'),
    path('get_combined_response',views.getResponse,name='get_combined_response'),

]