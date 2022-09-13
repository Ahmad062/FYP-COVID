from django.contrib import admin
from django.urls import path, include 
from fypapp import views

urlpatterns = [
    path('',views.index,name='fypapp'),
    path('login',views.login,name='login'),
    path('logout',views.logoutuser,name='logout'),
    path('about',views.about,name='about'),
    path('contact',views.contact,name='contact'),
    path('sops',views.sops,name='sops'),
    path('faqs',views.faqs,name='faqs'),
    path('camera',views.selection),
    path('video',views.selection),
    path('temp',views.temperature),
    # path('vidtest',views.vidtest),
]

