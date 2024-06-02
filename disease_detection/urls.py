# from django.contrib import admin
from django.urls import path
from . import views

#at now for forget password
from django.contrib.auth import views as auth_views

from .views import activate, image_upload_view


urlpatterns = [
    # path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('logo_redirect/', views.logo_redirect, name='logo_redirect'),
    path('login_page/', views.login_page, name='login_page'),
    path('signup_page/', views.signup_page, name='signup_page'),
    path('signup/', views.signup, name="signup"),
    path('login/', views.login, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('home/', views.home, name ='home'),

    path('image_upload/', views.image_upload_view, name='image_upload'),
    path('history/', views.history, name='history'),


    path('take_photo/', views.take_photo, name='take_photo'),
    # path('image_capture/', views.image_capture, name='image_capture'),



    path('pnf_back/', views.pnf_back, name='pnf_back'),




    path('password_reset/', auth_views.PasswordResetView.as_view(), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(), name='password_reset_complete'),




    path('activate/<uidb64>/<token>/', activate, name='activate'),


]


