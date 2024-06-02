from django.contrib import admin
from django.urls import include
from django.urls import path

from django.conf.urls.static import static
from django.conf import settings

from disease_detection.views import page_not_found_view


from django.views.static import serve


# from django.conf.urls import url

# from django.conf import settings

# from django.views.static import serve

from django.urls import re_path

urlpatterns = [
    re_path(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}),
    re_path(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),
    re_path(r'^model_dict/(?P<path>.*)$', serve,{'document_root': settings.MODEL_ROOT}),





    path('admin/', admin.site.urls),
    path('', include('disease_detection.urls')),

    



]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


from django.conf.urls import handler404

handler404 = 'disease_detection.views.page_not_found_view'


# if settings.DEBUG:
#     urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

urlpatterns = urlpatterns + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)