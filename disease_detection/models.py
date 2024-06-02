

from django.db import models
from django.contrib.auth.models import User
from django.utils.timezone import now
from django.utils.crypto import get_random_string


class DiseasePrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='disease_images')
    predicted_disease = models.CharField(max_length=100)
    treatment = models.CharField(max_length=200)
    how_to_use = models.TextField()
    caution = models.TextField(default='')
    created_at = models.DateTimeField(auto_now_add=True)
    verified = models.BooleanField(default=False)
    verification_token = models.CharField(max_length=32, blank=True)

    def save(self, *args, **kwargs):
        if not self.pk and not self.verification_token:
            self.verification_token = get_random_string(length=32)
        super().save(*args, **kwargs)

    def is_verified(self):
        return self.verified

    def verify(self, token):
        if self.verification_token == token:
            self.verified = True
            self.save()
            return True
        else:
            return False
