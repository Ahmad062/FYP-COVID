from django.db import models
from .validators import file_size
# Create your models here.

class Video(models.Model):
    class Meta:
        get_latest_by = 'id'
    caption=models.CharField(max_length=100)
    video=models.FileField(upload_to="video/%y",validators = [file_size])
    def __str__(self):
        return self.caption

